"""
論文・特許 AIラベリング v3 - PDF実データ版
"""

import os
import re
import json
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

from dotenv import load_dotenv
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PDF_DIR = "/workspace/論文/"

# 事前定義カテゴリ（Masaの仮説A-G）
PREDEFINED_CATEGORIES = {
    "A": "テナント満足度・サービス品質分析",
    "B": "データ基盤・データ整備・ETL",
    "C": "マルチバリアントセンサー異常検知",
    "D": "設備劣化予測・メンテナンス最適化",
    "E": "地震後建物リスク自動評価",
    "F": "電力異常検知・エネルギー最適化",
    "G": "インシデントテキスト分析・故障分類",
}

# ─────────────────────────────────────────
# STEP 1: PDF読み込み・テキスト抽出
# ─────────────────────────────────────────
def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\w\s\u3000-\u9fff\u30a0-\u30ff。、．，]', ' ', text)
    return text.strip()

def load_pdfs(pdf_dir: str) -> list[dict]:
    docs = []
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        print(f"  読み込み中: {fname}")
        raw = extract_pdf_text(path)
        text = clean_text(raw)
        docs.append({
            "filename": fname,
            "text": text,
            "char_count": len(text),
        })
        print(f"    → {len(text):,}文字抽出")
    return docs

# ─────────────────────────────────────────
# STEP 2: チャンク分割
# ─────────────────────────────────────────
def split_chunks(docs: list[dict], chunk_size: int = 300, stride: int = 100) -> list[dict]:
    chunks = []
    for doc in docs:
        text = doc["text"]
        start = 0
        chunk_id = 0
        while start < len(text):
            chunk = text[start:start + chunk_size].strip()
            if len(chunk) > 50:
                chunks.append({
                    "filename": doc["filename"],
                    "chunk_id": f"{doc['filename']}_{chunk_id:03d}",
                    "text": chunk,
                })
                chunk_id += 1
            start += chunk_size - stride
    return chunks

# ─────────────────────────────────────────
# STEP 3: Embedding
# ─────────────────────────────────────────
def embed_chunks(chunks: list[dict], model) -> np.ndarray:
    texts = ["passage: " + c["text"] for c in chunks]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

# ─────────────────────────────────────────
# STEP 4: HDBSCANクラスタリング
# ─────────────────────────────────────────
def cluster(embeddings: np.ndarray, min_cluster_size: int = 3) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).sum()
    print(f"  → {n}トピック検出（ノイズ: {noise}チャンク）")
    return labels

# ─────────────────────────────────────────
# STEP 5: タグ生成
# ─────────────────────────────────────────
def generate_tags_claude(texts: list[str]) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    sample = "\n".join(texts[:3])[:1000]
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": f"""以下のテキスト群のトピックを分析してください。

テキスト:
{sample}

必ずこの形式のJSONだけを返してください。他の文章は不要です:
{{"topic_name": "トピック名（15文字以内）", "keywords": ["キーワード1", "キーワード2", "キーワード3"]}}"""}]
    )
    raw = msg.content[0].text.strip()
    # JSONブロックが```で囲まれている場合の対処
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    print(f"    APIレスポンス確認: {raw[:80]}...")
    return json.loads(raw)

def generate_tags_fallback(texts: list[str]) -> dict:
    all_text = " ".join(texts)
    words = re.findall(r'[ァ-ン一-龥a-zA-Z]{2,}', all_text)
    stopwords = {"する", "した", "ある", "いる", "れる", "ため", "おい", "より", "から", "まで", "および", "おける"}
    freq: dict = {}
    for w in words:
        if w not in stopwords and len(w) >= 2:
            freq[w] = freq.get(w, 0) + 1
    top = sorted(freq, key=freq.get, reverse=True)[:5]
    return {"topic_name": top[0] if top else "不明", "keywords": top}

def generate_tags(texts: list[str]) -> dict:
    if ANTHROPIC_API_KEY:
        try:
            return generate_tags_claude(texts)
        except Exception as e:
            print(f"    Claude APIエラー: {e} → フォールバック")
    return generate_tags_fallback(texts)

# ─────────────────────────────────────────
# STEP 6: カテゴリマッピング
# ─────────────────────────────────────────
def map_to_category(topic_texts: list[str], model) -> tuple[str, float]:
    topic_emb = model.encode(
        ["passage: " + " ".join(topic_texts[:3])],
        normalize_embeddings=True
    )
    cat_emb = model.encode(
        ["passage: " + v for v in PREDEFINED_CATEGORIES.values()],
        normalize_embeddings=True
    )
    sims = cosine_similarity(topic_emb, cat_emb)[0]
    best_idx = int(np.argmax(sims))
    best_key = list(PREDEFINED_CATEGORIES.keys())[best_idx]
    return best_key, float(sims[best_idx])

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    use_api = bool(ANTHROPIC_API_KEY)
    print(f"\n{'='*55}")
    print(f"  論文・特許 AIラベリング v3 - PDF実データ版")
    print(f"  Claude API: {'✅ 使用' if use_api else '❌ 未設定（フォールバック）'}")
    print(f"{'='*55}\n")

    print("=== STEP 1: PDF読み込み ===")
    docs = load_pdfs(PDF_DIR)
    print(f"  合計 {len(docs)} 文書読み込み完了\n")

    print("=== STEP 2: チャンク分割 ===")
    chunks = split_chunks(docs)
    print(f"  {len(docs)}文書 → {len(chunks)}チャンクに分割\n")

    print("=== STEP 3: Embedding ===")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    embeddings = embed_chunks(chunks, model)
    print()

    print("=== STEP 4: クラスタリング ===")
    labels = cluster(embeddings)
    print()

    print("=== STEP 5 & 6: タグ生成 + カテゴリマッピング ===")
    results = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        idxs = [i for i, l in enumerate(labels) if l == cluster_id]
        topic_texts = [chunks[i]["text"] for i in idxs]
        topic_files = list(set(chunks[i]["filename"] for i in idxs))

        tags = generate_tags(topic_texts)
        cat_key, score = map_to_category(topic_texts, model)
        cat_name = PREDEFINED_CATEGORIES[cat_key]

        results.append({
            "cluster_id": int(cluster_id),
            "topic_name": tags["topic_name"],
            "keywords": tags["keywords"],
            "mapped_category": f"{cat_key}: {cat_name}",
            "similarity_score": float(round(score, 3)),
            "chunk_count": len(idxs),
            "source_files": topic_files,
        })

    # 結果表示
    print(f"\n{'='*55}")
    print(f"  検出トピック数: {len(results)}")
    print(f"{'='*55}")
    for r in results:
        print(f"\n📌 トピック {r['cluster_id']}: 【{r['topic_name']}】")
        print(f"   キーワード  : {', '.join(r['keywords'])}")
        print(f"   → 分類     : {r['mapped_category']}")
        print(f"   　類似度   : {r['similarity_score']:.1%}（{r['chunk_count']}チャンク）")
        print(f"   　出典論文 : {', '.join(r['source_files'])}")

    with open("/workspace/results_v3.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ results_v3.json に保存しました")

if __name__ == "__main__":
    main()