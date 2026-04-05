"""
論文・特許 AIラベリング v2
─────────────────────────
① 本文テキスト → チャンク分割
② Embedding → ベクトル化
③ HDBSCAN → トピック自動抽出（10〜20個）
④ Claude API → タグ・キーワード生成（APIキーなしはフォールバック）
⑤ 定義済み分類へのマッピング
"""

import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

# ── APIキー（あれば設定）──────────────────────────
from dotenv import load_dotenv
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── 事前定義した分類カテゴリ ──────────────────────
PREDEFINED_CATEGORIES = {
    "A": "テナント満足度・サービス品質",
    "B": "データ基盤・データ整備",
    "C": "マルチバリアントセンサー異常検知",
    "D": "設備劣化予測・メンテナンス",
    "E": "地震後建物リスク評価",
    "F": "電力異常検知・エネルギー最適化",
    "G": "インシデントテキスト分析",
}

# ── サンプル論文・特許テキスト（本文想定）────────
SAMPLE_TEXTS = [
    """
    本研究では、オフィスビルにおける空調システムの予測制御手法を提案する。
    従来のPID制御と異なり、機械学習モデルを用いて室内温度・湿度・CO2濃度を
    予測し、エネルギー消費を最小化しながら快適性を維持する。
    実験結果では、従来手法と比較して電力消費量を23%削減することに成功した。
    HVACシステムへの適用において、センサーデータのリアルタイム処理が重要である。
    """,
    """
    本特許は、建物設備の劣化予測システムに関する。
    振動センサー・電流センサーから取得したデータを多変量時系列解析にかけ、
    ポンプ・ファン・コンプレッサーの異常を事前に検出する。
    提案手法では、正常状態のベースラインからのマハラノビス距離を用いて
    劣化スコアを算出し、メンテナンスの最適タイミングを予測する。
    フィールドテストでは故障の92%を30日前に検出できた。
    """,
    """
    オフィスワーカーの満足度調査データを自然言語処理で分析した。
    自由記述回答をBERTでベクトル化し、不満トピックをクラスタリングで抽出した。
    温熱環境・照明・騒音が主要な不満要因であることが定量的に示された。
    テナント満足度スコアと空調設定値の相関分析から、
    個別ゾーン制御が満足度向上に有効であることが示唆される。
    """,
    """
    大規模地震発生後の建物安全性を自動評価するシステムを開発した。
    加速度センサーの応答データ・構造健全性モニタリング情報を組み合わせ、
    損傷レベルをリアルタイムで推定する。
    深層学習モデルによって建物の再入場可否を自動判定し、
    人手による点検の優先順位付けを支援する。
    東日本大震災のデータを用いた検証では精度87%を達成した。
    """,
    """
    ビルの電力消費データから異常パターンを検出する手法を提案する。
    正常時の消費プロファイルを時間帯・季節・曜日でモデル化し、
    実測値との乖離をリアルタイムで監視する。
    電力の無駄遣いや設備故障に起因する異常を早期発見することで、
    エネルギーコストの削減と設備の予防保全を同時に実現する。
    """,
    """
    設備保全インシデントレポートをテキストマイニングで分析した。
    形態素解析・TF-IDFによる特徴抽出後、LDAトピックモデリングで
    故障パターンを自動分類した。
    過去5年分の3万件のレポートから、空調・電気・給排水系統の
    頻出故障パターンを抽出し、予防保全計画の策定に活用した。
    """,
    """
    建物データ基盤の整備に関する研究である。
    複数ベンダーのBMSから取得される異種データを統合するETLパイプラインを構築した。
    データ品質管理・欠損値補完・外れ値検出を自動化し、
    分析基盤としてのデータレイクを整備した。
    標準化されたデータスキーマにより、施設横断的な分析が可能となった。
    """,
    """
    複数センサーの相関構造を利用した異常検知手法を提案する。
    温度・湿度・電力・流量センサーの正常時の相関をグラフィカルラッソで
    モデル化し、相関構造の変化を異常として検出する。
    単一センサー監視と比較して、複合的な異常の検出精度が大幅に向上した。
    特にHVACシステムの複合故障検出に有効であることが示された。
    """,
]


# ─────────────────────────────────────────────────
# STEP 1: チャンク分割
# ─────────────────────────────────────────────────
def split_chunks(texts: list[str], chunk_size: int = 200) -> list[dict]:
    chunks = []
    for doc_id, text in enumerate(texts):
        # 段落単位で分割
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]
        for para_id, para in enumerate(paragraphs):
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"doc{doc_id}_p{para_id}",
                "text": para,
            })
    return chunks


# ─────────────────────────────────────────────────
# STEP 2: Embedding
# ─────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> np.ndarray:
    print("  モデル読み込み中...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    prefixed = ["passage: " + t for t in texts]
    print(f"  {len(texts)}件をベクトル化中...")
    return model.encode(prefixed, normalize_embeddings=True, show_progress_bar=True)


# ─────────────────────────────────────────────────
# STEP 3: HDBSCANクラスタリング
# ─────────────────────────────────────────────────
def cluster(embeddings: np.ndarray) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  → {n}個のトピックを検出")
    return labels


# ─────────────────────────────────────────────────
# STEP 4: タグ生成（Claude API or フォールバック）
# ─────────────────────────────────────────────────
def generate_tags_claude(texts: list[str]) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    sample = "\n".join(texts[:3])
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""以下のテキスト群を読んで、このトピックを表す情報をJSONのみで返してください。
前後の説明は不要です。

テキスト:
{sample}

出力形式:
{{"topic_name": "トピック名（15文字以内）", "keywords": ["キーワード1", "キーワード2", "キーワード3"]}}"""
        }]
    )
    return json.loads(msg.content[0].text)


def generate_tags_fallback(texts: list[str]) -> dict:
    """APIキーなし時：頻出単語からキーワードを抽出"""
    all_text = " ".join(texts)
    # 簡易キーワード抽出（2文字以上の単語）
    words = re.findall(r'[ぁ-んァ-ン一-龥a-zA-Z]{2,}', all_text)
    stopwords = {"する", "した", "ある", "いる", "れる", "られ", "ため", "おい", "より", "から", "まで"}
    freq: dict = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    top_kw = sorted(freq, key=freq.get, reverse=True)[:3]
    return {"topic_name": top_kw[0] if top_kw else "不明", "keywords": top_kw}


def generate_tags(texts: list[str]) -> dict:
    if ANTHROPIC_API_KEY:
        try:
            return generate_tags_claude(texts)
        except Exception as e:
            print(f"    Claude APIエラー: {e} → フォールバック")
    return generate_tags_fallback(texts)


# ─────────────────────────────────────────────────
# STEP 5: 定義済みカテゴリへのマッピング
# ─────────────────────────────────────────────────
def map_to_category(topic_texts: list[str], model: SentenceTransformer) -> str:
    """トピックのテキストと定義済みカテゴリの類似度でマッピング"""
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
    best_score = float(sims[best_idx])
    return best_key, best_score


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────
def main():
    use_api = bool(ANTHROPIC_API_KEY)
    print(f"\n{'='*50}")
    print(f"  論文・特許 AIラベリング v2")
    print(f"  Claude API: {'✅ 使用' if use_api else '❌ 未設定（フォールバック）'}")
    print(f"{'='*50}\n")

    print("=== STEP 1: チャンク分割 ===")
    chunks = split_chunks(SAMPLE_TEXTS)
    texts = [c["text"] for c in chunks]
    print(f"  {len(SAMPLE_TEXTS)}文書 → {len(chunks)}チャンクに分割")

    print("\n=== STEP 2: Embedding ===")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    embeddings = model.encode(
        ["passage: " + t for t in texts],
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("\n=== STEP 3: HDBSCANクラスタリング ===")
    labels = cluster(embeddings)

    print("\n=== STEP 4 & 5: タグ生成 + カテゴリマッピング ===")
    results = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        idxs = [i for i, l in enumerate(labels) if l == cluster_id]
        topic_texts = [texts[i] for i in idxs]

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
        })

    # 結果表示
    print(f"\n{'='*50}")
    print(f"  検出トピック数: {len(results)}")
    print(f"{'='*50}")
    for r in results:
        print(f"\n📌 トピック {r['cluster_id']}: {r['topic_name']}")
        print(f"   キーワード : {', '.join(r['keywords'])}")
        print(f"   → 分類    : {r['mapped_category']}")
        print(f"   　類似度  : {r['similarity_score']:.1%}  ({r['chunk_count']}チャンク)")

    # JSON保存
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ results.json に保存しました")


if __name__ == "__main__":
    main()