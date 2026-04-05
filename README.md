# patent-paper-intelligence

論文・特許データのAI自動ラベリング・分類システム（POC）

## できること

- 論文・特許PDFを読み込み、意味的なトピックを自動検出
- Embeddingによるベクトル化 → HDBSCANクラスタリング
- Claude API（LLM）によるタグ・キーワード自動生成
- 事前定義カテゴリへの自動マッピング（コサイン類似度）

## 技術スタック

| 役割 | 技術 |
|---|---|
| PDF読み込み | PyMuPDF（fitz） |
| Embedding | multilingual-e5-small（日英対応） |
| クラスタリング | HDBSCAN |
| タグ生成 | Claude API（Anthropic） |
| 分類マッピング | コサイン類似度 |

## 実行結果サンプル

NTTファシリティーズの空調・電力系論文5本を処理した結果：
- 22トピックを自動検出
- 電力最適化・センサー異常検知・空調制御等に自動分類
- 分類精度：コサイン類似度 80〜89%

## セットアップ
```bash
pip install pymupdf sentence-transformers hdbscan \
            scikit-learn numpy pandas anthropic python-dotenv
```
```bash
# .envファイルを作成
ANTHROPIC_API_KEY=sk-ant-...
```
```bash
python sample_v3.py
```

## アーキテクチャ
```
論文・特許PDF
     ↓
テキスト抽出・チャンク分割（300文字・100文字オーバーラップ）
     ↓
Embedding（multilingual-e5-small / 384次元）
     ↓
HDBSCANクラスタリング（トピック自動検出）
     ↓
Claude APIでタグ・キーワード生成
     ↓
コサイン類似度で事前定義カテゴリにマッピング
```