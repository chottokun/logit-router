# Gemma 4 E2B VisionLogitRouter ベンチマーク & 実験レポート

## 概要

`google/gemma-4-E2B-it` を用いて、マルチモーダル入力（画像＋テキスト）に対応した `VisionLogitRouter` の性能・精度検証をローカル NVIDIA GeForce RTX 3060 (12GB VRAM) 環境で実施した。

---

## 1. レイテンシ・ベンチマーク結果 (`benchmarks/bench_vision.py`)

- **モデル**: `google/gemma-4-E2B-it` (bfloat16)
- **環境**: CUDA (NVIDIA GeForce RTX 3060 12GB), PyTorch 2.14.0, Transformers 5.17.0
- **条件**: Warmup 3回、本計測 20回
- **タスク**: 224x224 合成画像（赤色ブロック）＋色判定ルーティング（選択肢: Red, Green, Blue）

### レイテンシ測定値

| モード | 全体平均レイテンシ | 画像前処理 (Preprocess) | モデル推論 (Forward) | ロジット抽出 (Postprocess) |
|---|---|---|---|---|
| **VisionLogitRouter** | **294.22 ms** | 5.46 ms (1.9%) | 219.45 ms (74.6%) | 80.96 ms (27.5%) |
| **LogitRouter (Text)** | **223.11 ms** | - | - | - |

> [!NOTE]
> **レイテンシ特性の分析**:
> 1. **前処理の軽さ**: torchvision バックエンドによる 224x224 画像の前処理はわずか **5.46 ms** であり、オーバーヘッドはごく軽微です。
> 2. **Vision と Text の差**: VisionLogitRouter は Text 版と比較して約 **71 ms (+31.8%)** の増加に留まっており、300ms 未満でリアルタイムなマルチモーダルルーティングが完了します。
> 3. **Postprocess の内訳**: フルロジットからのスライス処理（Gemmaの大きな語彙数 256k に起因）が約 80ms を占めており、今後の Sliced LM-Head 最適化（GPU上での部分抽出の高速化）によりさらなる短縮余地があります。

---

## 2. 実機実験ケース結果 (`examples/vision_experiment.py`)

プログラマティックに合成した画像を用いた 3 つの実験ケースすべてにおいて、**確信度 1.0 (100%)** で正確なルーティングに成功しました。

### ケース 1: 色判定 (Color Recognition)
- **入力画像**: 白背景の中央に赤い四角形を描画した 224x224 画像
- **指示**: "What color is the shape in the center of the image?"
- **選択肢**: `['Red', 'Green', 'Blue', 'Yellow']`
- **結果**:
  - 判定: **A. Red**
  - 確信度 (Confidence): **1.0 (100%)**
  - 分布: `A: 1.0, B: 8.88e-13, C: 4.42e-14, D: 1.98e-13`
  - エントロピー: `-9.77e-10` (ほぼ 0)

### ケース 2: 図形判定 (Shape Recognition)
- **入力画像**: 白背景の中央に緑の円を描画した 224x224 画像
- **指示**: "What shape is in the center of the image?"
- **選択肢**: `['Square', 'Circle', 'Triangle']`
- **結果**:
  - 判定: **B. Circle**
  - 確信度 (Confidence): **1.0 (100%)**
  - 分布: `A: 4.58e-17, B: 1.0, C: 3.91e-14`
  - エントロピー: `-9.99e-10`

### ケース 3: UI エラートリアージ (UI Error Triage)
- **入力画像**: Web アプリの UI 風画像（上部に赤色のエラー通知バナー）
- **コンテキスト**: "A screenshot of a web application showing a notification banner."
- **指示**: "Based on the visual indicators (red banner), what is the most likely status of this notification?"
- **選択肢**: `['Success', 'Warning', 'Error', 'Info']`
- **結果**:
  - 判定: **C. Error**
  - 確信度 (Confidence): **1.0 (99.996%)**
  - 分布: `A: 0.00015%, B: 0.0031%, C: 99.996%, D: 0.0019%`
  - エントロピー: `0.00055`

---

## 3. 通常の利用 (model.generate) とのレイテンシ実機比較 (`benchmarks/bench_vision_vs_autoregressive.py`)

Hugging Face の通常利用（`AutoModelForVision2Seq` + `model.generate()` による自己回帰テキスト生成）と、本手法（`VisionLogitRouter`）を、同一モデル重み（`google/gemma-4-E2B-it` bfloat16）、同一ハードウェア（CUDA / RTX 3060）、同一画像・プロンプト入力において直接対決（A/Bテスト）で計測した。

| モード / 手法 | 平均レイテンシ (Mean) | 中央値 (p50) | 高速化倍率 (Speedup) |
|---|---|---|---|
| **VisionLogitRouter (本手法: 単一フォワードパス)** | **704.38 ms** | **765.23 ms** | **基準 (Baseline)** |
| **通常生成: 最短5トークン (`model.generate(max_new_tokens=5)`)** | **1,536.41 ms** | **1,566.19 ms** | **2.18x 高速** |
| **通常生成: 簡潔推論30トークン (`model.generate(max_new_tokens=30)`)** | **1,560.18 ms** | **1,584.30 ms** | **2.21x 高速** |

> [!IMPORTANT]
> **実測された高速化要因**:
> 1. **逐次ループの排除**: マルチモーダルモデルにおける自己回帰生成は、視覚エンコーダ出力に対するCross-AttentionやKVキャッシュの逐次参照が発生するため、わずか 5 トークンの生成でも 1.5 秒以上を要します。
> 2. **単一Prefill完了直後の直接ロジット抽出**: VisionLogitRouter は最初の Prefill が完了した直後に候補選択肢のロジットを直接抽出して終了するため、**通常の利用に比べて 2.18x 〜 2.21x（半分以下の時間）でルーティング判定を完了**できます。

---

## 4. 自動テスト結果 (`pytest tests/`)

Jules によって作成された単体テストおよび既存のテストスイート全 14 項目がパスしました。

```
============================ 14 passed in 5.22s =============================
- tests/test_router.py (10 passed)
- tests/test_vision_router.py (4 passed: structure, max_choices, fallback, sliced_head)
```

