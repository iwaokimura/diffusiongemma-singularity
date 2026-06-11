# DiffusionGemma + vLLM on Singularity CE

Google の拡散型言語モデル **DiffusionGemma 26B-A4B-it** を、
**Singularity CE** 環境の Ubuntu GPU サーバーで動かすための一式です。

## ファイル構成

```
.
├── diffusiongemma.def              # Singularity 定義ファイル (ビルドレシピ)
├── infer.py                        # 推論クライアントスクリプト
├── README.md
└── .github/
    └── workflows/
        └── build-singularity.yml  # GitHub Actions: ビルド & ghcr.io push
```

## GitHub Actions によるイメージビルド

`main`/`master` ブランチに push すると、自動的に:

1. `diffusiongemma.def` から SIF ファイルをビルド
2. `ghcr.io/<あなたのGitHubユーザー名>/diffusiongemma-vllm:latest` に push

### 必要な設定

- **リポジトリの Packages 設定**: `Settings > Actions > General > Workflow permissions` を
  `Read and write permissions` に変更する（または `GITHUB_TOKEN` に `packages: write` を許可）
- **HuggingFace Token**: モデルのダウンロードにトークンが必要な場合は
  `Settings > Secrets and variables > Actions > New repository secret` で
  `HF_TOKEN` を登録し、後述の実行コマンドで `--env HF_TOKEN=...` として渡す

## Ubuntu サーバーでの使い方

### 1. SIF ファイルを取得

```bash
# GitHub Container Registry から pull (ビルド済みの場合)
singularity pull \
  oras://ghcr.io/<あなたのGitHubユーザー名>/diffusiongemma-vllm:latest

# または、ローカルでビルド (要 root or --fakeroot)
singularity build --fakeroot diffusiongemma.sif diffusiongemma.def
```

### 2. vLLM サーバー起動

```bash
singularity run --nv \
  -B ~/.cache/huggingface:/cache/huggingface \
  diffusiongemma.sif
```

| オプション | 意味 |
|---|---|
| `--nv` | NVIDIA GPU を Singularity コンテナに渡す（必須） |
| `-B ~/.cache/huggingface:/cache/huggingface` | ホストの HF キャッシュをバインド（モデル再ダウンロード不要に） |

デフォルトモデルは **NVFP4 量子化版**
(`RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`、重み約 14GB) です。
RTX 4090 など 24GB GPU 単体での動作を想定しています。
BF16 フルモデル (`google/diffusiongemma-26B-A4B-it`) は重みだけで約 52GB
あるため 24GB GPU には載りません。

なお RTX 4090 (Ada) には NVFP4 のネイティブカーネルがないため、
vLLM の Marlin フォールバック (weight-only) で実行されます。
Blackwell GPU でのネイティブ実行より遅くなりますが動作します。

#### VRAM が十分な場合（60GB+）: BF16 フルモデル

```bash
singularity run --nv \
  -B ~/.cache/huggingface:/cache/huggingface \
  diffusiongemma.sif \
  --model google/diffusiongemma-26B-A4B-it
```

#### HuggingFace Token が必要な場合

```bash
singularity run --nv \
  --env HUGGING_FACE_HUB_TOKEN=hf_xxxxx \
  -B ~/.cache/huggingface:/cache/huggingface \
  diffusiongemma.sif
```

#### ポートを変える場合

```bash
singularity run --nv \
  -B ~/.cache/huggingface:/cache/huggingface \
  diffusiongemma.sif --port 8080
```

### 3. 推論 (別ターミナル)

```bash
# 基本
python infer.py --prompt "Why is the sky blue?"

# Thinking モード (数学の問題などに有効)
python infer.py --thinking --prompt "Prove that sqrt(2) is irrational."

# 長めの回答を生成
python infer.py --max-tokens 1024 --prompt "Explain Iwasawa theory briefly."
```

### 4. シェルに入る

```bash
singularity shell --nv diffusiongemma.sif
```

## GPU / VRAM の目安

| GPU | 精度 | 備考 |
|---|---|---|
| B200 / Blackwell | NVFP4 (ネイティブ) | RedHatAI のベンチマーク環境 |
| H100 / H200 (80GB+) | BF16 / FP8 | フルモデルが載る |
| RTX 4090 (24GB) | NVFP4 (Marlin フォールバック) | デフォルト構成の想定環境 |
| 24GB 未満の GPU | - | NVFP4 でも KV キャッシュ分が不足する可能性あり |

## トラブルシューティング

### `CUDA not available` と出る

`--nv` オプションが抜けていないか確認してください。

### OOM (Out of Memory) / KV キャッシュ不足

デフォルトは 24GB GPU 向けに `--max-model-len 32768` です。
それでも不足する場合はさらに下げるか、`--gpu-memory-utilization 0.7` を
併用してください:
```bash
singularity run --nv -B ~/.cache/huggingface:/cache/huggingface \
  diffusiongemma.sif --max-model-len 16384 --gpu-memory-utilization 0.7
```
VRAM の大きい GPU では `--max-model-len 262144` まで上げられます
(モデルの最大コンテキスト長)。

### `VLLM_USE_V2_MODEL_RUNNER` の警告

DiffusionGemma は ModelRunner v2 が必須です。コンテナ内では自動的に
`VLLM_USE_V2_MODEL_RUNNER=1` が設定されています。

## ライセンス

- DiffusionGemma モデル: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- この設定ファイル一式: MIT
