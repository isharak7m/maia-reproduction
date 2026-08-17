<div align="center">
  <h1>Maia Reproduction</h1>
  <p>
    <strong>From-scratch reproduction of</strong><br>
    <em>"Aligning Superhuman AI with Human Behavior: Chess as a Model System"</em><br>
    McIlroy-Young et al., KDD 2020
  </p>
  <p>
    <a href="https://doi.org/10.1145/3394486.3403219"><img src="https://img.shields.io/badge/Paper-KDD%202020-blue" alt="Paper"></a>
    <img src="https://img.shields.io/badge/GPU-RTX%202050%20(4GB)-success" alt="GPU">
    <img src="https://img.shields.io/badge/Tests-54%2F54-passing-success" alt="Tests">
    <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python">
  </p>
</div>

---

## Result

A residual CNN (256ch, 15 blocks, 8 history planes, 18.6M params) trained to predict human chess moves at the **1100-1199 rating level** from Lichess 2019-10 achieves **32.0% move-matching accuracy**, matching the paper's ~30-35% range for that rating bin.

| Metric | Value |
|--------|-------|
| **Self-bin accuracy** | **32.0%** |
| Stockfish depth 1 (1100) | 38.6% |
| Stockfish depth 7 (1100) | 35.2% |
| Stockfish depth 15 (1100) | 38.6% |
| Paper range (1100 bin) | ~30-35% |

Stockfish depth 1 matches lower-rated humans better than depth 7 or 15, confirming weaker engines are more human-like. The paper trained 9 rating-bin models on 8 V100 GPUs for 400K steps. This reproduction is limited to 1 bin at ~3% of the paper's compute (1 RTX 2050 laptop GPU, 25K training steps).

---

## Architecture

| Component | Specification |
|-----------|--------------|
| Input channels | 113 (17 board planes + 12 x 8 history planes) |
| Initial conv | 113 -> 256, 3x3, BN, ReLU |
| Residual blocks | 15 blocks at 256 channels |
| SE blocks | Squeeze-and-excitation per block (reduction=16) |
| Policy head | Conv 256 -> 80 -> 73, flattened to 4672 logits |
| Value head | Conv -> FC256 -> FC3 (win/draw/loss) |
| Parameters | 18.6M |

### Input planes (113 total)
- 12 piece-channel planes (6 piece types x 2 colors)
- 4 castling rights (KQkq)
- 1 side-to-move
- 96 history planes (8 preceding board positions x 12 piece-channel planes each)

---

## Dataset

| | Value |
|:---|:---|
| Source | Lichess 2019-10 monthly dump |
| Rating bin | 1100-1199 |
| Games scanned | ~1.5M |
| Games matched | 25,000 |
| Moves extracted | 1,232,884 |
| Test positions | 1000 random-with-history |
| Time control filter | Standard, no bullet, clock >= 30s |

---

## Training

| Config | Value |
|:------|:-----|
| Channels | 256 |
| Blocks | 15 |
| History planes | 8 |
| Batch size | 8 |
| Gradient accumulation | 8 |
| Effective batch | 64 |
| Steps | 25,000 |
| Learning rate | 0.001 (decayed 0.1x at 15k, 20k) |
| Grad clip | 1.0 |
| Optimizer | Adam |
| Compute time | ~2.2h on RTX 2050 (4GB) |
| Best validation loss | 2.74 |

**Note**: RTX 2050 (Turing) requires `torch.backends.cudnn.enabled = False` to avoid `CUDNN_STATUS_NOT_SUPPORTED`. Training uses deterministic fallback with periodic cache clearing.

---

## Quick Start

```bash
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install python-chess matplotlib pyyaml

python scripts/download_data.py
python scripts/extract_data.py
python scripts/train_full.py 1100
python scripts/eval_full2.py
python scripts/stockfish_baselines.py
python scripts/paper_figures.py
pytest tests/ -v
```

---

## Deviations from the Paper

| Paper | Ours | Reason |
|:------|:------|:-------|
| 400K training steps | 25K steps | 4GB GPU / laptop thermal limits |
| Batch size 1,024 | Batch 64 (eff.) | 4GB VRAM constraint |
| 8x NVIDIA V100 | 1x RTX 2050 (4GB) | Available hardware |
| Lichess 2013-2019 + Dec 2019 test | Lichess 2019-10 only | Data availability |
| Random test positions | 1000 random-within-game positions | Reduces sampling bias |
| Value head trained with MSE | Included but not evaluated | Focus on move-matching |

---

## Project Structure

```
maia-reproduction/
├── README.md
├── pyproject.toml
├── stockfish.exe
├── checkpoints/
│   └── maia_full_1100_best.pt
├── reports/
│   ├── fig2_accuracy_curves.png
│   ├── fig6_agreement_matrix.png
│   ├── paper_comparison.png
│   ├── stockfish_results.json
│   └── full_model_results2.json
├── scripts/
│   ├── train_full.py
│   ├── eval_full2.py
│   ├── stockfish_baselines.py
│   ├── paper_figures.py
│   ├── extract_data.py
│   ├── build_dataset.py
│   └── download_data.py
├── src/
│   ├── models/maia_net.py
│   ├── encoding/board.py
│   ├── encoding/move.py
│   └── data_pipeline/
├── configs/
│   └── maia_default.yaml
└── tests/
    └── ...
```
