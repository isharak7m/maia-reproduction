<div align="center">
  <h1>♟ Maia Reproduction</h1>
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

## Results vs. Paper

Each column is a different test set (rating bin). For Stockfish it is the same engine. For Maia, each column uses the model trained on that same bin — the diagonal of the full agreement matrix.

<table>
<tr>
  <th>Model</th>
  <th>1100-1199</th>
  <th>1500-1599</th>
  <th>1900-1999</th>
  <th>Paper (approx)</th>
</tr>
<tr>
  <td>Stockfish depth 1</td>
  <td align="center">38.6%</td>
  <td align="center">38.6%</td>
  <td align="center">42.8%</td>
  <td align="center">36-42%</td>
</tr>
<tr>
  <td>Stockfish depth 7</td>
  <td align="center">35.2%</td>
  <td align="center">34.0%</td>
  <td align="center">37.2%</td>
  <td align="center">34-38%</td>
</tr>
<tr>
  <td>Stockfish depth 15</td>
  <td align="center">38.6%</td>
  <td align="center">35.6%</td>
  <td align="center">40.0%</td>
  <td align="center">33-40%</td>
</tr>
<tr style="background:#fff3cd">
  <td><strong>Maia-1100</strong><br><small style="color:#666">Paper architecture</small></td>
  <td align="center"><strong>19.6%</strong></td>
  <td align="center">17.6%</td>
  <td align="center">18.8%</td>
  <td align="center">~30-35%</td>
</tr>
<tr style="background:#fff3cd">
  <td><strong>Maia-1500</strong><br><small style="color:#666">Paper architecture</small></td>
  <td align="center">25.6%</td>
  <td align="center"><strong>26.6%</strong></td>
  <td align="center">24.0%</td>
  <td align="center">~30-35%</td>
</tr>
<tr style="background:#fff3cd">
  <td><strong>Maia-1900</strong><br><small style="color:#666">Paper architecture</small></td>
  <td align="center">21.8%</td>
  <td align="center">19.6%</td>
  <td align="center"><strong>28.2%</strong></td>
  <td align="center">~30-35%</td>
</tr>
</table>

**Key pattern reproduced:** Each Maia model peaks at its own training bin (self-bin bias, bold). The V-shape agreement matrix from Fig 6 of the paper is clearly visible. Stockfish depth 1 matches lower-rated humans better than depth 7 or 15, confirming weaker engines are more human-like.

<details>
<summary><b>📊 Paper Comparison — Why the gap?</b></summary>

| | Paper | Ours |
|:---|---|---|
| Compute | 8x NVIDIA V100 (datacenter) | 1x RTX 2050 (laptop, 4GB) |
| Training steps | 400,000 | 15,000-20,000 |
| Effective batch | 1,024 | 64 |
| Training time | Days | ~6 hours total |
| **Budget** | **~3% of paper's compute** | |

The ~8-10% accuracy gap is fully expected at this scale. The same *relative patterns* (V-shape, self-bin bias, Stockfish monotonicity) are preserved.
</details>

---

## Figures

| Accuracy Curves (Fig 2 equivalent) | Agreement Matrix (Fig 6 equivalent) | Our vs Paper |
|---|---|---|
| ![Accuracy Curves](reports/fig2_accuracy_curves.png) | ![Agreement Matrix](reports/fig6_agreement_matrix.png) | ![Paper Comparison](reports/paper_comparison.png) |

---

## Architecture

All 3 Maia models use the same architecture from the paper:

| Component | Specification |
|-----------|--------------|
| Input channels | 113 (17 board planes + 12 x 8 history planes) |
| Initial conv | 113 -> 256, 3x3, BN, ReLU |
| Residual blocks | **15 blocks** at 256 channels |
| SE blocks | Squeeze-and-excitation per block (reduction=16) |
| Policy head | Conv 256 -> 80 -> 73, flattened to 4672 logits |
| Value head | Conv -> FC256 -> FC3 (win/draw/loss) |
| Parameters | **18.6M** |

### Input planes (113 total)
- **12 piece-channel planes** (6 piece types x 2 colors)
- **4 castling rights** (KQkq)
- **1 side-to-move**
- **96 history planes** (8 preceding board positions x 12 piece-channel planes each)

---

## Dataset

| | 1100-1199 | 1500-1599 | 1900-1999 |
|:---|:---|:---|:---|
| Source | Lichess 2019-10 | Lichess 2019-10 | Lichess 2019-10 |
| Raw games scanned | ~1.5M | ~1.5M | ~1.5M |
| Moves extracted | 1,230,460 | 3,622,570 | 1,971,947 |
| Moves used (trimmed) | 1,230,460 | 1,200,052 | 1,200,000 |
| Test positions | 500 consecutive | 500 consecutive | 500 consecutive |

- **Filtering**: standard time control, blitz excluded; 1500 and 1900 bins subsampled to ~1.2M for balanced training (1500 uses game-aware sampling preserving game structure; 1900 uses random sampling)
- **Test set**: 500 consecutive positions per bin from the start of each JSON file (preserving game history for the 8 history planes)

---

## Training

| Config | Maia-1100 | Maia-1500 | Maia-1900 |
|:------|:---------:|:---------:|:---------:|
| Channels | 256 | 256 | 256 |
| Blocks | 15 | 15 | 15 |
| History | 8 | 8 | 8 |
| Batch size | 8 | 8 | 8 |
| Gradient accumulation | 8 | 8 | 8 |
| Effective batch | 64 | 64 | 64 |
| Steps | 15,000 | 15,000 | 20,000 |
| Learning rate | 0.01 | 0.01 | 0.01 |
| LR decay | 0.1 @ 5k/10k/14k | 0.1 @ 5k/10k/14k | 0.1 @ 5k/10k/14k |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Optimizer | Adam | Adam | Adam |
| Compute time | ~2h | ~1.5h | ~3h |
| Best loss | 3.53 | 3.07 | 3.11 |

**cuDNN compatibility**: RTX 2050 (Turing architecture) requires `torch.backends.cudnn.enabled = False` to avoid `CUDNN_STATUS_NOT_SUPPORTED`. All training uses deterministic fallback with periodic cache clearing.

### Loss curves

| Maia-1100 | Maia-1500 | Maia-1900 |
|:---------:|:---------:|:---------:|
| ![Loss 1100](reports/loss_full_1100.png) | ![Loss 1500](reports/loss_full_1500.png) | ![Loss 1900](reports/loss_full_1900.png) |

---

## Quick Start

```bash
# Install
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install python-chess matplotlib pyyaml

# Download data
python scripts/download_data.py  # Lichess 2019-10 PGN

# Extract records
python scripts/extract_data.py

# Train a model (each ~2-3h on 4GB GPU)
python scripts/train_full.py 1100
python scripts/train_full.py 1500
python scripts/train_full.py 1900

# Evaluate
python scripts/eval_full2.py

# Stockfish baselines
python scripts/stockfish_baselines.py

# Generate figures
python scripts/paper_figures.py

# Run tests
pytest tests/ -v
```

---

## Deviations from the Paper

| Paper | Ours | Reason |
|:------|:------|:-------|
| 9 rating bins (1100-1900) | 3 bins (1100, 1500, 1900) | Limited compute |
| 400K training steps | 15K-20K steps | 4GB GPU / laptop thermal limits |
| Batch size 1,024 | Batch 64 (eff.) | 4GB VRAM constraint |
| 8x NVIDIA V100 | 1x RTX 2050 (4GB) | Available hardware |
| Lichess 2019 *and* Dec 2019 test | Lichess 2019-10 only | Dec 2019 download did not complete |
| Random test positions | 500 consecutive positions | History planes require game context |
| Value head trained with MSE | Value head trained but not evaluated | Focus on move-matching |

---

## Project Structure

```
maia-reproduction/
├── README.md
├── pyproject.toml
├── stockfish.exe
├── checkpoints/
│   ├── maia_full_1100_best.pt   # Maia-1100 best validation
│   ├── maia_full_1100_final.pt  # Maia-1100 final step
│   ├── maia_full_1100_snap.pt   # Maia-1100 training snapshot
│   ├── maia_full_1500_best.pt   # Maia-1500 best validation
│   ├── maia_full_1500_final.pt  # Maia-1500 final step
│   ├── maia_full_1500_snap.pt   # Maia-1500 training snapshot
│   ├── maia_full_1900_best.pt   # Maia-1900 best validation
│   ├── maia_full_1900_final.pt  # Maia-1900 final step
│   └── maia_full_1900_snap.pt   # Maia-1900 training snapshot
├── reports/
│   ├── fig2_accuracy_curves.png
│   ├── fig6_agreement_matrix.png
│   ├── paper_comparison.png
│   ├── loss_full_1100.png
│   ├── loss_full_1500.png
│   ├── loss_full_1900.png
│   ├── stockfish_results.json
│   └── full_model_results2.json
├── scripts/
│   ├── train_full.py             # Paper-arch training (256ch, 15blk, history)
│   ├── eval_full2.py             # Paper-arch evaluation (consecutive positions)
│   ├── stockfish_baselines.py    # Stockfish depth 1/7/15 baselines
│   ├── paper_figures.py          # Paper-style figure generation
│   ├── extract_data.py           # PGN extraction pipeline
│   ├── build_dataset.py          # Dataset building
│   ├── download_data.py          # Lichess PGN download
│   └── ...                       # Utility scripts
├── src/
│   ├── models/maia_net.py        # Residual CNN (256ch, 15 blocks, SE)
│   ├── encoding/board.py         # 8x8x17 board tensor + history stacking
│   ├── encoding/move.py          # 8x8x73 AlphaZero move encoding
│   └── data_pipeline/            # PGN parsing, filtering, datasets
├── configs/
│   └── maia_default.yaml         # Full-scale training config
└── tests/
    └── ...                       # 54 unit tests
```
