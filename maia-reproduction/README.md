<div align="center">
  <h1>Maia Partial Reproduction</h1>
  <p>
    <strong>Partial reproduction of</strong><br>
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

A residual CNN (256ch, 15 blocks, 8 history planes, 18.6M params) trained to predict human chess moves at the **1100-1199 rating level** from Lichess 2019-10 achieves **32.0% move-matching accuracy** on its own rating bin (1,000 test positions, roughly ±3 points at 95% confidence). This is below the paper's reported 46-52%, which is expected given the much smaller training set and compute budget.

### Self-bin bias (unimodal peak)

The core finding of the Maia paper: a model trained on a specific rating bin performs **best on that bin** and worse on higher-rated bins, rather than becoming a stronger overall chess predictor.

| Evaluated on | Accuracy | 95% CI |
|:-------------|:---------|:-------|
| **1100-1199 (self-bin)** | **32.0%** | ±3.0% |
| 1500-1599 | 28.6% | ±2.8% |
| 1900-1999 | 25.2% | ±2.7% |

The model peaks at its own training bin (32.0% > 28.6% > 25.2%), consistent with learning **rating-specific moves** rather than general chess strength. Note that the gaps between bins (3.4 and 6.8 percentage points) are barely outside the confidence intervals, so these differences should be interpreted cautiously.

### Stockfish baselines (1100-1199 positions, same 1,000 positions as Maia eval)

| Engine | Depth | Accuracy |
|:-------|:------|:---------|
| Stockfish | 1 | 38.6% |
| Stockfish | 7 | 35.2% |
| Stockfish | 15 | 38.6% |

**Note**: The identical accuracy for depth 1 and depth 15 (38.6%) is suspicious and may indicate an issue with the Stockfish binary or evaluation setup (e.g., depth parameter not being applied correctly, or the bundled Stockfish version not respecting depth limits). This should be re-run with a freshly downloaded Stockfish to verify. Additionally, Stockfish outscoring the Maia model (38.6% vs 32.0%) contradicts the paper's finding that Maia predicts human moves better than Stockfish. This likely points to undertraining of our model given the limited compute.

> The paper trained 9 rating-bin models on 8 V100 GPUs for 400K steps with batch size 1,024 (~410M training samples, ~12M games per bin). This reproduction is limited to 1 bin using ~0.2% of the paper's training samples (1 RTX 2050 laptop GPU, 15K steps × batch 64 = 960K samples, 25K games).

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
| Train/test split | **Potential leakage**: training splits 98/2 by game (seed 42), but eval samples from all games without excluding training games |

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
| Steps | 15,000 |
| Learning rate | 0.01 (decayed 0.1x at 5k, 10k, 14k) |
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

# Stockfish baselines (requires Stockfish installed or in PATH)
# Windows: https://stockfishchess.org/download/
# Linux: sudo apt install stockfish
# macOS: brew install stockfish
python scripts/stockfish_baselines.py

python scripts/paper_figures.py
pytest tests/ -v
```

---

## Deviations from the Paper

| Paper | Ours | Reason |
|:------|:------|:-------|
| 400K training steps | 15K steps | 4GB GPU / laptop thermal limits |
| Batch size 1,024 | Batch 64 (eff.) | 4GB VRAM constraint |
| 8x NVIDIA V100 | 1x RTX 2050 (4GB) | Available hardware |
| ~12M games per rating bin | 25K games | Data availability |
| ~6 residual blocks | 15 blocks | Paper uses smaller architecture |
| Lichess 2013-2019 + Dec 2019 test | Lichess 2019-10 only | Data availability |
| Random test positions | 1000 random-within-game positions | Reduces sampling bias |
| Value head trained with MSE | Included but not evaluated | Focus on move-matching |

---

## Project Structure

```
maia-reproduction/
├── README.md
├── pyproject.toml
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

---

## Limitations

This is a **partial reproduction** with several important limitations:

1. **Training set size**: We used 25K games per bin vs. the paper's ~12M games per bin (0.2% of the paper's data).
2. **Compute budget**: We used ~0.2% of the paper's training samples (960K vs. ~410M).
3. **Architecture difference**: We used 15 residual blocks vs. the paper's ~6 blocks, resulting in a larger model (18.6M params).
4. **Accuracy gap**: Our model achieves 32.0% vs. the paper's 46-52% move-matching accuracy.
5. **Stockfish baseline issue**: The Stockfish evaluation shows suspicious results (depth 1 = depth 15 accuracy) that need verification with a freshly downloaded Stockfish binary.
6. **Stockfish outperforms model**: Our model is outperformed by Stockfish, contradicting the paper's finding.
7. **Confidence intervals**: With only 1,000 test positions, the confidence intervals (±3%) are large relative to the differences between bins.
8. **Single bin**: We only trained on the 1100-1199 bin, while the paper trained on 9 bins.
9. **Train/test leakage**: The evaluation script samples test positions from all games, including those used in training. Training splits 98/2 by game internally, but the eval does not exclude training games. This may inflate reported accuracy.
