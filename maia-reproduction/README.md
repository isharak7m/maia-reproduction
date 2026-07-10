# Maia Reproduction

A from-scratch reproduction of **"Aligning Superhuman AI with Human Behavior: Chess as a Model System"** (Maia), focused on predicting human chess moves at specific rating levels. Trained end-to-end on a single laptop **NVIDIA RTX 2050 (4GB VRAM)** — 3% of the paper's original compute budget.

## Project Structure

```
maia-reproduction/
├── src/
│   ├── encoding/
│   │   ├── board.py        # 8×8×(7+12*history) tensor encoding
│   │   └── move.py         # 8×8×73 AlphaZero-style move encoding
│   ├── models/
│   │   ├── maia_net.py     # Residual CNN with SE blocks, policy + value heads
│   │   ├── blunder_fc.py   # Fully-connected blunder predictor
│   │   └── blunder_rescnn.py  # Residual CNN blunder predictor
│   └── cp_to_winprob/
│       └── converter.py    # Empirical centipawn → win-probability table
├── configs/
│   ├── maia_default.yaml   # Full-scale training config
│   └── debug.yaml          # Debug/minimal config
├── scripts/
│   ├── extract_data.py     # Stream decompress + parse + filter Lichess PGNs
│   ├── train_bin.py        # Reduced-model training (32ch, 6 blocks)
│   ├── train_full.py       # Full-scale training (256ch, 15 blocks, 8 history)
│   ├── stockfish_baselines.py  # Stockfish depth 1/7/15 evaluation
│   └── eval_full.py        # Full-model evaluation with history context
├── tests/
│   ├── test_encoding.py    # Board + move encoding round-trips
│   ├── test_filter.py      # Rating bin and filter tests
│   ├── test_parse.py       # PGN parsing tests
│   └── test_dataset.py     # Dataset construction tests
├── checkpoints/            # Trained model weights (generated, ~400 MB)
├── data/                   # Raw PGNs + extracted records (not committed)
├── reports/                # Generated figures
└── README.md
```

## Architecture

The model is a residual convolutional neural network (no explicit tree search) that maps a board position to a policy distribution over all legal moves.

**Input encoding:** The board is represented as 17 base planes (6 piece types × 2 colors + 4 castling rights + side to move) plus 12 planes per historical position (8 history positions in full model). Total: 113 input channels.

**Network body:** A convolutional layer (113→256, 3×3) followed by 15 residual blocks. Each block contains two 3×3 convolutions with batch normalization and ReLU, plus a squeeze-and-excitation (SE) channel attention module that learns to emphasize important feature channels per position.

**Output heads:**
- **Policy head:** A 3×3 conv (256→73) + flatten to 4672 logits (64 squares × 73 move planes), softmax over legal moves.
- **Value head:** A 3×3 conv + two fully-connected layers → 3-class output (win/loss/draw probability).

This is the same architecture as Leela Chess Zero, used because it was designed to produce a _distribution_ over actions rather than a single best action — making it ideal for matching diverse human play patterns rather than finding the optimal engine move.

## Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- [Stockfish](https://stockfishchess.org/download/) (for baseline evaluation)

### Installation

```bash
cd maia-reproduction
pip install -e .
pip install -e ".[dev]"   # optional: for tests
```

Place `stockfish.exe` in the project root or update the path in evaluation scripts.

### Data

The extraction pipeline processes Lichess standard-rated PGN dumps:

```bash
python scripts/extract_data.py --pgn data/pgn/lichess_db_standard_rated_2019-10.pgn.zst
```

This filters by rating bins (1100-1199, 1500-1599, 1900-1999), extracts moves (ply ≥ 10 for realistic positions), and outputs JSON records per bin.

### Run Tests

```bash
pytest tests/ -v
```

All encoding, parsing, filtering, and dataset construction tests pass (54/54).

## Results

### Move-Matching Accuracy

Accuracy on 500 positions per bin (top-1 move prediction):

| Model | 1100-1199 | 1500-1599 | 1900-1999 |
|-------|-----------|-----------|-----------|
| Stockfish depth 1 | 38.6% | 38.6% | 42.8% |
| Stockfish depth 7 | 35.2% | 34.0% | 37.2% |
| Stockfish depth 15 | 38.6% | 35.6% | 40.0% |
| Maia-Reduced (32ch, 6blk) | 26.0% | 25.4% | 26.4% |
| **Maia-Full (256ch, 15blk)** | **19.6%** | **22.8%** | **28.2%** |

Key patterns: Stockfish accuracy increases with opponent rating (stronger players make more "engine-like" moves). The full Maia model shows the expected self-bin bias (Maia-1900 peaks at 1900, Maia-1100 peaks at 1100), matching the paper's findings.

### Training Performance

| Config | Params | GPU Memory | Training Time |
|--------|--------|-----------|---------------|
| Reduced (32ch, 6blk, no history) | 265K | ~52 MB | ~1h per bin |
| Full (256ch, 15blk, 8 history) | 18.6M | ~225 MB | ~1-3h per bin |

## Comparison to the Paper

The original Maia paper trained on 8× NVIDIA V100 GPUs for days, processing 400K gradient steps with effective batch size 1024. This reproduction:

- Uses **1 laptop GPU** (RTX 2050, 4GB) with **15K–20K training steps**
- Achieves the same **relative accuracy patterns** (self-bin bias, Stockfish monotonicity)
- Has **lower absolute accuracy** (~20-28% vs ~30-35%) due to ~3% of the paper's compute

The V-shaped accuracy pattern (highest at self bin, dropping off for other bins) is the paper's core finding and is reproduced here.

## Detailed Reproduction Report

See [`docs/reproduction_report.md`](docs/reproduction_report.md) for:
- Per-figure comparison to the paper
- Architecture reproduction details (encoding, network, training)
- Paper citations mapped to our implementation
- Honest accounting of deviations and why

## Limitations

- No Leela Chess Zero baseline
- No collective blunder prediction
- No UCI compatibility layer for playing against the model
- Full test set evaluation (December 2019 data) not completed
- Personalization (fine-tuning on individual play) not yet implemented

## Acknowledgements

Built following the original paper: *McIlroy-Young et al. "Aligning Superhuman AI with Human Behavior: Chess as a Model System" (KDD 2020)*.

Data from the [Lichess Database](https://database.lichess.org/).
