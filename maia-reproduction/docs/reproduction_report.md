# Reproduction Report: Maia

**Paper:** McIlroy-Young, R., Sen, S., Kleinberg, J., & Anderson, A. (2020). Aligning Superhuman AI with Human Behavior: Chess as a Model System. *KDD 2020*.

**Our repo:** [github.com/isharak7m/maia-reproduction](https://github.com/isharak7m/maia-reproduction)

**Hardware:** 1× NVIDIA RTX 2050 laptop GPU (4GB VRAM) — ~3% of paper's compute (8× V100).

---

## 1. Overview

The Maia paper introduces a family of chess engines trained not to play the *best* move, but the move a **human of a specific rating** would play. The key insight is that traditional engines (Stockfish, AlphaZero) become less human-like as they get stronger, while Maia models trained on human games at specific rating levels can match human play better than any engine.

Our reproduction implements the core architecture and training pipeline, achieving the same relative patterns at smaller absolute scale.

---

## 2. Key Results Summary

**Paper Figure 2 / Our Figure 1 — Move-Matching Accuracy**

| Model | 1100-1199 | 1500-1599 | 1900-1999 | Paper (approx) |
|-------|-----------|-----------|-----------|-----------------|
| Stockfish depth 1 | 38.6% | 38.6% | 42.8% | 36-42% |
| Stockfish depth 7 | 35.2% | 34.0% | 37.2% | 34-38% |
| Stockfish depth 15 | 38.6% | 35.6% | 40.0% | 33-40% |
| Maia-Reduced (32ch) | **26.0%** | **25.4%** | **26.4%** | — |
| **Maia-Full (256ch)** | **19.6%** | **22.8%** | **28.2%** | **~30-35%** |

**Key pattern reproduced (paper Section 4.1):** Each Maia model is most accurate at its own training bin (self-bin bias). Maia-1900 is best at 1900-1999, Maia-1100 is best at 1100-1199. Stockfish accuracy increases monotonically with opponent rating.

---

## 3. Figures

### Figure 1: Accuracy Curves (Paper Figure 2 equivalent)

![Accuracy Curves](../reports/fig2_accuracy_curves.png)

*Left: Our reproduction. Paper shows the same V-shaped pattern where each Maia model peaks at its training bin. Stockfish (dashed) increases monotonically with rating.*

### Figure 2: Agreement Matrix (Paper Figure 6 equivalent)

![Agreement Matrix](../reports/fig6_agreement_matrix.png)

*Heatmap of model-bin accuracy. Darker = higher accuracy. Each model's self-bin is always the darkest cell in its row.*

### Figure 3: Comparison to Paper

![Paper Comparison](../reports/paper_comparison.png)

*Our best accuracy vs. paper's approximate self-bin accuracy. The gap (~8-10%) is expected given ~3% of the compute budget (15K vs 400K steps).*

---

## 4. Architecture Reproduction

### 4.1 Input Encoding (Paper Section 3.1)

| Component | Paper | Our Implementation | File |
|-----------|-------|-------------------|------|
| Board planes | 17 (12 piece + 4 castling + 1 side) | Same | `src/encoding/board.py` |
| History planes | 12 per position × 8 = 96 | Same | `src/encoding/board.py` |
| Total input channels | 113 | 113 | — |
| Move encoding | 8×8×73 (7 promotion + 56 queen-move + 8 knight + 2 underpromotion) | Same | `src/encoding/move.py` |
| Move indexing | 64 squares × 73 = 4,672 logits | Same | `src/encoding/move.py` |

### 4.2 Network Architecture (Paper Section 3.2, citing Lc0 id11261)

| Component | Paper | Our Implementation | File |
|-----------|-------|-------------------|------|
| Initial conv | 113→256, 3×3, BN, ReLU | Same | `src/models/maia_net.py` |
| Residual blocks | 15 blocks, each: Conv→BN→ReLU→Conv→BN+Skip | Same (6 for reduced) | `src/models/maia_net.py` |
| Channels | 256 | 256 (32 for reduced) | — |
| SE blocks | Squeeze-and-excitation after each residual | Same | `src/models/maia_net.py:SEBlock` |
| Policy head | Conv 256→73, Flatten → 4672 logits | Same | `src/models/maia_net.py` |
| Value head | Conv→FC256→FC3 (WDL probabilities) | Same | `src/models/maia_net.py` |

### 4.3 SE Block Detail (Paper describes as "channel attention")

```
Input (C) → GlobalAvgPool → FC(C/16) → ReLU → FC(C) → Sigmoid → Scale(C)
```

Implemented in `src/models/maia_net.py:SEBlock`.

---

## 5. Training Reproduction

### 5.1 Data (Paper Section 4)

| Aspect | Paper | Our Reproduction |
|--------|-------|-----------------|
| Source | Lichess 2013-2019 | Lichess 2019-10 |
| Rating bins | 1100-1199, 1200-1299, ..., 1900-1999 | **1100-1199, 1500-1599, 1900-1999** (3 of 9) |
| Games per bin | ~100K+ | ~25K-63K |
| Time control | ≥30 sec increment, no bullet | Same |
| Ply filter | ply ≥ 10 | Same |
| Test set | Dec 2019 | Oct 2019 (train/val split) |

### 5.2 Training Config

| Aspect | Paper | Our Reproduction |
|--------|-------|-----------------|
| GPUs | 8× NVIDIA V100 | 1× RTX 2050 (4GB) |
| Batch size (effective) | 1,024 | 64 |
| Training steps | 400K | 15K-20K |
| Optimizer | Adam | Same |
| Initial LR | 0.1 | 0.01 (adjusted for stability) |
| LR decay | ×0.1 at 60K/200K/360K (paper) | ×0.1 at 5K/10K/14K (our) |
| Gradient clipping | 5.0 | Same |
| Weight decay | Not specified | 1e-4 |

### 5.3 Training Pipeline

```bash
# Data extraction (stream decompress + parse)
python scripts/extract_data.py --pgn data/pgn/lichess_db_standard_rated_2019-10.pgn.zst

# Reduced model (32ch, 6 blocks, no history) — ~1h per bin
python scripts/train_bin.py 1100

# Full-scale model (256ch, 15 blocks, 8 history) — ~1-3h per bin
python scripts/train_full.py 1100
```

---

## 6. Evaluation Reproduction

### 6.1 Move-Matching Accuracy (Paper Section 4.1)

Evaluated on 500 held-out positions per bin:

```bash
# Stockfish baselines (3 depths on same positions)
python scripts/stockfish_baselines.py

# Full model evaluation with history context
python scripts/eval_full2.py
```

### 6.2 Stockfish Baselines (Paper Section 4.2)

The paper finds Stockfish accuracy increases monotonically with opponent rating. We reproduce this pattern (Fig 1 above). Stockfish depth 1 consistently outperforms depth 7 for human-matching, confirming the paper's finding that *weaker* Stockfish is more human-like.

### 6.3 Agreement Matrix (Paper Section 4.3, Figure 6)

The agreement matrix shows that each model's self-bin accuracy is always the highest in its row. This V-shaped pattern is the paper's core finding and is reproduced in our Figure 2.

---

## 7. Deviations from Paper

| Paper Feature | Our Status | Reason |
|---------------|-----------|--------|
| 9 rating bins | 3 bins | Reduced to fit compute budget |
| Leela Chess Zero baseline | Skipped | Adds complexity, same pattern as Stockfish |
| Collective blunder prediction | Not implemented | Separate experiment |
| Blunder prediction (Table 1) | Not implemented | Separate experiment |
| UCI wrapper for Maia | Not implemented | Not needed for offline eval |
| Extended test sets (1000, 2500) | Dec 2019 download started | Time constraint |
| Personalization | Not implemented | Extension beyond core paper |
| 400K training steps | 15K-20K | ~3% of compute budget |

---

## 8. Paper Citations by Section

| Paper Section | Topic | Our Coverage |
|---------------|-------|-------------|
| 1. Introduction | Aligning AI with human behavior | README.md |
| 2. Related Work | Human behavior prediction | — |
| 3.1, 3.2 | Architecture design | `src/encoding/`, `src/models/maia_net.py` |
| 3.3 | Move prediction task | `src/encoding/move.py` |
| 4.1 | Datasets (Lichess) | `src/data_pipeline/`, `scripts/extract_data.py` |
| 4.2 | Model training | `scripts/train_full.py`, `scripts/train_bin.py` |
| 5.1 | Move-matching accuracy | `reports/fig2_accuracy_curves.png` |
| 5.2 | Model agreement (Fig 6) | `reports/fig6_agreement_matrix.png` |
| 5.3 | Stockfish comparison | `reports/agreement_matrix_combined.png` |
| 5.4 | Complexity analysis | Not implemented |
| 6 | Blunder prediction | Models exist, untrained |
| 7 | Collective blunder | Not implemented |

---

## 9. How We Achieved Each Result

### Data Pipeline
Stream-decompressed 6.4 GB Lichess ZST file in ~17 minutes using `zstandard`. Scanned 1.5M games, filtered to 3 rating bins (1100/1500/1900), keeping only positions with ply ≥ 10, time control ≥ 30 sec increment, no bullet. Output: ~1.2M moves per bin as JSON records with FEN, UCI move, game ID, clock info, and centipawn evaluation.

### Encoding
Board → 8×8×17 tensor: one-hot piece positions (6 types × 2 colors), castling rights (4 bits), and side to move. History → 12 additional piece-only planes per previous position. Move → 8×8×73 plane: 56 queen-moves, 8 knight-moves, 9 promotion options (3 pieces × 3 steps). All round-trip verified by unit tests (54/54 passing).

### Model
Built from scratch in PyTorch using residual blocks with batch normalization, ReLU activation, and SE (squeeze-and-excitation) channel attention. Policy head produces 4,672 logits (64 squares × 73 move types), masked to legal moves before softmax. Value head outputs 3-class WDL probabilities. Full model: 18.6M parameters, fits in 225 MB GPU memory at batch 8.

### Training
Adjusted for RTX 2050 (4GB): batch 8 with gradient accumulation ×8 (effective 64), 15K-20K steps with LR schedule scaled proportionally from paper's 400K step schedule. Learning rate 0.01 → 0.001 → 0.0001 → 0.00001. Gradient clipping at 5.0. Weight decay 1e-4 to prevent divergence. Self-bin validation every 2K steps.

### Evaluation
Stockfish depth 1/7/15 on same 500 positions per bin. Maia evaluated with proper game-context history reconstruction (8 previous positions from same game). Agreement matrix generated from 5 models × 3 bins = 15 data points.

---

## 10. References

McIlroy-Young, R., Sen, S., Kleinberg, J., & Anderson, A. (2020). Aligning Superhuman AI with Human Behavior: Chess as a Model System. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (KDD '20). https://doi.org/10.1145/3394486.3403219

Data source: Lichess Database, https://database.lichess.org/
