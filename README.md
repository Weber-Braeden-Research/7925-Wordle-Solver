# 7925-Wordle-Solver

Source code for **"Near-Optimal Wordle Without Search: Pattern-Count Maximization, Variable Hierarchy, and the Limits of Greedy Play"**

A greedy Wordle solver achieving **7,925 total guesses** across all 2,315 words (average 3.4233), which is only 5 guesses above the provably optimal 7,920 of Selby (2022). Uses no lookahead, no entropy, and no search.

## Results

| Guesses | Words | % |
|---------|-------|------|
| 2 | 79 | 3.4% |
| 3 | 1,221 | 52.8% |
| 4 | 973 | 42.0% |
| 5 | 40 | 1.7% |
| 6 | 2 | 0.09% |

**Worst-case:** 6 guesses (HOVER, JOKER)

## Algorithm Summary

1. Open with **SALET**
2. If one word remains, guess it
3. If |R| ≤ 20 and a perfect differentiator exists, play it (prefer answer words)
4. Otherwise, find all guesses maximizing pattern count. Among those:
   - Prefer guesses from the remaining answer set
   - If |R| ≤ 17 and no dangerous suffix cluster exists: minimize expected bucket size
   - Otherwise: minimize second-largest bucket size

The complete algorithm fits in four sentences and runs in under a minute.

## Comparison

| Solver | Total | Gap | Method |
|--------|-------|-----|--------|
| Selby (2022) | 7,920 | 0 | Exhaustive minimax |
| **This work** | **7,925** | +5 | Greedy, adaptive cascade |
| Sanderson (2022) | ~7,948 | +28 | Entropy + depth-2 lookahead |

## Requirements

```
pip install -r requirements.txt
```

Requires Python 3.7+ and NumPy.

## Usage

**Reproduce the 7,925 result:**
```bash
python solver_benchmark.py
```

**Interactive play:**
```bash
python solver_interactive.py
```

Feedback format: `g` = green, `y` = yellow, `b` = black/gray

## Files

| File | Description |
|------|-------------|
| `/src/Solver_Benchmark.py` | Automated benchmark across all 2,315 words |
| `/src/Solver_Interactive.py` | Interactive solver with suggestions |
| `/data/wordlist_orig_hidden` | 2,315 answer words (from original Wordle source) |
| `/data/wordlist_orig_all` | 12,972 valid guesses |

## Key Findings

- **Answer-set preference** accounts for 98.7% of improvement over naive pattern-count maximization
- **Shannon entropy fails**: belongs to the same metric cluster as pattern count, actively harmful in clustered regimes
- **Lookahead fails**: triggers score saturation—the large guess dictionary provides differentiators for virtually any small bucket
- **Variable hierarchy**: dozens of tiebreaker metrics collapse into two latent families; second-largest bucket is the actionable independent dimension
- **Self-differentiation capacity**: English answer set achieves 0.899 average ratio, explaining why greedy play succeeds

## Word Lists

Both word lists originate from the original Wordle source code as archived by [Selby (2022)](https://github.com/alex1770/wordle). The 2,315-word answer list predates the New York Times acquisition.

## Citation

```bibtex
@article{weber2026wordle,
  title={Near-Optimal Wordle Without Search: Pattern-Count Maximization, 
         Variable Hierarchy, and the Limits of Greedy Play},
  author={Weber, Braeden},
  year={2026}
}
```

## License

MIT

## Acknowledgments

- Alex Selby for publishing optimal decision trees and word lists
- Jonathan Olson for independent verification of optimal solutions
