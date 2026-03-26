"""
Interactive Wordle Solver (7925 methodology)

Decision logic:
  1. Perfect differentiator (if ≤20 words remain)
  2. Dangerous suffix cluster detection
  3. If ≤17 remaining AND no clusters → Expected-size class
     Otherwise → Second-largest-bucket class
  4. Among top-bucket candidates, prefer answer words

Feedback: g = green, y = yellow, b = black/gray

Requires:
  - wordlist_orig_hidden  (2315 answer words)
  - wordlist_orig_all     (12972 valid guesses)
"""

import time
import numpy as np
from collections import defaultdict
from typing import List, Optional

GRAY, YELLOW, GREEN = 0, 1, 2
ALL_GREEN = 242
PERFECT_DIFF_GATE = 20
OPENER = "salet"
BASE3 = np.array([81, 27, 9, 3, 1], dtype=np.int32)
TOP_N_SHOW = 10


def load_words(path: str) -> List[str]:
    with open(path) as f:
        return [ln.strip().lower() for ln in f if len(ln.strip()) == 5]


def feedback_to_int(fb: str) -> int:
    m = {'g': 2, 'y': 1, 'b': 0}
    return sum(m[c] * b for c, b in zip(fb, [81, 27, 9, 3, 1]))


def int_to_feedback(val: int) -> str:
    m = {0: 'b', 1: 'y', 2: 'g'}
    out = []
    for b in [81, 27, 9, 3, 1]:
        out.append(m[val // b]); val %= b
    return ''.join(out)


def build_pattern_matrix(guesses: List[str], answers: List[str]) -> np.ndarray:
    G, A = len(guesses), len(answers)
    print(f"  Building {G}×{A} pattern matrix … ", end="", flush=True)
    t0 = time.time()

    gc = np.array([[ord(c) for c in w] for w in guesses], dtype=np.uint8)
    ac = np.array([[ord(c) for c in w] for w in answers], dtype=np.uint8)
    mat = np.zeros((G, A), dtype=np.uint8)

    for i in range(G):
        g = gc[i]
        green = ac == g

        freq = np.zeros((A, 26), dtype=np.int8)
        for p in range(5):
            freq[np.arange(A), ac[:, p] - ord('a')] += 1
        for p in range(5):
            freq[green[:, p], g[p] - ord('a')] -= 1

        yellow = np.zeros((A, 5), dtype=bool)
        for p in range(5):
            if green[:, p].all():
                continue
            ng = ~green[:, p]
            li = g[p] - ord('a')
            yl = ng & (freq[:, li] > 0)
            yellow[:, p] = yl
            freq[yl, li] -= 1

        vals = green.astype(np.int32) * 2 + yellow.astype(np.int32)
        mat[i] = (vals @ BASE3).astype(np.uint8)

    print(f"done ({time.time()-t0:.1f}s)")
    return mat


class WordleSolver:
    def __init__(self, answers, guesses, matrix):
        self.answers  = answers
        self.guesses  = guesses
        self.matrix   = matrix
        self.A        = len(answers)
        self.G        = len(guesses)
        self.ans_idx  = {w: i for i, w in enumerate(answers)}
        self.gue_idx  = {w: i for i, w in enumerate(guesses)}
        self.ans_rows = np.array([self.gue_idx[w] for w in answers], dtype=np.int32)
        self.all_rows = np.arange(self.G, dtype=np.int32)
        self.opener   = self.gue_idx[OPENER]

    def filter(self, R, grow, pat):
        return R[self.matrix[grow, R] == pat]

    def _num_buckets(self, cand_rows, R):
        sub = self.matrix[np.ix_(cand_rows, R)]
        if sub.shape[1] <= 1:
            return np.ones(len(cand_rows), dtype=np.int32)
        s = np.sort(sub, axis=1)
        return (np.diff(s, axis=1) != 0).sum(axis=1).astype(np.int32) + 1

    def _bucket_profile(self, grow, R):
        _, cnts = np.unique(self.matrix[grow, R], return_counts=True)
        return sorted(cnts.tolist(), reverse=True)

    def _metrics(self, grow, R):
        prof = self._bucket_profile(grow, R)
        pad  = prof + [0] * 16
        n    = len(R)
        ssq  = sum(c * c for c in prof)
        return {
            "buckets":      len(prof),
            "expected":     ssq / n,
            "worst":        pad[0],
            "second_worst": pad[1],
            "lex_tail":     tuple(pad[1:13]),
        }

    def _top_bucket_rows(self, R):
        nb   = self._num_buckets(self.all_rows, R)
        best = int(nb.max())
        return self.all_rows[nb == best]

    def _perfect_diff(self, R) -> Optional[int]:
        n = len(R)
        if n > PERFECT_DIFF_GATE:
            return None
        ar = self.ans_rows[R]
        nb = self._num_buckets(ar, R)
        if (nb == n).any():
            return int(ar[np.argmax(nb == n)])
        nb = self._num_buckets(self.all_rows, R)
        if (nb == n).any():
            return int(self.all_rows[np.argmax(nb == n)])
        return None

    def _dangerous_cluster(self, R, min_sz=3) -> int:
        words = [self.answers[c] for c in R]
        worst = 0
        for slen in [4, 3, 2]:
            groups = defaultdict(list)
            for i, w in enumerate(words):
                groups[w[5 - slen:]].append(i)
            for idxs in groups.values():
                if len(idxs) < min_sz:
                    continue
                cols = R[np.array(idxs, dtype=np.int32)]
                nb   = self._num_buckets(self.all_rows, cols)
                if not (nb == len(cols)).any():
                    worst = max(worst, len(idxs))
        return worst

    def _pick_expected(self, R, show=False):
        top  = self._top_bucket_rows(R)
        rset = set(self.ans_rows[R].tolist())

        scored = []
        for r in top:
            r = int(r)
            m = self._metrics(r, R)
            k = (m["expected"], m["worst"], m["lex_tail"], r)
            scored.append((k, r, m))
        scored.sort(key=lambda x: x[0])

        pick = None
        for k, r, m in scored:
            if r in rset:
                pick = r
                break
        if pick is None:
            pick = scored[0][1]

        if show:
            self._show_candidates(scored, rset, "EXPECTED-SIZE", pick)

        return pick

    def _pick_second(self, R, show=False):
        top  = self._top_bucket_rows(R)
        rset = set(self.ans_rows[R].tolist())

        scored = []
        for r in top:
            r = int(r)
            m = self._metrics(r, R)
            k = (m["second_worst"], m["expected"],
                 m["worst"], m["lex_tail"], r)
            scored.append((k, r, m))
        scored.sort(key=lambda x: x[0])

        pick = None
        for k, r, m in scored:
            if r in rset:
                pick = r
                break
        if pick is None:
            pick = scored[0][1]

        if show:
            self._show_candidates(scored, rset, "2ND-LARGEST-BUCKET", pick)

        return pick

    def _show_candidates(self, scored, rset, class_name, pick):
        n_show = min(TOP_N_SHOW, len(scored))

        pick_rank = None
        for i, (_, r, _) in enumerate(scored):
            if r == pick:
                pick_rank = i + 1
                break

        print(f"\n  ┌─ Strategy: {class_name}")
        print(f"  │  Showing {n_show} of {len(scored)} top-bucket candidates")
        print(f"  │  (✓ = possible answer, ◀ = selected)")
        if pick_rank and pick_rank > 1:
            print(f"  │  NOTE: Picking #{pick_rank} because answer words are preferred")
        print(f"  │")
        print(f"  │  {'#':>3}  {'Word':<7} {'Ans?':>4}"
              f"  {'Bkts':>4}  {'Worst':>5}  {'2nd':>5}"
              f"  {'E[size]':>8}")
        print(f"  │  {'─'*3}  {'─'*7} {'─'*4}"
              f"  {'─'*4}  {'─'*5}  {'─'*5}"
              f"  {'─'*8}")

        for rank, (_, r, m) in enumerate(scored[:n_show], 1):
            word   = self.guesses[r]
            is_ans = "  ✓" if r in rset else ""
            marker = " ◀" if r == pick else ""
            print(f"  │  {rank:>3}  {word.upper():<7} {is_ans:>4}"
                  f"  {m['buckets']:>4}  {m['worst']:>5}  {m['second_worst']:>5}"
                  f"  {m['expected']:>8.3f}{marker}")

        if pick_rank and pick_rank > n_show:
            print(f"  │  ...")
            pick_data = scored[pick_rank - 1]
            m = pick_data[2]
            word = self.guesses[pick]
            is_ans = "  ✓" if pick in rset else ""
            print(f"  │  {pick_rank:>3}  {word.upper():<7} {is_ans:>4}"
                  f"  {m['buckets']:>4}  {m['worst']:>5}  {m['second_worst']:>5}"
                  f"  {m['expected']:>8.3f} ◀")
        elif len(scored) > n_show:
            print(f"  │  ... and {len(scored) - n_show} more")

        print(f"  └{'─'*55}")

    def choose(self, R, show=True):
        if len(R) == 1:
            return self.ans_rows[R[0]]

        pd = self._perfect_diff(R)
        if pd is not None:
            if show:
                word = self.guesses[pd]
                rset = set(self.ans_rows[R].tolist())
                tag  = " (answer word ✓)" if pd in rset else " (non-answer)"
                print(f"\n  ┌─ PERFECT DIFFERENTIATOR found!")
                print(f"  │  {word.upper()}{tag} splits all"
                      f" {len(R)} words uniquely")
                print(f"  └{'─'*55}")
            return pd

        cluster = self._dangerous_cluster(R)
        if show and cluster > 0:
            print(f"\n  ⚠  Dangerous suffix cluster detected"
                  f" (size {cluster})")

        if len(R) <= 17 and cluster <= 0:
            return self._pick_expected(R, show=show)
        else:
            return self._pick_second(R, show=show)

    def get_row_for_word(self, word: str) -> Optional[int]:
        return self.gue_idx.get(word.lower())


def play(solver: WordleSolver):
    R = np.arange(solver.A, dtype=np.int32)
    turn = 0

    print(f"\n{'═'*56}")
    print(f"  WORDLE SOLVER")
    print(f"  Feedback: g = green · y = yellow · b = black")
    print(f"  You can override any suggestion with your own word")
    print(f"{'═'*56}")

    guess_word = OPENER
    guess_row  = solver.opener

    while True:
        turn += 1

        rset = set(solver.ans_rows[R].tolist())
        ans_tag = " ✓" if guess_row in rset else ""
        print(f"\n  ╔══════════════════════════════════╗")
        print(f"  ║  Guess #{turn}:  {guess_word.upper():<16}{ans_tag:>5} ║")
        print(f"  ╚══════════════════════════════════╝")
        print(f"  ({len(R)} possible answer"
              f"{'s' if len(R) != 1 else ''} remain)")

        override = input(f"\n  Press Enter to play {guess_word.upper()}"
                         f", or type a different word ▸ ").strip().lower()

        if override in ('q', 'quit', 'exit'):
            print("  Quitting this game.")
            return None

        if override:
            row = solver.get_row_for_word(override)
            if row is None:
                print(f"  ✗ '{override}' not in guess list! "
                      f"Using {guess_word.upper()} instead.")
            else:
                guess_word = override
                guess_row  = row
                print(f"  → Playing {guess_word.upper()} instead")

        while True:
            fb = input(f"\n  Feedback for {guess_word.upper()} ▸ ").strip().lower()
            if fb in ('q', 'quit', 'exit'):
                print("  Quitting this game.")
                return None
            if len(fb) == 5 and all(c in 'gyb' for c in fb):
                break
            print("  ✗ Enter exactly 5 chars using g / y / b"
                  "  (or 'q' to quit)")

        colors = {'g': '\033[42;30m', 'y': '\033[43;30m',
                  'b': '\033[40;37m'}
        reset  = '\033[0m'
        disp = "  "
        for ch, f in zip(guess_word.upper(), fb):
            disp += f"{colors[f]} {ch} {reset}"
        print(disp)

        if fb == "ggggg":
            print(f"\n  ✅ Solved in {turn} guess{'es' if turn > 1 else ''}!")
            return turn

        pat = feedback_to_int(fb)
        R   = solver.filter(R, guess_row, pat)

        if len(R) == 0:
            print("\n  ⚠  No words match that feedback!"
                  "  Double-check and try again.")
            return None

        if len(R) <= 20:
            words = [solver.answers[c] for c in R]
            print(f"\n  Remaining ({len(R)}): "
                  f"{', '.join(w.upper() for w in words)}")

        if len(R) == 1:
            guess_word = solver.answers[R[0]]
            guess_row  = solver.ans_rows[R[0]]
            continue

        print("\n  Thinking …", flush=True)
        t0 = time.time()
        guess_row  = solver.choose(R, show=True)
        guess_word = solver.guesses[guess_row]
        elapsed    = time.time() - t0
        print(f"  (chose in {elapsed:.2f}s)")


if __name__ == "__main__":
    print("Loading word lists …")
    answers = load_words("data/wordlist_orig_hidden")
    guesses = load_words("data/wordlist_orig_all")
    print(f"  {len(answers)} answers · {len(guesses)} guesses")

    matrix = build_pattern_matrix(guesses, answers)
    solver = WordleSolver(answers, guesses, matrix)

    while True:
        play(solver)
        again = input("\n  Play again? (y/n) ▸ ").strip().lower()
        if again != 'y':
            print("  👋  Goodbye!")
            break
