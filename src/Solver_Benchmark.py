"""
Benchmark for 7925 Wordle Solver
Times full evaluation across all 2,315 words and verifies total guess count.
"""

import time
import numpy as np
from collections import defaultdict
from typing import List, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────
GRAY, YELLOW, GREEN = 0, 1, 2
ALL_GREEN = 242
PERFECT_DIFF_GATE = 20
OPENER = "salet"
BASE3 = np.array([81, 27, 9, 3, 1], dtype=np.int32)


def load_words(path: str) -> List[str]:
    with open(path) as f:
        return [ln.strip().lower() for ln in f if len(ln.strip()) == 5]


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

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")
    return mat, elapsed


class WordleSolver:
    def __init__(self, answers, guesses, matrix):
        self.answers = answers
        self.guesses = guesses
        self.matrix = matrix
        self.A = len(answers)
        self.G = len(guesses)
        self.ans_idx = {w: i for i, w in enumerate(answers)}
        self.gue_idx = {w: i for i, w in enumerate(guesses)}
        self.ans_rows = np.array([self.gue_idx[w] for w in answers], dtype=np.int32)
        self.all_rows = np.arange(self.G, dtype=np.int32)
        self.opener = self.gue_idx[OPENER]

        # Benchmark tracking
        self.cluster_check_count = 0
        self.cluster_check_states = set()
        self.decision_count = 0
        self.perfect_diff_count = 0

    def reset_stats(self):
        self.cluster_check_count = 0
        self.cluster_check_states = set()
        self.decision_count = 0
        self.perfect_diff_count = 0

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
        pad = prof + [0] * 16
        n = len(R)
        ssq = sum(c * c for c in prof)
        return {
            "buckets": len(prof),
            "expected": ssq / n,
            "worst": pad[0],
            "second_worst": pad[1],
            "lex_tail": tuple(pad[1:13]),
        }

    def _top_bucket_rows(self, R):
        nb = self._num_buckets(self.all_rows, R)
        best = int(nb.max())
        return self.all_rows[nb == best]

    def _perfect_diff(self, R) -> Optional[int]:
        n = len(R)
        if n > PERFECT_DIFF_GATE:
            return None
        ar = self.ans_rows[R]
        nb = self._num_buckets(ar, R)
        if (nb == n).any():
            self.perfect_diff_count += 1
            return int(ar[np.argmax(nb == n)])
        nb = self._num_buckets(self.all_rows, R)
        if (nb == n).any():
            self.perfect_diff_count += 1
            return int(self.all_rows[np.argmax(nb == n)])
        return None

    def _dangerous_cluster(self, R, min_sz=3) -> int:
        # Track this call
        state_key = tuple(sorted(R.tolist()))
        self.cluster_check_count += 1
        self.cluster_check_states.add(state_key)

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
                nb = self._num_buckets(self.all_rows, cols)
                if not (nb == len(cols)).any():
                    worst = max(worst, len(idxs))
        return worst

    def _pick_expected(self, R):
        top = self._top_bucket_rows(R)
        rset = set(self.ans_rows[R].tolist())

        scored = []
        for r in top:
            r = int(r)
            m = self._metrics(r, R)
            k = (m["expected"], m["worst"], m["lex_tail"], r)
            scored.append((k, r))
        scored.sort(key=lambda x: x[0])

        for k, r in scored:
            if r in rset:
                return r
        return scored[0][1]

    def _pick_second(self, R):
        top = self._top_bucket_rows(R)
        rset = set(self.ans_rows[R].tolist())

        scored = []
        for r in top:
            r = int(r)
            m = self._metrics(r, R)
            k = (m["second_worst"], m["expected"],
                 m["worst"], m["lex_tail"], r)
            scored.append((k, r))
        scored.sort(key=lambda x: x[0])

        for k, r in scored:
            if r in rset:
                return r
        return scored[0][1]

    def choose(self, R):
        self.decision_count += 1

        if len(R) == 1:
            return self.ans_rows[R[0]]

        pd = self._perfect_diff(R)
        if pd is not None:
            return pd

        cluster = self._dangerous_cluster(R)

        if len(R) <= 17 and cluster <= 0:
            return self._pick_expected(R)
        else:
            return self._pick_second(R)

    def solve(self, target_idx: int) -> Tuple[int, List[str]]:
        """Solve for a single target word. Returns (num_guesses, path)."""
        R = np.arange(self.A, dtype=np.int32)
        target = self.answers[target_idx]
        target_row = self.ans_rows[target_idx]

        guess_row = self.opener
        path = [self.guesses[guess_row]]

        while True:
            pat = self.matrix[guess_row, target_idx]
            R = self.filter(R, guess_row, pat)

            if pat == ALL_GREEN:
                return len(path), path

            if len(R) == 0:
                raise ValueError(f"No words remain for {target}")

            guess_row = self.choose(R)
            path.append(self.guesses[guess_row])

            if len(path) > 10:
                raise ValueError(f"Too many guesses for {target}: {path}")

        return len(path), path


def run_benchmark(solver: WordleSolver) -> dict:
    """Run full benchmark across all 2,315 words."""

    solver.reset_stats()

    total_guesses = 0
    distribution = defaultdict(int)
    six_plus_words = []
    worst_case = 0

    t0 = time.time()

    for i in range(solver.A):
        n_guesses, path = solver.solve(i)
        total_guesses += n_guesses
        distribution[n_guesses] += 1

        if n_guesses >= 6:
            six_plus_words.append((solver.answers[i], path))

        worst_case = max(worst_case, n_guesses)

    solve_time = time.time() - t0

    return {
        "total": total_guesses,
        "average": total_guesses / solver.A,
        "distribution": dict(sorted(distribution.items())),
        "worst_case": worst_case,
        "six_plus": six_plus_words,
        "solve_time": solve_time,
        "decision_count": solver.decision_count,
        "perfect_diff_count": solver.perfect_diff_count,
        "cluster_checks_total": solver.cluster_check_count,
        "cluster_checks_unique": len(solver.cluster_check_states),
    }


def main():
    print("=" * 60)
    print("  WORDLE SOLVER BENCHMARK")
    print("=" * 60)

    # Load words
    print("\nLoading word lists …")
    t_load_start = time.time()
    answers = load_words("data/wordlist_orig_hidden")
    guesses = load_words("data/wordlist_orig_all")
    t_load = time.time() - t_load_start
    print(f"  {len(answers)} answers · {len(guesses)} guesses ({t_load:.2f}s)")

    # Build matrix
    print("\nBuilding pattern matrix …")
    matrix, t_matrix = build_pattern_matrix(guesses, answers)

    # Create solver
    solver = WordleSolver(answers, guesses, matrix)

    # Run benchmark
    print("\nRunning benchmark across all words …")
    results = run_benchmark(solver)

    # Report results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\n  Total guesses:     {results['total']}")
    print(f"  Average:           {results['average']:.4f}")
    print(f"  Worst case:        {results['worst_case']} guesses")

    print(f"\n  Distribution:")
    for k, v in results['distribution'].items():
        print(f"    {k} guesses: {v:>4} words ({100*v/len(answers):>5.1f}%)")

    if results['six_plus']:
        print(f"\n  Six-guess words ({len(results['six_plus'])}):")
        for word, path in results['six_plus']:
            print(f"    {word.upper()}: {' → '.join(p.upper() for p in path)}")

    print(f"\n  Timing:")
    print(f"    Word list load:     {t_load:.2f}s")
    print(f"    Pattern matrix:     {t_matrix:.1f}s")
    print(f"    Solve all words:    {results['solve_time']:.1f}s")
    print(f"    Total runtime:      {t_load + t_matrix + results['solve_time']:.1f}s")

    print(f"\n  Decision statistics:")
    print(f"    Total decisions:              {results['decision_count']}")
    print(f"    Perfect differentiator used:  {results['perfect_diff_count']}")
    print(f"    Cluster checks (total):       {results['cluster_checks_total']}")
    print(f"    Cluster checks (unique):      {results['cluster_checks_unique']}")

    # Verification
    print("\n" + "=" * 60)
    print("  VERIFICATION")
    print("=" * 60)

    expected = 7925
    if results['total'] == expected:
        print(f"\n  ✓ PASSED: Total {results['total']} matches expected {expected}")
    else:
        print(f"\n  ✗ FAILED: Total {results['total']} != expected {expected}")
        print(f"    Difference: {results['total'] - expected:+d}")

    return results


if __name__ == "__main__":
    results = main()
