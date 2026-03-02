#!/usr/bin/env python3
"""
B21: Markov Coupling and the Persistent O(1/l²) Correction

B21 found that ⟨|det(I - M_l/l^s)|⟩ ≠ |1 - l^{-s}|^{-1} even for D→∞.
The error is O(1/l²), not O(1/l). This means the identity is NOT exact.

Two possibilities:
  (a) The error can be absorbed by a CORRECTION FACTOR R(l), making
      ⟨|det|⟩ · R(l) = |1-l^{-s}|^{-1} exactly
  (b) The error is non-trivial and depends on s

This experiment:
  A) Measure the exact correction ratio for many l at s=2
  B) Test if the ratio is s-independent (i.e., a simple multiplicative correction)
  C) Fit the correction to known constants (ζ(2), π², etc.)
  D) Check if the RENORMALIZED product converges to ζ(s)^{-1}
  E) Test the trace anomaly prediction: correction ∝ ⟨Tr(M)⟩²/l²
"""

import sys, os, time, random, math, cmath
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, carry_poly_int, quotient_poly_int

random.seed(42)
np.random.seed(42)


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def compute_det_and_trace(p, q, l, s, base=2):
    """Compute |det(I - M_l/l^s)| and Tr(M_l) for the carry companion matrix."""
    C = carry_poly_int(p, q, base)
    Q = quotient_poly_int(C, base)
    if len(Q) < 3:
        return None, None
    lead = float(Q[-1])
    if abs(lead) < 1e-30:
        return None, None
    n = len(Q) - 1
    M = np.zeros((n, n), dtype=complex)
    for i in range(n - 1):
        M[i + 1, i] = 1.0
    for i in range(n):
        M[i, n - 1] = -float(Q[i]) / lead

    trace = np.real(np.trace(M))
    ls = l ** (-s)
    det_val = np.linalg.det(np.eye(n, dtype=complex) - M * ls)
    return abs(det_val), trace


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("P1-04: MARKOV CORRECTION AND THE O(1/l²) TERM")
    pr("=" * 72)

    n_samp = 5000
    bits = 16

    # ═══════════════════════════════════════════════════════════════
    # PART A: EXACT CORRECTION RATIO AT s = 2
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: CORRECTION RATIO R(l) = ⟨|det|⟩ / target AT s = 2")
    pr(f"{'═' * 72}\n")

    s_val = 2.0
    test_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    ratios_s2 = {}
    for l in test_primes:
        target = abs(1.0 / (1.0 - l ** (-s_val)))
        det_vals = []
        trace_vals = []
        for _ in range(n_samp):
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            d, tr = compute_det_and_trace(p, q, l, s_val)
            if d is not None and d > 0 and not math.isnan(d) and not math.isinf(d):
                det_vals.append(d)
                trace_vals.append(tr)

        if len(det_vals) < 100:
            continue
        mean_det = np.mean(det_vals)
        ratio = mean_det / target
        mean_trace = np.mean(trace_vals)
        ratios_s2[l] = ratio
        pr(f"  l = {l:3d}: R(l) = {ratio:.8f}, "
           f"(R-1) = {ratio - 1:.6f}, "
           f"l²·(R-1) = {l*l*(ratio-1):.4f}, "
           f"⟨Tr⟩ = {mean_trace:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PART B: IS R(l) INDEPENDENT OF s?
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: R(l) AT DIFFERENT VALUES OF s")
    pr(f"{'═' * 72}")
    pr("If R(l) is the same for all s, it's a simple multiplicative correction.\n")

    s_values = [1.5, 2.0, 3.0, complex(0.5, 14.13), complex(0.75, 5.0)]
    test_l = [3, 5, 7, 11]

    for l in test_l:
        pr(f"  l = {l}:")
        for s in s_values:
            target = abs(1.0 / (1.0 - l ** (-s)))
            det_vals = []
            for _ in range(n_samp):
                p = random_prime(bits)
                q = random_prime(bits)
                if p == q:
                    continue
                d, _ = compute_det_and_trace(p, q, l, s)
                if d is not None and d > 0 and not math.isnan(d) and not math.isinf(d):
                    det_vals.append(d)
            if len(det_vals) < 100:
                continue
            mean_det = np.mean(det_vals)
            ratio = mean_det / target
            s_str = f"{s}" if isinstance(s, float) else f"{s.real:.2f}+{s.imag:.2f}i"
            pr(f"    s = {s_str:>12s}: R = {ratio:.6f}, (R-1) = {ratio-1:.6f}")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART C: FIT R-1 TO KNOWN CONSTANTS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: FIT (R-1) vs 1/l²")
    pr(f"{'═' * 72}\n")

    if len(ratios_s2) >= 5:
        l_arr = np.array(sorted(ratios_s2.keys()), dtype=float)
        r_arr = np.array([ratios_s2[int(l)] for l in l_arr])
        rm1 = r_arr - 1.0

        l2_product = l_arr ** 2 * rm1
        pr(f"  l²·(R-1) values:")
        for l, prod in zip(l_arr, l2_product):
            pr(f"    l = {l:3.0f}: l²·(R-1) = {prod:.6f}")
        pr(f"\n  Mean l²·(R-1) = {l2_product.mean():.6f}")
        pr(f"  Std  l²·(R-1) = {l2_product.std():.6f}")

        pr(f"\n  Candidate constants:")
        candidates = {
            '1/4': 0.25,
            'ζ(2)/6 = π²/36': math.pi**2 / 36,
            '1/(2π)': 1 / (2 * math.pi),
            'ln(2)': math.log(2),
            '1/3': 1/3.0,
            '1/2': 0.5,
        }
        for name, val in candidates.items():
            pr(f"    {name:>20s} = {val:.6f}")

        fit_l = l_arr[l_arr >= 7]
        fit_r = np.array([ratios_s2[int(l)] for l in fit_l])
        fit_rm1 = fit_r - 1.0
        fit_product = fit_l ** 2 * fit_rm1
        pr(f"\n  For l ≥ 7: mean l²·(R-1) = {fit_product.mean():.6f}")

        coeffs = np.polyfit(1.0/l_arr**2, rm1, 1)
        pr(f"\n  Linear fit (R-1) = c/l² + d:")
        pr(f"    c = {coeffs[0]:.6f}")
        pr(f"    d = {coeffs[1]:.8f}")

    # ═══════════════════════════════════════════════════════════════
    # PART D: RENORMALIZED EULER PRODUCT
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: RENORMALIZED EULER PRODUCT vs ζ(s)^{-1}")
    pr(f"{'═' * 72}\n")

    s_test = 2.0
    zeta_2 = math.pi ** 2 / 6
    target_product = 1.0 / zeta_2  # ζ(2)^{-1} = 6/π²

    max_l_values = [10, 20, 30, 50]
    for max_l in max_l_values:
        primes_used = [p for p in test_primes if p <= max_l]
        if len(primes_used) < 2:
            continue

        raw_product = 1.0
        corrected_product = 1.0
        for l in primes_used:
            target_factor = abs(1.0 / (1.0 - l ** (-s_test)))
            r = ratios_s2.get(l, 1.0)
            raw_product *= r * target_factor
            corrected_product *= target_factor

        raw_inv = 1.0 / raw_product if raw_product > 0 else float('inf')
        corr_inv = 1.0 / corrected_product if corrected_product > 0 else float('inf')

        pr(f"  Primes ≤ {max_l}: "
           f"raw ∏⟨det⟩ = {raw_inv:.8f}, "
           f"ideal ∏|1-l^-s|^-1 = {corr_inv:.8f}, "
           f"target 1/ζ(2) = {target_product:.8f}")

    # ═══════════════════════════════════════════════════════════════
    # PART E: TRACE ANOMALY CONNECTION
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: TRACE ANOMALY — IS (R-1) ∝ ⟨Tr⟩²/l²?")
    pr(f"{'═' * 72}\n")

    for l in [3, 5, 7, 11, 13]:
        trace_vals = []
        for _ in range(n_samp):
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            _, tr = compute_det_and_trace(p, q, l, 2.0)
            if tr is not None:
                trace_vals.append(tr)

        mean_tr = np.mean(trace_vals)
        var_tr = np.var(trace_vals)
        r = ratios_s2.get(l, 1.0)
        rm1 = r - 1.0

        pr(f"  l = {l:3d}: ⟨Tr⟩ = {mean_tr:.4f}, "
           f"Var(Tr) = {var_tr:.4f}, "
           f"⟨Tr⟩²/l² = {mean_tr**2/l**2:.6f}, "
           f"(R-1) = {rm1:.6f}, "
           f"ratio = {rm1 / (mean_tr**2/l**2) if mean_tr != 0 else 'N/A':.4f}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS")
    pr(f"{'═' * 72}")
    pr("""
  The carry per-factor identity has a PERSISTENT correction:
    ⟨|det(I - M_l/l^s)|⟩ = |1 - l^{-s}|^{-1} · R(l)

  where R(l) = 1 + c/l² + higher order terms.

  Key questions answered:
    1. Is R(l) independent of s? → Part B
    2. What is c? → Part C
    3. Does it come from the trace anomaly? → Part E

  If R(l) is s-independent, the renormalized product would be:
    ∏_l ⟨|det|⟩ / R(l) = ∏_l |1-l^{-s}|^{-1} = ζ(s)

  And the correction ∏_l R(l) would be a convergent product (since Σ 1/l² < ∞).
""")

    pr(f"  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
