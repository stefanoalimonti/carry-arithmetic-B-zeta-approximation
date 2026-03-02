#!/usr/bin/env python3
"""
B20: Exact Per-Factor Identity Convergence as D → ∞

The per-factor identity: ⟨|det(I - M_l/l^s)|⟩ ≈ |1 - l^{-s}|^{-1} + O(1/l)

If this becomes EXACT as D → ∞ (bit-size of semiprimes → ∞), then the
carry Euler product IS ζ(s)^{-1}. This would close Gap 1.

Strategy:
  - For fixed test primes l = 3, 5, 7, 11
  - For increasing D (bit-size = 8, 12, 16, 20, 24, 28, 32, 40, 48)
  - Compute ⟨|det(I - M_l/l^s)|⟩ over many semiprimes
  - Measure the error vs |1 - l^{-s}|^{-1}
  - Determine if error → 0 as D → ∞ and measure the convergence rate

Parts:
  A) Error vs D for s = 2 (deep in convergence region)
  B) Error vs D for s = 1/2 + 14.13i (first Riemann zero)
  C) Scaling law: error ∝ D^{-α} or error ∝ b^{-D}?
  D) Error vs l for fixed D — verify the O(1/l) scaling
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


def compute_det_ratio(p, q, l, s, base=2):
    """Compute |det(I - M_l/l^s)| for the carry companion matrix."""
    C = carry_poly_int(p, q, base)
    Q = quotient_poly_int(C, base)
    if len(Q) < 3:
        return None
    lead = float(Q[-1])
    if abs(lead) < 1e-30:
        return None
    n = len(Q) - 1
    M = np.zeros((n, n), dtype=complex)
    for i in range(n - 1):
        M[i + 1, i] = 1.0
    for i in range(n):
        M[i, n - 1] = -float(Q[i]) / lead

    ls = l ** (-s)
    det_val = np.linalg.det(np.eye(n, dtype=complex) - M * ls)
    return abs(det_val)


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("P1-03: EXACT PER-FACTOR IDENTITY CONVERGENCE")
    pr("=" * 72)

    test_primes = [3, 5, 7, 11, 13]
    bit_sizes = [8, 12, 16, 20, 24, 28, 32]
    n_samples = 3000

    # ═══════════════════════════════════════════════════════════════
    # PART A: ERROR vs D FOR s = 2
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: ERROR vs D FOR s = 2 (REAL)")
    pr(f"{'═' * 72}")
    pr("Target: |1 - l^{-2}|^{-1} = l²/(l²-1)\n")

    s_val = 2.0

    results_A = {}
    for l in test_primes:
        target = abs(1.0 / (1.0 - l ** (-s_val)))
        pr(f"  l = {l}: target = {target:.10f}")
        errors = {}

        for bits in bit_sizes:
            det_vals = []
            for _ in range(n_samples):
                p = random_prime(bits)
                q = random_prime(bits)
                if p == q:
                    continue
                d = compute_det_ratio(p, q, l, s_val)
                if d is not None and d > 0:
                    det_vals.append(d)

            if len(det_vals) < 100:
                continue
            mean_det = np.mean(det_vals)
            std_det = np.std(det_vals) / math.sqrt(len(det_vals))
            rel_error = abs(mean_det - target) / target
            errors[bits] = (mean_det, rel_error, std_det, len(det_vals))

            pr(f"    D≈{2*bits:3d}: ⟨|det|⟩ = {mean_det:.8f}, "
               f"error = {rel_error:.2e}, "
               f"σ_mean = {std_det:.2e}")

        results_A[l] = errors
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART B: ERROR vs D FOR s = 1/2 + 14.13i (CRITICAL LINE)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: ERROR vs D ON THE CRITICAL LINE")
    pr(f"{'═' * 72}")
    pr("s = 1/2 + 14.13i (near first Riemann zero)\n")

    s_crit = complex(0.5, 14.134725)

    results_B = {}
    for l in [3, 5, 7]:
        target = abs(1.0 / (1.0 - l ** (-s_crit)))
        pr(f"  l = {l}: target = {target:.10f}")
        errors = {}

        for bits in bit_sizes:
            det_vals = []
            for _ in range(n_samples):
                p = random_prime(bits)
                q = random_prime(bits)
                if p == q:
                    continue
                d = compute_det_ratio(p, q, l, s_crit)
                if d is not None and d > 0:
                    det_vals.append(d)

            if len(det_vals) < 100:
                continue
            mean_det = np.mean(det_vals)
            std_det = np.std(det_vals) / math.sqrt(len(det_vals))
            rel_error = abs(mean_det - target) / target
            errors[bits] = (mean_det, rel_error, std_det, len(det_vals))

            pr(f"    D≈{2*bits:3d}: ⟨|det|⟩ = {mean_det:.8f}, "
               f"error = {rel_error:.2e}")

        results_B[l] = errors
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART C: SCALING LAW — FIT error ∝ D^{-α} or b^{-D}
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: CONVERGENCE RATE ANALYSIS")
    pr(f"{'═' * 72}\n")

    for l in [3, 5, 7]:
        errors = results_A.get(l, {})
        if len(errors) < 3:
            continue

        Ds = sorted(errors.keys())
        errs = [errors[d][1] for d in Ds]
        D_arr = np.array([2 * d for d in Ds], dtype=float)
        err_arr = np.array(errs)

        mask = err_arr > 0
        if np.sum(mask) < 3:
            continue

        log_D = np.log(D_arr[mask])
        log_err = np.log(err_arr[mask])

        if len(log_D) >= 2:
            coeffs = np.polyfit(log_D, log_err, 1)
            alpha = -coeffs[0]
            pr(f"  l = {l}: power-law fit error ∝ D^{{-{alpha:.2f}}}")

        log2_D = D_arr[mask]
        if len(log2_D) >= 2:
            coeffs_exp = np.polyfit(log2_D, log_err, 1)
            rate = -coeffs_exp[0] / math.log(2)
            pr(f"  l = {l}: exponential fit error ∝ 2^{{-{rate:.3f}·D}}")

    # ═══════════════════════════════════════════════════════════════
    # PART D: ERROR vs l FOR FIXED D — THE O(1/l) LAW
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: ERROR vs l FOR FIXED D = 32 bits")
    pr(f"{'═' * 72}")
    pr("Testing if relative error ∝ 1/l\n")

    test_l = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    bits_fixed = 16
    n_samp = 5000
    s_test = 2.0

    l_errors = []
    for l in test_l:
        target = abs(1.0 / (1.0 - l ** (-s_test)))
        det_vals = []
        for _ in range(n_samp):
            p = random_prime(bits_fixed)
            q = random_prime(bits_fixed)
            if p == q:
                continue
            d = compute_det_ratio(p, q, l, s_test)
            if d is not None and d > 0:
                det_vals.append(d)

        if len(det_vals) < 100:
            continue
        mean_det = np.mean(det_vals)
        rel_error = abs(mean_det - target) / target
        l_errors.append((l, rel_error))
        pr(f"  l = {l:3d}: error = {rel_error:.6f}, "
           f"l·error = {l * rel_error:.4f}")

    if len(l_errors) >= 3:
        l_arr = np.array([x[0] for x in l_errors], dtype=float)
        e_arr = np.array([x[1] for x in l_errors])
        product = l_arr * e_arr
        pr(f"\n  l · error: mean = {product.mean():.4f}, "
           f"std = {product.std():.4f}")
        pr(f"  If l·error ≈ const, then error = c/l with c ≈ {product.mean():.4f}")

        coeffs = np.polyfit(np.log(l_arr), np.log(e_arr), 1)
        pr(f"  Power-law fit: error ∝ l^{{{coeffs[0]:.3f}}}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS — PER-FACTOR CONVERGENCE")
    pr(f"{'═' * 72}")

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
