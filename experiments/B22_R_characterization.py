#!/usr/bin/env python3
"""
B22: Characterize R(l,s) Analytically

From prior experiments: R(l,s) = ⟨|det(I - M_l/l^s)|⟩ / |1 - l^{-s}|^{-1}

At s=2: l²·(R-1) ≈ 0.168 ≈ 1/6. But R depends on s!

Hypothesis: R(l,s) involves terms like l^{-2s} (second Euler factor).
The log-expansion of det gives:
  ln|det(I-M/l^s)| = -Re[Tr(M)/l^s + Tr(M²)/(2l^{2s}) + ...]

So R might have the form: R(l,s) ≈ 1 + A/l^{2σ} + B/l^{3σ} + ...
where σ = Re(s), with coefficients depending on moments of M.

Parts:
  A) Measure R(l,s) on a fine grid of s values (real axis σ ∈ [1, 4])
  B) Test the hypothesis R-1 ≈ c(s)/l² vs c(s)/l^{2σ}
  C) Measure R on the critical line s = 1/2 + it for varying t
  D) Extract the s-dependence of c(s) and fit to Tr(M²)/l^{2s} prediction
  E) The Euler factor expansion: compare with |1-l^{-s}|^{-1} · |1-l^{-2s}|^{-c}
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


def compute_det(p, q, l, s, base=2):
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


def measure_R(l, s, n_samp=4000, bits=16):
    target = abs(1.0 / (1.0 - l ** (-s)))
    det_vals = []
    for _ in range(n_samp):
        p = random_prime(bits)
        q = random_prime(bits)
        if p == q:
            continue
        d = compute_det(p, q, l, s)
        if d is not None and d > 0 and not math.isnan(d) and not math.isinf(d):
            det_vals.append(d)
    if len(det_vals) < 100:
        return None, None
    mean_det = np.mean(det_vals)
    return mean_det / target, np.std(det_vals) / (math.sqrt(len(det_vals)) * target)


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("P2-01: CHARACTERIZE R(l,s) ANALYTICALLY")
    pr("=" * 72)

    # ═══════════════════════════════════════════════════════════════
    # PART A: R(l,s) ON THE REAL AXIS σ ∈ [1, 4]
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: R(l,σ) FOR REAL s = σ ∈ [1.0, 4.0]")
    pr(f"{'═' * 72}\n")

    sigma_vals = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
    test_primes = [3, 5, 7, 11, 13, 23, 47]

    R_grid = {}
    for l in test_primes:
        pr(f"  l = {l}:")
        for sigma in sigma_vals:
            R, R_err = measure_R(l, sigma)
            if R is None:
                continue
            R_grid[(l, sigma)] = R
            pr(f"    σ = {sigma:.2f}: R = {R:.8f}, "
               f"R-1 = {R-1:+.6f}, "
               f"l²(R-1) = {l*l*(R-1):+.6f}, "
               f"l^{{2σ}}(R-1) = {l**(2*sigma)*(R-1):+.6f}")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART B: TEST R-1 ≈ c/l² vs c/l^{2σ}
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: IS (R-1) ∝ 1/l² OR 1/l^{2σ}?")
    pr(f"{'═' * 72}\n")

    pr("  If (R-1) = c/l², then l²(R-1) is constant across l (at fixed σ).")
    pr("  If (R-1) = c/l^{2σ}, then l^{2σ}(R-1) is constant across l.\n")

    for sigma in [1.5, 2.0, 3.0]:
        pr(f"  σ = {sigma}:")
        l2_products = []
        l2s_products = []
        for l in test_primes:
            R = R_grid.get((l, sigma))
            if R is None:
                continue
            l2_prod = l ** 2 * (R - 1)
            l2s_prod = l ** (2 * sigma) * (R - 1)
            l2_products.append(l2_prod)
            l2s_products.append(l2s_prod)
            pr(f"    l = {l:3d}: l²(R-1) = {l2_prod:+.6f}, "
               f"l^{{2σ}}(R-1) = {l2s_prod:+.6f}")

        arr_l2 = np.array(l2_products)
        arr_l2s = np.array(l2s_products)
        pr(f"    → l²(R-1):    mean = {arr_l2.mean():.6f}, "
           f"std/mean = {arr_l2.std()/abs(arr_l2.mean()):.3f}")
        pr(f"    → l^{{2σ}}(R-1): mean = {arr_l2s.mean():.6f}, "
           f"std/mean = {arr_l2s.std()/abs(arr_l2s.mean()):.3f}")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART C: R ON THE CRITICAL LINE s = 1/2 + it
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: R(l, 1/2 + it) ON THE CRITICAL LINE")
    pr(f"{'═' * 72}\n")

    t_vals = [0.0, 5.0, 10.0, 14.13, 21.02, 30.0, 50.0]

    for l in [3, 5, 7, 11]:
        pr(f"  l = {l}:")
        for t in t_vals:
            s = complex(0.5, t)
            R, R_err = measure_R(l, s)
            if R is None:
                continue
            pr(f"    t = {t:5.1f}: R = {R:.6f}, R-1 = {R-1:+.6f}")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART D: EXTRACT c(σ) AND TEST Tr(M²) PREDICTION
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: c(σ) = l²·(R-1) vs σ — FUNCTIONAL FORM")
    pr(f"{'═' * 72}")
    pr("""
  From the log-expansion:
    ln⟨|det(I-M/l^s)|⟩ ≈ ln|1-l^{-s}|^{-1} + ⟨Tr(M)⟩²/(2l^{2σ}) + ...

  This predicts c(σ) ∝ l^{2-2σ}, i.e., l²(R-1) ∝ l^{2-2σ}.
  At σ=1: l²(R-1) ∝ 1 (constant)
  At σ=2: l²(R-1) ∝ l^{-2} (decreasing) — BUT our data shows constant!

  Alternative: c(σ) truly constant → (R-1) = c₀/l² independent of σ.
  Let's fit c(σ) = l²·(R-1) as a function of σ for fixed l.
""")

    for l in [5, 7, 11, 23]:
        pr(f"  l = {l}: c(σ) = l²·(R-1) vs σ:")
        sigmas = []
        cs = []
        for sigma in sigma_vals:
            R = R_grid.get((l, sigma))
            if R is None:
                continue
            c = l ** 2 * (R - 1)
            sigmas.append(sigma)
            cs.append(c)
            pr(f"    σ = {sigma:.2f}: c = {c:+.6f}")

        if len(sigmas) >= 3:
            s_arr = np.array(sigmas)
            c_arr = np.array(cs)
            coeffs = np.polyfit(s_arr, c_arr, 1)
            pr(f"    Linear fit: c(σ) = {coeffs[0]:.6f}·σ + {coeffs[1]:.6f}")
            positive = c_arr[c_arr > 0]
            if len(positive) >= 2:
                log_c = np.log(positive)
                log_s = np.array(sigmas)[:len(positive)]
                exp_fit = np.polyfit(log_s, log_c, 1)
                pr(f"    Power fit: c ∝ σ^{{{exp_fit[0]:.3f}}}")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART E: EULER FACTOR EXPANSION TEST
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: DOES R(l,s) ≈ |1 - l^{-2s}|^{c₀}?")
    pr(f"{'═' * 72}")
    pr("If R = |1-l^{-2s}|^c, then ln(R) = c·ln|1-l^{-2s}|.\n")

    for l in [3, 5, 7, 11]:
        pr(f"  l = {l}:")
        for sigma in [1.5, 2.0, 2.5, 3.0]:
            R = R_grid.get((l, sigma))
            if R is None:
                continue
            second_factor = abs(1 - l ** (-2 * sigma))
            if R > 0 and second_factor > 0:
                c_eff = math.log(R) / math.log(second_factor) if abs(math.log(second_factor)) > 1e-10 else float('inf')
                pr(f"    σ = {sigma:.1f}: R = {R:.8f}, "
                   f"|1-l^{{-2σ}}| = {second_factor:.6f}, "
                   f"c_eff = {c_eff:.6f}")
        pr()

    # Test on critical line too
    pr("  On critical line (l=5):")
    for t in [0.0, 5.0, 14.13, 30.0]:
        s = complex(0.5, t)
        R, _ = measure_R(5, s, n_samp=5000)
        if R is None:
            continue
        second_factor = abs(1 - 5 ** (-2 * s))
        if R > 0 and second_factor > 0:
            c_eff = math.log(R) / math.log(second_factor) if abs(math.log(second_factor)) > 1e-10 else float('inf')
            pr(f"    s = 0.5+{t:.2f}i: R = {R:.6f}, "
               f"|1-5^{{-2s}}| = {second_factor:.6f}, "
               f"c_eff = {c_eff:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS — R(l,s) CHARACTERIZATION")
    pr(f"{'═' * 72}")
    pr("""
  Key findings:
  1. Is (R-1) ∝ 1/l² or 1/l^{2σ}?
     → Part B answers this definitively.

  2. How does c(σ) depend on σ?
     → Part D reveals the functional form.

  3. Does R factor as |1-l^{-2s}|^{c₀}?
     → Part E tests if R is a power of the SECOND Euler factor.

  The ultimate goal: find an EXACT formula R(l,s) = f(l,s) such that
    ⟨|det(I - M_l/l^s)|⟩ / R(l,s) = |1 - l^{-s}|^{-1}  exactly.
""")

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
