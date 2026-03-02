#!/usr/bin/env python3
"""
B27: Jensen Gap Test — Is the GEOMETRIC Mean the Exact Euler Factor?

Core hypothesis: ⟨|det|⟩ ≠ |1-l^{-s}|^{-1} because of Jensen's inequality.
But exp(⟨ln|det|⟩) MIGHT equal |1-l^{-s}|^{-1} exactly.

If true, the correct carry-ζ identity is:
  ∏_l exp(⟨ln|det(I - M_l/l^s)|⟩) = |ζ(s)|

This would close Gap 1 and provide the exact framework for RH.

Strategy:
  A) Compute arithmetic mean ⟨|det|⟩ AND geometric mean exp(⟨ln|det|⟩)
     for many primes l and several σ values
  B) Compare both to |1-l^{-σ}|^{-1} (the Euler target)
  C) Measure Jensen gap = ln⟨|det|⟩ - ⟨ln|det|⟩ and check if it equals c₁/l^σ
  D) Compute the partial product ∏_{l≤L} and compare to ζ(σ)
  E) Test on the critical line s = 1/2 + it
"""

import sys, os, time, random, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, carry_poly_int, quotient_poly_int, is_prime

random.seed(42)
np.random.seed(42)


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def compute_det(p, q, l, s, base=2):
    """Compute |det(I - M_l * l^{-s})| for semiprime N=p*q."""
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


def zeta_partial(sigma, primes):
    """Compute partial ζ(σ) = ∏_{l in primes} (1 - l^{-σ})^{-1}."""
    prod = 1.0
    for l in primes:
        prod /= (1 - l ** (-sigma))
    return prod


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("P4-01: JENSEN GAP — GEOMETRIC MEAN = EXACT EULER FACTOR?")
    pr("=" * 72)
    pr("""
  Key question: does exp(⟨ln|det|⟩) = |1-l^{-s}|^{-1} exactly?

  If yes: the ARITHMETIC mean overshoots by the Jensen gap,
  and the GEOMETRIC mean gives the exact Euler product.

  Prediction: Jensen gap ≈ c₁/l^σ ≈ ln(2)/(4·l^σ)
""")

    # ═══════════════════════════════════════════════════════════════
    # PART A: ARITHMETIC vs GEOMETRIC MEAN — COMPARISON TO EULER
    # ═══════════════════════════════════════════════════════════════
    pr(f"{'═' * 72}")
    pr("PART A: ARITHMETIC vs GEOMETRIC MEAN vs EULER TARGET")
    pr(f"{'═' * 72}\n")

    bits = 20
    n_samp = 20000
    sigma_vals = [1.5, 2.0, 2.5, 3.0, 4.0]
    test_primes = [3, 5, 7, 11, 13, 17, 23, 31, 47, 67, 97]

    semiprimes = []
    for _ in range(n_samp):
        p = random_prime(bits)
        q = random_prime(bits)
        if p != q:
            semiprimes.append((p, q))

    for sigma in sigma_vals:
        pr(f"\n  σ = {sigma}:")
        pr(f"  {'l':>5s}  {'⟨|det|⟩':>12s}  {'exp⟨ln⟩':>12s}  "
           f"{'Euler':>12s}  {'R_arith':>10s}  {'R_geom':>10s}  "
           f"{'Jensen':>10s}  {'c₁/l^σ':>10s}")
        pr(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  "
           f"{'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

        for l in test_primes:
            euler_target = abs(1.0 / (1.0 - l ** (-sigma)))
            c1_pred = math.log(2) / (4 * l ** sigma)

            dets = []
            log_dets = []
            for p, q in semiprimes:
                d = compute_det(p, q, l, sigma)
                if d is not None and d > 0 and not math.isnan(d) and not math.isinf(d):
                    dets.append(d)
                    log_dets.append(math.log(d))

            if len(dets) < 1000:
                continue

            arith_mean = np.mean(dets)
            geom_mean = math.exp(np.mean(log_dets))
            R_arith = arith_mean / euler_target
            R_geom = geom_mean / euler_target
            jensen_gap = math.log(arith_mean) - np.mean(log_dets)

            pr(f"  {l:5d}  {arith_mean:12.6f}  {geom_mean:12.6f}  "
               f"{euler_target:12.6f}  {R_arith:10.6f}  {R_geom:10.6f}  "
               f"{jensen_gap:10.6f}  {c1_pred:10.6f}")

    # ═══════════════════════════════════════════════════════════════
    # PART B: PRECISION TEST — R_geom vs 1.0 AT LARGE l
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: R_geom = exp(⟨ln|det|⟩) / Euler — IS IT 1.000?")
    pr(f"{'═' * 72}\n")

    sigma = 2.0
    bits_list = [16, 20, 24, 28, 32]

    for bits in bits_list:
        pr(f"  bits={bits} (σ={sigma}):")
        semiprimes_b = []
        for _ in range(15000):
            p = random_prime(bits)
            q = random_prime(bits)
            if p != q:
                semiprimes_b.append((p, q))

        for l in [5, 11, 23, 47, 97]:
            euler_target = abs(1.0 / (1.0 - l ** (-sigma)))
            dets = []
            log_dets = []
            for p, q in semiprimes_b:
                d = compute_det(p, q, l, sigma)
                if d is not None and d > 0 and not math.isnan(d) and not math.isinf(d):
                    dets.append(d)
                    log_dets.append(math.log(d))
            if len(dets) < 500:
                continue
            arith = np.mean(dets)
            geom = math.exp(np.mean(log_dets))
            R_a = arith / euler_target
            R_g = geom / euler_target
            se_log = np.std(log_dets) / math.sqrt(len(log_dets))
            pr(f"    l={l:3d}: R_arith={R_a:.6f}, R_geom={R_g:.6f}, "
               f"|R_geom-1|={abs(R_g-1):.6f}, σ(ln)={se_log:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # PART C: JENSEN GAP = c₁/l^σ ?
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: JENSEN GAP = c₁/l^σ = ln(2)/(4·l^σ) ?")
    pr(f"{'═' * 72}\n")

    bits = 24
    semiprimes_c = []
    for _ in range(20000):
        p = random_prime(bits)
        q = random_prime(bits)
        if p != q:
            semiprimes_c.append((p, q))

    for sigma in [2.0, 2.5, 3.0]:
        pr(f"  σ = {sigma}:")
        for l in [3, 5, 7, 11, 23, 47, 97]:
            dets = []
            log_dets = []
            for p, q in semiprimes_c:
                d = compute_det(p, q, l, sigma)
                if d is not None and d > 0 and not math.isnan(d) and not math.isinf(d):
                    dets.append(d)
                    log_dets.append(math.log(d))
            if len(dets) < 1000:
                continue
            jensen = math.log(np.mean(dets)) - np.mean(log_dets)
            c1_pred = math.log(2) / (4 * l ** sigma)
            ratio = jensen / c1_pred if c1_pred > 1e-10 else float('inf')
            pr(f"    l={l:3d}: Jensen={jensen:.8f}, "
               f"c₁/l^σ={c1_pred:.8f}, ratio={ratio:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PART D: PARTIAL EULER PRODUCT
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: PARTIAL EULER PRODUCT ∏_{l≤L} exp(⟨ln|det|⟩)")
    pr(f"{'═' * 72}\n")

    bits = 20
    semiprimes_d = []
    for _ in range(15000):
        p = random_prime(bits)
        q = random_prime(bits)
        if p != q:
            semiprimes_d.append((p, q))

    primes_list = [l for l in range(2, 100) if is_prime(l)]

    for sigma in [2.0, 3.0]:
        pr(f"  σ = {sigma}:")
        prod_arith = 1.0
        prod_geom = 1.0
        zeta_true = zeta_partial(sigma, primes_list)

        for idx, l in enumerate(primes_list):
            euler_factor = 1.0 / (1.0 - l ** (-sigma))
            dets = []
            log_dets = []
            for p, q in semiprimes_d:
                d = compute_det(p, q, l, sigma)
                if d is not None and d > 0 and not math.isnan(d) and not math.isinf(d):
                    dets.append(d)
                    log_dets.append(math.log(d))
            if len(dets) < 500:
                continue

            arith = np.mean(dets)
            geom = math.exp(np.mean(log_dets))

            prod_arith *= arith
            prod_geom *= geom
            zeta_partial_val = zeta_partial(sigma, primes_list[:idx + 1])

            if l in [2, 3, 5, 11, 23, 47, 97] or l == primes_list[-1]:
                pr(f"    l≤{l:3d}: ∏arith={prod_arith:12.6f}, "
                   f"∏geom={prod_geom:12.6f}, "
                   f"ζ_partial={zeta_partial_val:12.6f}, "
                   f"∏geom/ζ={prod_geom/zeta_partial_val:.6f}")

        pr(f"    ζ({sigma}) partial (l≤97) = {zeta_true:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # PART E: CRITICAL LINE s = 1/2 + it
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: CRITICAL LINE s = 1/2 + it")
    pr(f"{'═' * 72}\n")

    bits = 16
    semiprimes_e = []
    for _ in range(15000):
        p = random_prime(bits)
        q = random_prime(bits)
        if p != q:
            semiprimes_e.append((p, q))

    for t_val in [0, 5.0, 14.134725, 21.022]:
        s = complex(0.5, t_val)
        pr(f"  s = 1/2 + {t_val:.3f}i:")

        for l in [3, 5, 11, 23, 47]:
            euler_factor = abs(1.0 / (1.0 - l ** (-s)))
            dets = []
            log_dets = []
            for p, q in semiprimes_e:
                d = compute_det(p, q, l, s)
                if d is not None and d > 0 and not math.isnan(d) and not math.isinf(d):
                    dets.append(d)
                    log_dets.append(math.log(d))
            if len(dets) < 500:
                continue
            arith = np.mean(dets)
            geom = math.exp(np.mean(log_dets))
            R_a = arith / euler_factor
            R_g = geom / euler_factor
            pr(f"    l={l:3d}: R_arith={R_a:.6f}, R_geom={R_g:.6f}, "
               f"|R_geom-1|={abs(R_g-1):.6f}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS")
    pr(f"{'═' * 72}")
    pr("""
  Three possible outcomes:

  1. R_geom ≡ 1: The geometric mean IS the exact Euler factor.
     ⟹ The correct identity is ∏_l exp(⟨ln|det|⟩) = |ζ(s)|.
     This closes Gap 1 and provides the framework for RH.

  2. R_geom ≠ 1 but ∏ R_geom → known function:
     ⟹ The product has a correction that needs to be understood.
     May still lead to RH via a modified identity.

  3. R_geom ≈ 1 but not exactly:
     ⟹ The carry framework is an approximation, not an identity.
     Useful for heuristics but not a proof of RH.
""")

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
