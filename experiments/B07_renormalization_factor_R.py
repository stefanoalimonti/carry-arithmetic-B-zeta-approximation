#!/usr/bin/env python3
"""
B07: The Renormalization Factor R(l, s).

From prior experiments we know that the spectral determinant det(I - M/l^s) captures
Riemann zeros better than the root-count L-function, but the two differ
by a factor R(l, s) that converges to 1 as l → ∞.

Strategy:
  1. Measure R(l) = <|det(I - M/l)|> / (1 - N_l/l) for many primes l
  2. Fit R(l) analytically: power law? 1 + c/l? 1 + c·log(l)/l?
  3. Measure R(l, s) on the critical line for several t values
  4. Test factorization: R(l, s) = g(l)·h(s)?
  5. Build the renormalized product and test Riemann zero recovery
  6. Identify connection to ζ(2)
"""

import sys
import os
import random
import math
import cmath
import numpy as np
from numpy.linalg import eig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import (
    random_prime, carry_poly_int, quotient_poly_int,
    primes_up_to, poly_roots_mod
)

random.seed(42)
np.random.seed(42)

BASE = 2
HALF_BITS = 16
N_SEMIPRIMES = 300

RIEMANN_ZEROS_T = [
    14.135, 21.022, 25.011, 30.425, 32.935, 37.586, 40.919,
    43.327, 48.005, 49.774, 52.970, 56.446, 59.347, 60.832,
    65.113, 67.080, 69.546, 72.067, 75.705, 77.145,
]


def companion_matrix(poly):
    n = len(poly) - 1
    if n < 1:
        return None
    M = np.zeros((n, n), dtype=complex)
    for i in range(n - 1):
        M[i + 1, i] = 1.0
    lead = poly[-1]
    if abs(lead) < 1e-30:
        return None
    for i in range(n):
        M[i, n - 1] = -poly[i] / lead
    if not np.all(np.isfinite(M)):
        return None
    return M


def spectral_det(eigenvalues, z):
    result = complex(1.0, 0.0)
    for lam in eigenvalues:
        result *= (1.0 - lam * z)
    return result


def main():
    print("=" * 70)
    print("B07: THE RENORMALIZATION FACTOR R(l, s)")
    print("=" * 70)

    test_primes = primes_up_to(500)
    test_primes = [l for l in test_primes if l > 2]

    # ─── Phase 1: Build operators ───
    print(f"\n─── Phase 1: Building {N_SEMIPRIMES} carry operators ───")
    operators = []
    for trial in range(N_SEMIPRIMES):
        if trial % 100 == 0:
            print(f"  trial {trial}...")
        p = random_prime(HALF_BITS)
        q = random_prime(HALF_BITS)
        while q == p:
            q = random_prime(HALF_BITS)
        C = carry_poly_int(p, q, BASE)
        Q = quotient_poly_int(C, BASE)
        M = companion_matrix(Q)
        if M is None:
            continue
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigenvalues = eig(M)[0]
        if not np.all(np.isfinite(eigenvalues)):
            continue
        operators.append({'Q': Q, 'evals': eigenvalues, 'dim': M.shape[0]})

    print(f"  Valid: {len(operators)}")
    mean_dim = np.mean([op['dim'] for op in operators])
    print(f"  Mean dimension: {mean_dim:.1f}")

    # ─── Phase 2: Measure R(l) = R(l, s=1) for many primes ───
    print(f"\n─── Phase 2: R(l) = <|det(I - M/l)|> / |1 - N_l/l| for {len(test_primes)} primes ───")

    Nl_map = {}
    for l in test_primes:
        counts = []
        for op in operators:
            roots = poly_roots_mod(op['Q'], l)
            counts.append(len(roots))
        Nl_map[l] = np.mean(counts)

    R_values = {}
    det_values = {}
    root_values = {}

    print(f"\n  {'l':>5s}  {'dim':>4s}  {'<|det|>':>10s}  {'|1-Nl/l|':>10s}  "
          f"{'R(l)':>8s}  {'Nl/l':>6s}  {'log(R)':>8s}")
    print(f"  {'-'*62}")

    for l in test_primes[:80]:
        euler_spec = []
        for op in operators:
            z = complex(1.0 / l, 0)
            sd = spectral_det(op['evals'], z)
            euler_spec.append(abs(sd))

        mean_det = np.mean(euler_spec)
        nl_frac = Nl_map[l] / l
        root_factor = abs(1.0 - nl_frac)

        if root_factor > 1e-10:
            R = mean_det / root_factor
        else:
            R = float('inf')

        R_values[l] = R
        det_values[l] = mean_det
        root_values[l] = root_factor

        if l <= 101 or l in [149, 197, 251, 307, 397, 499]:
            print(f"  {l:>5d}  {mean_dim:>4.0f}  {mean_det:>10.6f}  {root_factor:>10.6f}  "
                  f"{R:>8.4f}  {nl_frac:>6.3f}  {math.log(R):>8.4f}")

    # ─── Phase 3: Fit R(l) ───
    print(f"\n─── Phase 3: Fitting R(l) — which model? ───")

    primes_arr = np.array(sorted(R_values.keys()))
    R_arr = np.array([R_values[l] for l in primes_arr])
    valid = np.isfinite(R_arr) & (R_arr > 0) & (R_arr < 100)
    primes_fit = primes_arr[valid]
    R_fit = R_arr[valid]

    log_l = np.log(primes_fit)
    log_R = np.log(R_fit)
    R_minus_1 = R_fit - 1.0

    models = {}

    # Model 1: R(l) = 1 + a/l
    if len(primes_fit) > 2:
        inv_l = 1.0 / primes_fit
        a_fit = np.polyfit(inv_l, R_minus_1, 1)
        pred_1 = 1.0 + a_fit[0] * inv_l + a_fit[1]
        resid_1 = np.sum((R_fit - pred_1)**2)
        models['1 + a/l'] = {'params': a_fit, 'residual': resid_1, 'a': a_fit[0]}
        print(f"  Model 1: R(l) = 1 + {a_fit[0]:.4f}/l + {a_fit[1]:.6f}")
        print(f"    Residual: {resid_1:.6f}")

    # Model 2: R(l) = l^α  (power law from 1)
    if len(log_l) > 2:
        alpha_fit = np.polyfit(log_l, log_R, 1)
        pred_2 = np.exp(alpha_fit[0] * log_l + alpha_fit[1])
        resid_2 = np.sum((R_fit - pred_2)**2)
        models['l^α'] = {'params': alpha_fit, 'residual': resid_2, 'alpha': alpha_fit[0]}
        print(f"  Model 2: R(l) = exp({alpha_fit[1]:.4f}) · l^{alpha_fit[0]:.4f}")
        print(f"    Residual: {resid_2:.6f}")

    # Model 3: R(l) = 1 + c·dim/l  (dimension-dependent)
    if len(primes_fit) > 2:
        c_dim = np.polyfit(mean_dim / primes_fit, R_minus_1, 1)
        pred_3 = 1.0 + c_dim[0] * mean_dim / primes_fit + c_dim[1]
        resid_3 = np.sum((R_fit - pred_3)**2)
        models['1 + c·d/l'] = {'params': c_dim, 'residual': resid_3, 'c': c_dim[0]}
        print(f"  Model 3: R(l) = 1 + {c_dim[0]:.4f}·dim/l + {c_dim[1]:.6f}")
        print(f"    Residual: {resid_3:.6f}")

    # Model 4: R(l) = 1 + c·log(l)/l
    if len(primes_fit) > 2:
        log_over_l = np.log(primes_fit) / primes_fit
        c_log = np.polyfit(log_over_l, R_minus_1, 1)
        pred_4 = 1.0 + c_log[0] * log_over_l + c_log[1]
        resid_4 = np.sum((R_fit - pred_4)**2)
        models['1 + c·log(l)/l'] = {'params': c_log, 'residual': resid_4, 'c': c_log[0]}
        print(f"  Model 4: R(l) = 1 + {c_log[0]:.4f}·log(l)/l + {c_log[1]:.6f}")
        print(f"    Residual: {resid_4:.6f}")

    # Model 5: R(l) = (1 - 1/l)^{-α} ≈ 1 + α/l for large l
    if len(primes_fit) > 2:
        neg_log_1ml = -np.log(1 - 1.0 / primes_fit)
        alpha_5 = np.polyfit(neg_log_1ml, log_R, 1)
        pred_5 = (1 - 1.0 / primes_fit) ** (-alpha_5[0]) * np.exp(alpha_5[1])
        resid_5 = np.sum((R_fit - pred_5)**2)
        models['(1-1/l)^{-α}'] = {'params': alpha_5, 'residual': resid_5, 'alpha': alpha_5[0]}
        print(f"  Model 5: R(l) = exp({alpha_5[1]:.4f}) · (1 - 1/l)^{{-{alpha_5[0]:.4f}}}")
        print(f"    Residual: {resid_5:.6f}")

    best_model = min(models.keys(), key=lambda k: models[k]['residual'])
    print(f"\n  ★ Best model: {best_model} (residual {models[best_model]['residual']:.6f})")

    # Check if key constant relates to ζ(2), dim, etc.
    print(f"\n  Key constant analysis:")
    if '1 + a/l' in models:
        a = models['1 + a/l']['a']
        print(f"    a = {a:.4f}")
        print(f"    a/dim = {a/mean_dim:.4f}")
        print(f"    a vs dim-1 = {a:.4f} vs {mean_dim-1:.1f}")
        print(f"    a vs ζ(2)·dim = {a:.4f} vs {(math.pi**2/6)*mean_dim:.4f}")
        print(f"    a vs (dim-1)/2 = {a:.4f} vs {(mean_dim-1)/2:.4f}")

    # ─── Phase 4: R(l, s) on the critical line ───
    print(f"\n─── Phase 4: R(l, s) dependence on s = 1/2 + it ───")

    t_test = [0.0, 5.0, 14.135, 21.022, 30.0, 50.0]

    print(f"\n  {'l':>5s}", end="")
    for t in t_test:
        label = f"t={t:.1f}"
        print(f"  {label:>10s}", end="")
    print()
    print(f"  {'-'*(5 + 12*len(t_test))}")

    R_matrix = {}
    for l in test_primes[:40]:
        R_row = {}
        for t in t_test:
            s = complex(0.5, t)
            z = BASE ** (-s) / l ** s if t != 0 else complex(1.0 / l, 0)

            euler_spec = []
            for op in operators:
                sd = spectral_det(op['evals'], complex(1.0, 0) / l ** s)
                euler_spec.append(abs(sd))
            mean_det_s = np.mean(euler_spec)

            nl_frac = Nl_map[l] / l
            al = nl_frac
            root_euler_s = abs(1.0 - complex(al, 0) / l ** s)

            R_s = mean_det_s / root_euler_s if root_euler_s > 1e-10 else float('inf')
            R_row[t] = R_s

        R_matrix[l] = R_row

        if l <= 53 or l in [97, 197, 397]:
            print(f"  {l:>5d}", end="")
            for t in t_test:
                print(f"  {R_row[t]:>10.4f}", end="")
            print()

    # ─── Phase 5: Factorization test R(l,s) = g(l)·h(s)? ───
    print(f"\n─── Phase 5: Does R(l, s) factor as g(l)·h(s)? ───")

    factorization_test = []
    for l in test_primes[:30]:
        if l not in R_matrix:
            continue
        row = R_matrix[l]
        if row[0.0] > 0 and all(row[t] > 0 for t in t_test):
            normalized = [row[t] / row[0.0] for t in t_test]
            factorization_test.append(normalized)

    if factorization_test:
        fact_arr = np.array(factorization_test)
        col_std = np.std(fact_arr, axis=0)
        col_mean = np.mean(fact_arr, axis=0)
        print(f"  If R factors, R(l,t)/R(l,0) should be independent of l:")
        for i, t in enumerate(t_test):
            print(f"    t={t:6.1f}: mean ratio = {col_mean[i]:.4f} ± {col_std[i]:.4f} "
                  f"(CV = {col_std[i]/max(col_mean[i],1e-10)*100:.1f}%)")

        mean_cv = np.mean(col_std / np.maximum(col_mean, 1e-10))
        if mean_cv < 0.1:
            print(f"  → R(l,s) FACTORS as g(l)·h(s) with CV < 10%!")
        elif mean_cv < 0.3:
            print(f"  → APPROXIMATE factorization (CV = {mean_cv*100:.1f}%)")
        else:
            print(f"  → R(l,s) does NOT factor cleanly (CV = {mean_cv*100:.1f}%)")

    # ─── Phase 6: Build renormalized product and test Riemann zeros ───
    print(f"\n─── Phase 6: Renormalized Euler product → Riemann zeros ───")

    # Use best-fit R(l) to renormalize
    best = models[best_model]
    t_scan = np.linspace(1, 80, 2000)

    # Raw spectral product
    raw_product = np.zeros(len(t_scan))
    # Renormalized product
    renorm_product = np.zeros(len(t_scan))
    # Root-count L
    rootcount_L = np.zeros(len(t_scan))

    use_primes = test_primes[:50]

    for idx, t in enumerate(t_scan):
        s = complex(0.5, t)
        log_raw = 0.0
        log_renorm = 0.0
        log_root = 0.0

        for l in use_primes:
            # Spectral det Euler factor
            euler_vals = []
            for op in operators[:100]:
                sd = spectral_det(op['evals'], complex(1.0, 0) / l ** s)
                euler_vals.append(abs(sd))
            mean_euler = np.mean(euler_vals)
            log_raw += math.log(max(mean_euler, 1e-30))

            # Renormalization: divide by R(l)
            if best_model == '1 + a/l':
                R_l = 1.0 + best['a'] / l + best['params'][1]
            elif best_model == 'l^α':
                R_l = math.exp(best['params'][1]) * l ** best['alpha']
            elif best_model == '1 + c·d/l':
                R_l = 1.0 + best['c'] * mean_dim / l + best['params'][1]
            elif best_model == '1 + c·log(l)/l':
                R_l = 1.0 + best['c'] * math.log(l) / l + best['params'][1]
            else:
                R_l = 1.0

            R_l = max(R_l, 0.01)
            log_renorm += math.log(max(mean_euler / R_l, 1e-30))

            # Root-count factor
            al = Nl_map[l] / l
            root_euler = abs(1.0 - complex(al, 0) / l ** s)
            log_root += math.log(max(root_euler, 1e-30))

        raw_product[idx] = log_raw
        renorm_product[idx] = log_renorm
        rootcount_L[idx] = log_root

    # Find minima in each
    def find_mins(vals, t_arr, threshold_pct=30):
        mins = []
        for i in range(1, len(vals) - 1):
            if vals[i] < vals[i-1] and vals[i] < vals[i+1]:
                if vals[i] < np.percentile(vals, threshold_pct):
                    mins.append(t_arr[i])
        return mins

    raw_zeros = find_mins(raw_product, t_scan)
    renorm_zeros = find_mins(renorm_product, t_scan)
    root_zeros = find_mins(rootcount_L, t_scan)

    print(f"  Pseudo-zeros found:")
    print(f"    Raw spectral:     {len(raw_zeros)}")
    print(f"    Renormalized:     {len(renorm_zeros)}")
    print(f"    Root-count:       {len(root_zeros)}")

    print(f"\n  {'Riemann t':>12s}  {'Raw Δ':>8s}  {'Renorm Δ':>10s}  {'Root Δ':>8s}  {'Best':>8s}")
    print(f"  {'-'*52}")

    raw_hits = 0
    renorm_hits = 0
    root_hits = 0

    for t_r in RIEMANN_ZEROS_T:
        d_raw = min([abs(z - t_r) for z in raw_zeros]) if raw_zeros else 99
        d_ren = min([abs(z - t_r) for z in renorm_zeros]) if renorm_zeros else 99
        d_root = min([abs(z - t_r) for z in root_zeros]) if root_zeros else 99

        if d_raw < 3: raw_hits += 1
        if d_ren < 3: renorm_hits += 1
        if d_root < 3: root_hits += 1

        best_d = min(d_raw, d_ren, d_root)
        tag = ""
        if best_d == d_ren and d_ren < 3: tag = "★Renorm"
        elif best_d == d_raw and d_raw < 3: tag = "Raw"
        elif best_d == d_root and d_root < 3: tag = "Root"

        print(f"  {t_r:>12.3f}  {d_raw:>8.3f}  {d_ren:>10.3f}  {d_root:>8.3f}  {tag:>8s}")

    n_tested = len(RIEMANN_ZEROS_T)
    print(f"\n  RIEMANN ZERO RECOVERY:")
    print(f"    Raw spectral:   {raw_hits}/{n_tested} ({raw_hits/n_tested*100:.0f}%)")
    print(f"    Renormalized:   {renorm_hits}/{n_tested} ({renorm_hits/n_tested*100:.0f}%)")
    print(f"    Root-count:     {root_hits}/{n_tested} ({root_hits/n_tested*100:.0f}%)")

    if renorm_hits > raw_hits:
        improvement = (renorm_hits - raw_hits) / max(raw_hits, 1) * 100
        print(f"\n  ★★★ Renormalization IMPROVES zero recovery by {improvement:.0f}%!")
    elif renorm_hits == raw_hits:
        print(f"\n  Renormalization preserves zero recovery")
    else:
        print(f"\n  Renormalization does not improve (raw is already better)")

    # ─── Phase 7: Correlation improvement ───
    print(f"\n─── Phase 7: Correlation with root-count L ───")

    corr_raw = np.corrcoef(raw_product, rootcount_L)[0, 1]
    corr_renorm = np.corrcoef(renorm_product, rootcount_L)[0, 1]

    print(f"  Corr(raw, root-count):      {corr_raw:+.6f}")
    print(f"  Corr(renorm, root-count):   {corr_renorm:+.6f}")

    if abs(corr_renorm) > abs(corr_raw):
        print(f"  → Renormalization INCREASES alignment with L_carry")
    else:
        print(f"  → Raw spectral product already better aligned")

    # Spectral coherence
    fft_ren = np.fft.rfft(renorm_product - np.mean(renorm_product))
    fft_root = np.fft.rfft(rootcount_L - np.mean(rootcount_L))
    cross = fft_ren * np.conj(fft_root)
    coherence = np.abs(cross)**2 / (np.abs(fft_ren)**2 * np.abs(fft_root)**2 + 1e-30)
    print(f"  Spectral coherence (renorm ↔ root): {np.mean(coherence[1:100]):.4f}")

    np.savez(os.path.join(os.path.dirname(__file__), "e_r9_data.npz"),
             primes_fit=primes_fit, R_fit=R_fit,
             best_model=best_model,
             raw_hits=raw_hits, renorm_hits=renorm_hits, root_hits=root_hits)
    print("\nData saved to e_r9_data.npz")


if __name__ == "__main__":
    main()
