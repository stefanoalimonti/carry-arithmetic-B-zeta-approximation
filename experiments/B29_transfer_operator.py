#!/usr/bin/env python3
"""
B29_transfer_operator.py — Transfer operator analysis for c₁ = ln(2)/4

Approach B: Construct the Markov transfer operator whose iteration
approximates the exact enumeration of carry polynomials.

The transfer operator acts on carry values at each bit-position offset
from the MSB.  At offset j the convolution conv[j] has a known marginal
distribution, and the carry update is

    carry_at[j-1] = floor( (conv[j] + carry_at[j]) / 2 ).

The MARKOV APPROXIMATION treats convolutions at different offsets as
independent of the carry history (ignoring shared-bit correlations).

Key questions addressed:
  1. Does the Markov approximation give ln(2)/4 in the limit?
  2. What is the spectral gap / convergence rate?
  3. How do eigenvalues relate to rho ~ 0.64?
"""

import numpy as np
import math
import sys
import time
from scipy.special import comb

LN2 = math.log(2)
LN2_OVER_4 = LN2 / 4

EXACT_C1 = {
    1: 2.750, 2: 1.190, 3: 0.517, 4: 0.431, 5: 0.375, 6: 0.321,
    7: 0.27716, 8: 0.24230, 9: 0.21789, 10: 0.20157,
    11: 0.19102, 12: 0.18437, 13: 0.18030, 14: 0.17788,
}


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════
# CONVOLUTION DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════

def v_dist_correct(j, v_max):
    """Marginal distribution of conv[j] at offset j from the MSB.

    conv[0] = 1   (deterministic, g'_0 h'_0 = 1)
    conv[1] = a_1 + b_1           ~ Binom(2, 1/2)
    conv[j] = a_j + b_j + sum_{i=1}^{j-1} a_i b_{j-i}   for j >= 2
            ~ Binom(2, 1/2) + Binom(j-1, 1/4)
    """
    dist = np.zeros(v_max + 1)
    if j == 0:
        if v_max >= 1:
            dist[1] = 1.0
        return dist
    if j == 1:
        for v in range(min(3, v_max + 1)):
            dist[v] = comb(2, v, exact=True) * 0.25  # 0.5^2
        return dist
    n_cross = j - 1
    for s1 in range(3):
        p1 = comb(2, s1, exact=True) * 0.25
        for s2 in range(min(n_cross + 1, v_max + 1)):
            p2 = float(comb(n_cross, s2, exact=True)) * 0.25**s2 * 0.75**(n_cross - s2)
            v = s1 + s2
            if v <= v_max:
                dist[v] += p1 * p2
    return dist


def v_dist_naive(j, v_max):
    """Naive approximation: conv[j] ~ Binom(j+1, 1/4)."""
    dist = np.zeros(v_max + 1)
    n = j + 1
    for v in range(min(n + 1, v_max + 1)):
        dist[v] = float(comb(n, v, exact=True)) * 0.25**v * 0.75**(n - v)
    return dist


# ══════════════════════════════════════════════════════════════════════
# TRANSFER MATRIX CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════

def build_transfer(v_dist, c_max):
    """T[c', c] = P(floor((v + c)/2) = c') for given v distribution."""
    n = c_max + 1
    T = np.zeros((n, n))
    for c in range(n):
        for v in range(len(v_dist)):
            if v_dist[v] < 1e-30:
                continue
            c_new = min((v + c) // 2, c_max)
            T[c_new, c] += v_dist[v]
    return T


def bulk_stationary(n_cross, c_max):
    """Stationary distribution of the interior carry chain
    with v ~ Binom(n_cross, 1/4)."""
    v = np.zeros(n_cross + 1)
    for k in range(n_cross + 1):
        v[k] = float(comb(n_cross, k, exact=True)) * 0.25**k * 0.75**(n_cross - k)
    T = build_transfer(v, c_max)
    pi = np.ones(c_max + 1) / (c_max + 1)
    for _ in range(500_000):
        pi_new = T @ pi
        pi_new /= pi_new.sum()
        if np.max(np.abs(pi_new - pi)) < 1e-16:
            break
        pi = pi_new
    return pi


# ══════════════════════════════════════════════════════════════════════
# c₁(K) VIA MARKOV TRANSFER OPERATOR
# ══════════════════════════════════════════════════════════════════════

def compute_c1_markov(K, c_max, model='correct'):
    """Compute c₁(K) using the Markov transfer-operator approximation.

    Parameters
    ----------
    K : int       – window size (number of explicit top-bit offsets)
    c_max : int   – truncation of carry state space {0, ..., c_max}
    model : str   – 'correct' or 'naive'

    Returns
    -------
    dict with c1, P_even, P_odd, E_ctop1, and diagnostics
    """
    n = c_max + 1
    c_vals = np.arange(n, dtype=np.float64)

    n_bulk = max(K + 10, 30)
    pi_entry = bulk_stationary(n_bulk, c_max)

    vfunc = v_dist_correct if model == 'correct' else v_dist_naive
    v_max_build = min(2 * c_max, 80)

    transfers = {}
    for j in range(1, K + 1):
        vd = vfunc(j, v_max_build)
        transfers[j] = build_transfer(vd, c_max)

    # Propagate carry distribution:  pi[j] = transfers[j+1] @ pi[j+1]
    pi_at = [None] * (K + 1)
    pi_at[K] = pi_entry.copy()
    for j in range(K - 1, -1, -1):
        pi_at[j] = transfers[j + 1] @ pi_at[j + 1]

    P_even = 1.0 - pi_at[0][0]
    P_odd = pi_at[0][0]

    # D-even contribution:  c_{top-1} = carry_at[0]  for carry_at[0] >= 1
    E_even = np.dot(c_vals, pi_at[0])   # c=0 term contributes 0

    # D-odd contribution via cascade depth J
    #   beta_J[c] = P(carry_at[0]=...=carry_at[J-1]=0 | carry_at[J]=c)
    #             = transfers[J][0, c] * prod_{k=1}^{J-1} transfers[k][0, 0]
    cum_p = np.empty(K + 1)
    cum_p[0] = 1.0
    for k in range(1, K + 1):
        cum_p[k] = cum_p[k - 1] * transfers[k][0, 0]

    D_odd_sum = 0.0

    for J in range(1, K):
        beta_J = transfers[J][0, :] * cum_p[J - 1]
        # c_{top-1} = carry_at[J+1]  for cascade depth J
        w = c_vals * pi_at[J + 1]                 # carry-weighted distribution
        inner = transfers[J + 1] @ w              # propagated through T_{J+1}
        D_odd_sum += np.dot(beta_J[1:], inner[1:])  # sum over c_J >= 1

    # Deep cascade: all carry_at[0..K-1] = 0 → c_{top-1} = carry_at[K]
    beta_K = transfers[K][0, :] * cum_p[K - 1]
    D_odd_sum += np.dot(beta_K, c_vals * pi_at[K])

    E_ctop1 = E_even + D_odd_sum
    c1 = E_ctop1 - 1.0

    return {
        'c1': c1,
        'P_even': P_even, 'P_odd': P_odd,
        'E_ctop1': E_ctop1,
        'E_even': E_even / P_even if P_even > 1e-15 else 0.0,
        'pi_at': pi_at,
        'transfers': transfers,
        'cum_p': cum_p,
    }


# ══════════════════════════════════════════════════════════════════════
# EIGENVALUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def eigenvalue_analysis(transfers, K, c_max):
    """Eigenvalues of individual T_j and of the product T_1 · T_2 · ... · T_K."""
    n = c_max + 1

    # M_K = T_1 @ T_2 @ ... @ T_K  (T_K acts first on the vector)
    product = np.eye(n)
    for k in range(K, 0, -1):
        product = transfers[k] @ product

    ev_product = np.sort(np.abs(np.linalg.eigvals(product)))[::-1]

    return ev_product


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    c_max = 20
    K_max = 50

    pr("=" * 80)
    pr("  B29: TRANSFER OPERATOR ANALYSIS — c₁ = ln(2)/4")
    pr("=" * 80)
    pr(f"  Target: ln(2)/4 = {LN2_OVER_4:.12f}")
    pr(f"  State space: carry values 0..{c_max}")
    pr(f"  Window sizes: K = 1..{K_max}")

    # ── PART 1: Convolution distributions ─────────────────────────
    pr(f"\n{'═' * 80}")
    pr("  PART 1: CONVOLUTION DISTRIBUTIONS AT EACH OFFSET")
    pr(f"{'═' * 80}\n")
    pr(f"  {'j':>3s}  {'model':>8s}  {'E[v]':>8s}  {'Var[v]':>8s}  {'P(v=0)':>8s}  "
       f"{'P(v=1)':>8s}  {'P(v=2)':>8s}  {'P(v=3)':>8s}")
    pr(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for j in range(8):
        for model_name, vfunc in [('correct', v_dist_correct), ('naive', v_dist_naive)]:
            vd = vfunc(j, 40)
            vals = np.arange(len(vd))
            ev = np.dot(vals, vd)
            var_v = np.dot(vals**2, vd) - ev**2
            pv = [vd[k] if k < len(vd) else 0.0 for k in range(4)]
            pr(f"  {j:3d}  {model_name:>8s}  {ev:8.4f}  {var_v:8.4f}  "
               f"{pv[0]:8.5f}  {pv[1]:8.5f}  {pv[2]:8.5f}  {pv[3]:8.5f}")

    # ── PART 2: Bulk stationary distribution ──────────────────────
    pr(f"\n{'═' * 80}")
    pr("  PART 2: BULK CARRY STATIONARY DISTRIBUTION")
    pr(f"{'═' * 80}\n")

    for n_cross in [10, 20, 30, 40, 50]:
        pi = bulk_stationary(n_cross, c_max)
        mean_c = np.dot(np.arange(c_max + 1), pi)
        var_c = np.dot(np.arange(c_max + 1)**2, pi) - mean_c**2
        pr(f"  n_cross={n_cross:3d}:  E[c]={mean_c:.4f}  Var[c]={var_c:.4f}  "
           f"pi[0..4]=[{pi[0]:.4f}, {pi[1]:.4f}, {pi[2]:.4f}, {pi[3]:.4f}, {pi[4]:.4f}]")

    # ── PART 3: Markov c₁(K) — correct model ─────────────────────
    pr(f"\n{'═' * 80}")
    pr("  PART 3: MARKOV APPROXIMATION c₁(K) — CORRECT CONVOLUTION MODEL")
    pr(f"{'═' * 80}\n")

    pr(f"  {'K':>3s}  {'c₁_markov':>12s}  {'c₁_exact':>12s}  {'Δ(markov)':>12s}  "
       f"{'Δ(exact)':>12s}  {'ratio':>8s}  {'P_even':>8s}")
    pr(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")

    results_correct = {}
    for K in range(1, K_max + 1):
        r = compute_c1_markov(K, c_max, model='correct')
        results_correct[K] = r
        delta_m = r['c1'] - LN2_OVER_4
        exact = EXACT_C1.get(K, None)
        exact_str = f"{exact:12.5f}" if exact is not None else f"{'—':>12s}"
        delta_e_str = f"{exact - LN2_OVER_4:+12.5f}" if exact is not None else f"{'—':>12s}"
        # Convergence ratio
        if K >= 2:
            prev_delta = results_correct[K - 1]['c1'] - LN2_OVER_4
            ratio = delta_m / prev_delta if abs(prev_delta) > 1e-15 else float('nan')
            ratio_str = f"{ratio:8.5f}"
        else:
            ratio_str = f"{'—':>8s}"
        pr(f"  {K:3d}  {r['c1']:12.8f}  {exact_str}  {delta_m:+12.8f}  "
           f"{delta_e_str}  {ratio_str}  {r['P_even']:8.5f}")

    # ── PART 4: Convergence analysis ──────────────────────────────
    pr(f"\n{'═' * 80}")
    pr("  PART 4: CONVERGENCE ANALYSIS (Markov model)")
    pr(f"{'═' * 80}\n")

    deltas = [(K, results_correct[K]['c1'] - LN2_OVER_4) for K in range(5, K_max + 1)]
    ratios = []
    for i in range(1, len(deltas)):
        K, d = deltas[i]
        _, d_prev = deltas[i - 1]
        if abs(d_prev) > 1e-15:
            ratios.append((K, d / d_prev))

    pr(f"  Convergence ratios Δ(K)/Δ(K-1) for K >= 6:")
    for K, ratio in ratios:
        pr(f"    K={K:3d}: ratio = {ratio:.8f}")

    if len(ratios) >= 5:
        recent = [r for _, r in ratios[-10:]]
        rho_mean = np.mean(recent)
        rho_std = np.std(recent)
        pr(f"\n  Average ρ (last {len(recent)} ratios): {rho_mean:.8f} ± {rho_std:.8f}")

    # Extrapolation
    c1_vals = [(K, results_correct[K]['c1']) for K in range(8, K_max + 1)]
    if len(c1_vals) >= 5:
        from scipy.optimize import curve_fit

        Ks = np.array([K for K, _ in c1_vals], dtype=float)
        c1s = np.array([c for _, c in c1_vals])

        def geo_model(K, c_inf, C_coeff, rho):
            return c_inf + C_coeff * rho**K

        try:
            popt, pcov = curve_fit(geo_model, Ks, c1s,
                                   p0=[LN2_OVER_4, 1.0, 0.5],
                                   bounds=([0.0, -100, 0.1], [1.0, 100, 0.99]),
                                   maxfev=50000)
            perr = np.sqrt(np.diag(pcov))
            c_inf, C_fit, rho_fit = popt
            pr(f"\n  Geometric fit  c₁(K) = c_inf + C · ρ^K  (K >= 8):")
            pr(f"    c₁(∞)  = {c_inf:.12f} ± {perr[0]:.2e}")
            pr(f"    C      = {C_fit:.8f}")
            pr(f"    ρ      = {rho_fit:.8f} ± {perr[2]:.2e}")
            pr(f"    ln(2)/4= {LN2_OVER_4:.12f}")
            pr(f"    Δ      = {c_inf - LN2_OVER_4:+.2e}")
        except Exception as e:
            pr(f"  Geometric fit failed: {e}")

        # Richardson extrapolation (last 3 points)
        K1, c1_1 = c1_vals[-3]
        K2, c1_2 = c1_vals[-2]
        K3, c1_3 = c1_vals[-1]
        d1 = c1_2 - c1_1
        d2 = c1_3 - c1_2
        if abs(d1) > 1e-15:
            rho_rich = d2 / d1
            c1_rich = c1_3 + d2 * rho_rich / (1 - rho_rich)
            pr(f"\n  Richardson extrapolation (K={K1},{K2},{K3}):")
            pr(f"    ρ_Rich  = {rho_rich:.8f}")
            pr(f"    c₁(∞)  = {c1_rich:.12f}")
            pr(f"    Δ       = {c1_rich - LN2_OVER_4:+.2e}")

    # ── PART 5: Comparison with exact values ──────────────────────
    pr(f"\n{'═' * 80}")
    pr("  PART 5: MARKOV vs EXACT ENUMERATION")
    pr(f"{'═' * 80}\n")

    pr(f"  {'K':>3s}  {'c₁_exact':>12s}  {'c₁_markov':>12s}  "
       f"{'Δ_exact':>12s}  {'Δ_markov':>12s}  {'markov/exact':>14s}")
    pr(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*14}")

    for K in sorted(EXACT_C1.keys()):
        if K > K_max:
            break
        exact = EXACT_C1[K]
        markov = results_correct[K]['c1']
        d_exact = exact - LN2_OVER_4
        d_markov = markov - LN2_OVER_4
        ratio = d_markov / d_exact if abs(d_exact) > 1e-8 else float('nan')
        pr(f"  {K:3d}  {exact:12.5f}  {markov:12.8f}  "
           f"{d_exact:+12.5f}  {d_markov:+12.8f}  {ratio:14.6f}")

    # ── PART 6: Naive Binomial model ──────────────────────────────
    pr(f"\n{'═' * 80}")
    pr("  PART 6: NAIVE Binom(j+1, 1/4) MODEL COMPARISON")
    pr(f"{'═' * 80}\n")

    pr(f"  {'K':>3s}  {'c₁_correct':>12s}  {'c₁_naive':>12s}  "
       f"{'Δ_correct':>12s}  {'Δ_naive':>12s}")
    pr(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")

    results_naive = {}
    for K in range(1, min(K_max + 1, 51)):
        r_n = compute_c1_markov(K, c_max, model='naive')
        results_naive[K] = r_n
        r_c = results_correct[K]
        pr(f"  {K:3d}  {r_c['c1']:12.8f}  {r_n['c1']:12.8f}  "
           f"{r_c['c1'] - LN2_OVER_4:+12.8f}  {r_n['c1'] - LN2_OVER_4:+12.8f}")

    # Naive model limit
    naive_vals = [(K, results_naive[K]['c1']) for K in range(8, min(K_max + 1, 51))]
    if len(naive_vals) >= 5:
        Ks_n = np.array([K for K, _ in naive_vals], dtype=float)
        c1s_n = np.array([c for _, c in naive_vals])
        try:
            popt_n, _ = curve_fit(geo_model, Ks_n, c1s_n,
                                  p0=[LN2_OVER_4, 1.0, 0.5],
                                  bounds=([0.0, -100, 0.1], [1.0, 100, 0.99]),
                                  maxfev=50000)
            pr(f"\n  Naive model extrapolation:")
            pr(f"    c₁(∞) = {popt_n[0]:.12f}  (ln(2)/4 = {LN2_OVER_4:.12f})")
            pr(f"    Δ     = {popt_n[0] - LN2_OVER_4:+.2e}")
            pr(f"    ρ     = {popt_n[2]:.8f}")
        except Exception as e:
            pr(f"  Naive extrapolation failed: {e}")

    # ── PART 7: Eigenvalue analysis ───────────────────────────────
    pr(f"\n{'═' * 80}")
    pr("  PART 7: EIGENVALUE ANALYSIS")
    pr(f"{'═' * 80}\n")

    pr("  Individual transfer matrix eigenvalues (|λ_0| ≥ |λ_1| ≥ ...):")
    pr(f"  {'j':>3s}  {'|λ_0|':>10s}  {'|λ_1|':>10s}  {'|λ_2|':>10s}  "
       f"{'|λ_3|':>10s}  {'|λ_1|/|λ_0|':>12s}")
    pr(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")

    for j in range(1, min(21, K_max + 1)):
        vd = v_dist_correct(j, min(2 * c_max, 80))
        T = build_transfer(vd, c_max)
        ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
        ratio = ev[1] / ev[0] if ev[0] > 1e-15 else float('nan')
        pr(f"  {j:3d}  {ev[0]:10.6f}  {ev[1]:10.6f}  {ev[2]:10.6f}  "
           f"{ev[3]:10.6f}  {ratio:12.8f}")

    pr(f"\n  Diaconis-Fulman prediction: |λ_k| = 1/2^k → "
       f"|λ_1| = 0.5, |λ_2| = 0.25, |λ_3| = 0.125")

    # Product matrix eigenvalues
    pr(f"\n  Product matrix M_K = T_1 · T_2 · ... · T_K:")
    pr(f"  {'K':>3s}  {'|λ_0|':>12s}  {'|λ_1|':>12s}  {'|λ_2|':>12s}  "
       f"{'|λ_1/λ_0|':>12s}  {'|λ_1/λ_0|^(1/K)':>15s}")
    pr(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*15}")

    for K in [2, 5, 8, 10, 12, 15, 20, 25, 30]:
        if K > K_max:
            break
        r = results_correct[K]
        ev_prod = eigenvalue_analysis(r['transfers'], K, c_max)
        ratio = ev_prod[1] / ev_prod[0] if ev_prod[0] > 1e-15 else 0.0
        per_step = ratio**(1.0/K) if ratio > 0 and K > 0 else 0.0
        pr(f"  {K:3d}  {ev_prod[0]:12.6e}  {ev_prod[1]:12.6e}  "
           f"{ev_prod[2]:12.6e}  {ratio:12.6e}  {per_step:15.8f}")

    # ── PART 8: Closed-form analysis ──────────────────────────────
    pr(f"\n{'═' * 80}")
    pr("  PART 8: ANALYTICAL STRUCTURE OF THE TRANSFER OPERATOR")
    pr(f"{'═' * 80}\n")

    pr("  Transfer matrix T_1 (at offset 1, conv[1] ~ Binom(2, 1/2)):")
    vd1 = v_dist_correct(1, 40)
    T1 = build_transfer(vd1, min(c_max, 6))
    n_show = min(c_max + 1, 7)
    pr(f"    {'':>8s}", end='')
    for c in range(n_show):
        pr(f"  c={c:d}   ", end='')
    pr()
    for c_new in range(n_show):
        pr(f"  c'={c_new:d}  ", end='')
        for c in range(n_show):
            pr(f"  {T1[c_new, c]:6.4f}", end='')
        pr()

    pr(f"\n  Key entry: T_1[0,0] = {T1[0, 0]:.6f}")
    pr(f"  This is P(carry=0 | carry=0, v~Binom(2,1/2))")
    pr(f"  = P(v <= 1) = P(0) + P(1) = 1/4 + 1/2 = 3/4 = {0.75:.4f}")

    pr(f"\n  Transfer matrix T_2 (at offset 2, conv ~ Binom(2,1/2)+Binom(1,1/4)):")
    vd2 = v_dist_correct(2, 40)
    T2 = build_transfer(vd2, min(c_max, 6))
    pr(f"    {'':>8s}", end='')
    for c in range(n_show):
        pr(f"  c={c:d}   ", end='')
    pr()
    for c_new in range(n_show):
        pr(f"  c'={c_new:d}  ", end='')
        for c in range(n_show):
            pr(f"  {T2[c_new, c]:6.4f}", end='')
        pr()

    # ── PART 9: Stability check with larger c_max ──────────────────
    pr(f"\n{'═' * 80}")
    pr("  PART 9: STABILITY CHECK — VARYING c_max")
    pr(f"{'═' * 80}\n")

    pr(f"  {'c_max':>5s}  {'c₁_markov(K=40)':>18s}  {'Δ from c_max=20':>18s}")
    pr(f"  {'─'*5}  {'─'*18}  {'─'*18}")
    c1_ref = results_correct[min(40, K_max)]['c1']
    for cm in [10, 15, 20, 25, 30]:
        r_test = compute_c1_markov(40, cm, model='correct')
        diff = r_test['c1'] - c1_ref
        pr(f"  {cm:5d}  {r_test['c1']:18.12f}  {diff:+18.2e}")

    # ── PART 10: HIGH-PRECISION Markov limit via mpmath ─────────
    pr(f"\n{'═' * 80}")
    pr("  PART 10: HIGH-PRECISION MARKOV LIMIT & PSLQ")
    pr(f"{'═' * 80}\n")

    c1_markov_limit = results_correct[K_max]['c1']
    pr(f"  Double-precision limit ≈ {c1_markov_limit:.15f}")

    try:
        import mpmath
        from fractions import Fraction
        mpmath.mp.dps = 60

        def compute_markov_limit_mpmath(K_hi=40, c_max_hi=15):
            """Recompute Markov c₁ limit using mpmath for full precision."""
            mp = mpmath
            n = c_max_hi + 1

            def v_dist_mp(j):
                dist = [mp.mpf(0)] * (n * 2)
                if j == 0:
                    dist[1] = mp.mpf(1)
                    return dist
                if j == 1:
                    for v in range(min(3, len(dist))):
                        dist[v] = mp.binomial(2, v) * mp.power(mp.mpf(1)/2, 2)
                    return dist
                nc = j - 1
                for s1 in range(3):
                    p1 = mp.binomial(2, s1) * mp.power(mp.mpf(1)/2, 2)
                    for s2 in range(min(nc + 1, len(dist))):
                        p2 = mp.binomial(nc, s2) * mp.power(mp.mpf(1)/4, s2) * mp.power(mp.mpf(3)/4, nc - s2)
                        v = s1 + s2
                        if v < len(dist):
                            dist[v] += p1 * p2
                return dist

            def build_T_mp(vd):
                T = mp.matrix(n, n)
                for c in range(n):
                    for v in range(len(vd)):
                        if vd[v] < mp.mpf(10)**(-50):
                            continue
                        cn = min((v + c) // 2, c_max_hi)
                        T[cn, c] += vd[v]
                return T

            def bulk_stat_mp(nc):
                vd = [mp.mpf(0)] * (nc + 1)
                for v in range(nc + 1):
                    vd[v] = mp.binomial(nc, v) * mp.power(mp.mpf(1)/4, v) * mp.power(mp.mpf(3)/4, nc - v)
                T = build_T_mp(vd)
                pi = mp.matrix(n, 1)
                for i in range(n):
                    pi[i] = mp.mpf(1) / n
                for _ in range(500000):
                    pi_new = T * pi
                    s = sum(pi_new[i] for i in range(n))
                    for i in range(n):
                        pi_new[i] /= s
                    diff = max(abs(pi_new[i] - pi[i]) for i in range(n))
                    pi = pi_new
                    if diff < mp.mpf(10)**(-55):
                        break
                return pi

            n_bulk = max(K_hi + 10, 30)
            pi_entry = bulk_stat_mp(n_bulk)

            transfers_mp = {}
            for j in range(1, K_hi + 1):
                vd = v_dist_mp(j)
                transfers_mp[j] = build_T_mp(vd)

            pi_at = [None] * (K_hi + 1)
            pi_at[K_hi] = pi_entry
            for j in range(K_hi - 1, -1, -1):
                pi_at[j] = transfers_mp[j + 1] * pi_at[j + 1]

            E_even = sum(mp.mpf(c) * pi_at[0][c] for c in range(n))

            cum_p = [mp.mpf(0)] * (K_hi + 1)
            cum_p[0] = mp.mpf(1)
            for k in range(1, K_hi + 1):
                cum_p[k] = cum_p[k - 1] * transfers_mp[k][0, 0]

            D_odd_sum = mp.mpf(0)
            for J in range(1, K_hi):
                beta_J = [transfers_mp[J][0, c] * cum_p[J - 1] for c in range(n)]
                w = [mp.mpf(c) * pi_at[J + 1][c] for c in range(n)]
                inner = transfers_mp[J + 1] * mp.matrix(w)
                for cJ in range(1, n):
                    D_odd_sum += beta_J[cJ] * inner[cJ]

            beta_K = [transfers_mp[K_hi][0, c] * cum_p[K_hi - 1] for c in range(n)]
            for c in range(n):
                D_odd_sum += beta_K[c] * mp.mpf(c) * pi_at[K_hi][c]

            return E_even + D_odd_sum - 1

        pr("  Computing with mpmath (60 decimal digits, K=40, c_max=20)...")
        t_mp = time.time()
        c1_hp = compute_markov_limit_mpmath(40, 20)
        pr(f"  Computed in {time.time() - t_mp:.1f}s")
        pr(f"  c₁_markov (high prec) = {mpmath.nstr(c1_hp, 40)}")
        pr(f"  ln(2)/4               = {mpmath.nstr(mpmath.log(2)/4, 40)}")
        pr(f"  Δ                     = {mpmath.nstr(c1_hp - mpmath.log(2)/4, 15)}")

        ln2 = mpmath.log(2)
        ln3 = mpmath.log(3)

        pr(f"\n  PSLQ on [c₁_markov, ln2, ln3, 1] with 50-digit precision:")
        rel = mpmath.pslq([c1_hp, ln2, ln3, mpmath.mpf(1)], maxcoeff=1000)
        if rel:
            pr(f"    Found: {rel[0]}·c₁ + {rel[1]}·ln2 + {rel[2]}·ln3 + {rel[3]} = 0")
            if rel[0] != 0:
                c_ln2 = Fraction(-rel[1], rel[0])
                c_ln3 = Fraction(-rel[2], rel[0])
                c_1 = Fraction(-rel[3], rel[0])
                pr(f"    → c₁_markov = ({c_ln2})·ln2 + ({c_ln3})·ln3 + ({c_1})")
                val = c_ln2 * ln2 + c_ln3 * ln3 + c_1
                pr(f"    Verification: {mpmath.nstr(val, 30)}")
                pr(f"    Residual:     {mpmath.nstr(abs(val - c1_hp), 5)}")
        else:
            pr(f"    No relation found (not a simple ln(2),ln(3) combination)")

        pr(f"\n  PSLQ on [c₁_markov, ln2, ln²2, 1]:")
        rel2 = mpmath.pslq([c1_hp, ln2, ln2**2, mpmath.mpf(1)], maxcoeff=1000)
        if rel2:
            pr(f"    Found: {rel2}")
            if rel2[0] != 0:
                c_ln2 = Fraction(-rel2[1], rel2[0])
                c_ln2sq = Fraction(-rel2[2], rel2[0])
                c_1 = Fraction(-rel2[3], rel2[0])
                pr(f"    → c₁_markov = ({c_ln2})·ln2 + ({c_ln2sq})·ln²2 + ({c_1})")
                val = c_ln2 * ln2 + c_ln2sq * ln2**2 + c_1
                pr(f"    Residual: {mpmath.nstr(abs(val - c1_hp), 5)}")
        else:
            pr(f"    No relation found")

        pr(f"\n  PSLQ on [c₁_markov, 1] (rational?):")
        rel3 = mpmath.pslq([c1_hp, mpmath.mpf(1)], maxcoeff=100000)
        if rel3:
            frac = Fraction(-rel3[1], rel3[0])
            pr(f"    Candidate: {frac} = {float(frac):.15f}")
            pr(f"    Residual: {mpmath.nstr(abs(mpmath.mpf(frac.numerator)/frac.denominator - c1_hp), 5)}")
        else:
            pr(f"    Not a simple rational")

        # Non-Markovian correction
        correction = mpmath.log(2)/4 - c1_hp
        pr(f"\n  Non-Markovian correction δ_NM = {mpmath.nstr(correction, 30)}")
        pr(f"\n  PSLQ on [δ_NM, ln2, ln3, ln²2, 1]:")
        rel_nm = mpmath.pslq([correction, ln2, ln3, ln2**2, mpmath.mpf(1)], maxcoeff=1000)
        if rel_nm:
            pr(f"    Found: {rel_nm}")
            if rel_nm[0] != 0:
                c_ln2 = Fraction(-rel_nm[1], rel_nm[0])
                c_ln3 = Fraction(-rel_nm[2], rel_nm[0])
                c_ln2sq = Fraction(-rel_nm[3], rel_nm[0])
                c_1 = Fraction(-rel_nm[4], rel_nm[0])
                pr(f"    → δ_NM = ({c_ln2})·ln2 + ({c_ln3})·ln3 + ({c_ln2sq})·ln²2 + ({c_1})")
        else:
            pr(f"    No relation found")

        pr(f"\n  PSLQ on [δ_NM, ln2, ln3, π, 1]:")
        rel_nm2 = mpmath.pslq([correction, ln2, ln3, mpmath.pi, mpmath.mpf(1)], maxcoeff=1000)
        if rel_nm2:
            pr(f"    Found: {rel_nm2}")
        else:
            pr(f"    No relation found")

    except ImportError:
        pr("  mpmath not available — skipping high-precision PSLQ")

    # ── PART 11: VERDICT ──────────────────────────────────────────
    pr(f"\n{'═' * 80}")
    pr("  VERDICT")
    pr(f"{'═' * 80}\n")

    c1_markov_last = results_correct[K_max]['c1']
    delta_markov = c1_markov_last - LN2_OVER_4

    pr(f"  Markov transfer operator c₁(K={K_max}):")
    pr(f"    c₁_markov  = {c1_markov_last:.12f}")
    pr(f"    ln(2)/4    = {LN2_OVER_4:.12f}")
    pr(f"    Δ          = {delta_markov:+.2e}")
    pr()

    # Check if Markov and exact agree
    common_Ks = sorted(k for k in EXACT_C1 if k <= K_max)
    if common_Ks:
        max_ratio = 0
        for K in common_Ks:
            d_e = EXACT_C1[K] - LN2_OVER_4
            d_m = results_correct[K]['c1'] - LN2_OVER_4
            if abs(d_e) > 1e-8:
                max_ratio = max(max_ratio, abs(d_m / d_e))

    if abs(delta_markov) < 1e-6:
        pr("  CONCLUSION: Markov approximation appears to converge to ln(2)/4.")
        pr("  The non-Markovian corrections may cancel in the limit!")
    elif abs(delta_markov) < 0.01:
        pr("  CONCLUSION: Markov limit is CLOSE to ln(2)/4 but may differ.")
        pr(f"  Residual = {delta_markov:+.6f}")
    else:
        pr("  CONCLUSION: Markov limit appears to DIFFER from ln(2)/4.")
        pr(f"  Markov limit ≈ {c1_markov_last:.8f}, target = {LN2_OVER_4:.8f}")

    pr(f"\n  Markov vs Exact discrepancy (at shared K values):")
    for K in common_Ks:
        diff = results_correct[K]['c1'] - EXACT_C1[K]
        pr(f"    K={K:2d}: |c₁_markov - c₁_exact| = {abs(diff):.6f}")

    # Does ρ match?
    if len(ratios) >= 5:
        recent = [r for _, r in ratios[-10:]]
        rho_m = np.mean(recent)
        pr(f"\n  Convergence rate ρ:")
        pr(f"    Markov model: ρ = {rho_m:.6f}")
        pr(f"    Exact enum:   ρ ≈ 0.64")
        pr(f"    Ratio:        {rho_m / 0.64:.6f}")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")
    pr("=" * 80)


if __name__ == '__main__':
    main()
