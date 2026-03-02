#!/usr/bin/env python3
"""
B25: Asymptotic Limit c₁(D → ∞)

From prior experiments: c₁ = -⟨Tr(M)⟩ - 1, which depends on D (bit size).
This experiment measures c₁(D) at many bit sizes to find:
  1. The asymptotic limit c₁(∞)
  2. The convergence rate: c₁(D) = c₁(∞) + α/D^β
  3. Whether c₁(∞) is a recognizable constant

The trace formula gives: c₁ = ⟨c_{D-2}/c_{D-1}⟩ - 1
where c_{D-1}, c_{D-2} are the top carries. Since c_{D-1} ∈ {2,3}
(base 2), c₁ is determined by the distribution of c_{D-2}.

Strategy:
  A) Direct measurement of ⟨Tr(M)⟩ at many bit sizes (8 to 48)
  B) Separate c_{D-1}=2 and c_{D-1}=3 cases
  C) Fit c₁(D) → c₁(∞) with power-law correction
  D) High-precision at large D using regression method
  E) Compare c₁(∞) against known constants
"""

import sys, os, time, random, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, carry_poly_int, quotient_poly_int, to_digits

random.seed(42)
np.random.seed(42)


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def full_carry_sequence(p, q, base=2):
    N = p * q
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    fd = to_digits(N, base)
    D = len(fd)
    conv = [0] * (D + 2)
    for i, a in enumerate(gd):
        for j, b_ in enumerate(hd):
            if i + j < len(conv):
                conv[i + j] += a * b_
    carries = [0] * (D + 2)
    for k in range(D):
        total = conv[k] + carries[k]
        carries[k + 1] = total // base
    return carries[:D + 1], conv[:D + 1], fd


def compute_trace(p, q, base=2):
    """Compute Tr(M) of carry companion matrix."""
    C = carry_poly_int(p, q, base)
    Q = quotient_poly_int(C, base)
    if len(Q) < 3:
        return None
    lead = float(Q[-1])
    if abs(lead) < 1e-30:
        return None
    n = len(Q) - 1
    return -float(Q[n - 1]) / lead


def measure_R(l, sigma, semiprimes, base=2):
    """Measure R(l,σ) from pre-generated semiprimes."""
    target = abs(1.0 / (1.0 - l ** (-sigma)))
    det_vals = []
    for p, q in semiprimes:
        C = carry_poly_int(p, q, base)
        Q = quotient_poly_int(C, base)
        if len(Q) < 3:
            continue
        lead = float(Q[-1])
        if abs(lead) < 1e-30:
            continue
        n = len(Q) - 1
        M = np.zeros((n, n), dtype=complex)
        for i in range(n - 1):
            M[i + 1, i] = 1.0
        for i in range(n):
            M[i, n - 1] = -float(Q[i]) / lead
        ls = l ** (-sigma)
        det_val = abs(np.linalg.det(np.eye(n, dtype=complex) - M * ls))
        if det_val > 0 and not math.isnan(det_val) and not math.isinf(det_val):
            det_vals.append(det_val)
    if len(det_vals) < 100:
        return None
    return np.mean(det_vals) / target


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("P3-02: ASYMPTOTIC LIMIT c₁(D → ∞)")
    pr("=" * 72)

    # ═══════════════════════════════════════════════════════════════
    # PART A: DIRECT ⟨Tr(M)⟩ AT MANY BIT SIZES
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: ⟨Tr(M)⟩ AND c₁ = -⟨Tr(M)⟩ - 1 VS BIT SIZE")
    pr(f"{'═' * 72}\n")

    bit_sizes = [8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40]
    n_samp = 6000

    c1_from_trace = []
    D_values = []

    pr(f"  {'bits':>5s}  {'D≈':>5s}  {'⟨Tr(M)⟩':>10s}  {'±':>8s}  {'c₁':>10s}  {'±':>8s}")
    pr(f"  {'─'*5}  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*8}")

    for bits in bit_sizes:
        traces = []
        Ds = []
        for _ in range(n_samp):
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            tr = compute_trace(p, q)
            if tr is not None and not math.isnan(tr):
                traces.append(tr)
                N = p * q
                Ds.append(N.bit_length())

        if not traces:
            continue

        tr_arr = np.array(traces)
        D_mean = np.mean(Ds)
        tr_mean = tr_arr.mean()
        tr_se = tr_arr.std() / math.sqrt(len(traces))
        c1 = -tr_mean - 1
        c1_se = tr_se

        c1_from_trace.append(c1)
        D_values.append(D_mean)

        pr(f"  {bits:5d}  {D_mean:5.0f}  {tr_mean:10.6f}  {tr_se:8.6f}  {c1:10.6f}  {c1_se:8.6f}")

    # ═══════════════════════════════════════════════════════════════
    # PART B: SEPARATE c_{D-1}=2 AND c_{D-1}=3 CASES
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: CONDITIONAL ⟨Tr(M)⟩ BY c_{D-1} VALUE")
    pr(f"{'═' * 72}\n")

    pr(f"  Recall: Tr(M) = -c_{{D-2}}/c_{{D-1}}")
    pr(f"  For c_{{D-1}}=2: Tr(M) = -c_{{D-2}}/2")
    pr(f"  For c_{{D-1}}=3: Tr(M) = -c_{{D-2}}/3\n")

    for bits in [12, 16, 20, 24, 32]:
        tr_cond2 = []
        tr_cond3 = []
        cd2_cond2 = []
        cd2_cond3 = []

        for _ in range(n_samp):
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            carries, conv, fd = full_carry_sequence(p, q, base=2)
            D = len(fd)
            if D < 4:
                continue

            cd1 = carries[D - 1]
            cd2 = carries[D - 2]
            tr = compute_trace(p, q)
            if tr is None:
                continue

            if cd1 == 2:
                tr_cond2.append(tr)
                cd2_cond2.append(cd2)
            elif cd1 == 3:
                tr_cond3.append(tr)
                cd2_cond3.append(cd2)

        pr(f"  bits={bits}:")
        if tr_cond2:
            arr = np.array(tr_cond2)
            cd2_arr = np.array(cd2_cond2, dtype=float)
            pr(f"    c_{{D-1}}=2 (n={len(tr_cond2)}): ⟨Tr⟩={arr.mean():.6f}, "
               f"⟨c_{{D-2}}⟩={cd2_arr.mean():.4f}, "
               f"c₁={-arr.mean()-1:.6f}")
        if tr_cond3:
            arr = np.array(tr_cond3)
            cd2_arr = np.array(cd2_cond3, dtype=float)
            pr(f"    c_{{D-1}}=3 (n={len(tr_cond3)}): ⟨Tr⟩={arr.mean():.6f}, "
               f"⟨c_{{D-2}}⟩={cd2_arr.mean():.4f}, "
               f"c₁={-arr.mean()-1:.6f}")
        frac2 = len(tr_cond2) / (len(tr_cond2) + len(tr_cond3)) if (tr_cond2 or tr_cond3) else 0
        pr(f"    fraction c_{{D-1}}=2: {frac2:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PART C: FIT c₁(D) → c₁(∞) WITH POWER-LAW CORRECTION
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: FIT c₁(D) = c₁(∞) + α/D^β")
    pr(f"{'═' * 72}\n")

    if len(c1_from_trace) >= 5 and len(D_values) >= 5:
        D_arr = np.array(D_values)
        c1_arr = np.array(c1_from_trace)

        pr(f"  Data points (D, c₁):")
        for d, c in zip(D_arr, c1_arr):
            pr(f"    D={d:6.1f}  c₁={c:.6f}")

        from scipy.optimize import curve_fit

        def model(D, c1_inf, alpha, beta):
            return c1_inf + alpha / D ** beta

        try:
            popt, pcov = curve_fit(model, D_arr, c1_arr,
                                   p0=[0.15, 1.0, 0.5],
                                   bounds=([0, -10, 0.01], [0.5, 10, 3]),
                                   maxfev=10000)
            c1_inf, alpha, beta = popt
            perr = np.sqrt(np.diag(pcov))
            residuals = c1_arr - model(D_arr, *popt)

            pr(f"\n  Power-law fit: c₁(D) = {c1_inf:.6f} + {alpha:.4f}/D^{beta:.4f}")
            pr(f"    c₁(∞) = {c1_inf:.6f} ± {perr[0]:.6f}")
            pr(f"    α = {alpha:.6f} ± {perr[1]:.6f}")
            pr(f"    β = {beta:.6f} ± {perr[2]:.6f}")
            pr(f"    Max residual: {np.max(np.abs(residuals)):.6f}")
            pr(f"    R² = {1 - np.var(residuals)/np.var(c1_arr):.6f}")

            c1_inf_val = c1_inf
        except Exception as e:
            pr(f"  Fit failed: {e}")
            c1_inf_val = np.mean(c1_arr)

        try:
            def linear_model(D, c1_inf, a):
                return c1_inf + a / D

            popt2, pcov2 = curve_fit(linear_model, D_arr, c1_arr, p0=[0.15, 1.0])
            perr2 = np.sqrt(np.diag(pcov2))
            resid2 = c1_arr - linear_model(D_arr, *popt2)
            pr(f"\n  Linear fit: c₁(D) = {popt2[0]:.6f} + {popt2[1]:.4f}/D")
            pr(f"    c₁(∞) = {popt2[0]:.6f} ± {perr2[0]:.6f}")
            pr(f"    R² = {1 - np.var(resid2)/np.var(c1_arr):.6f}")
        except Exception as e:
            pr(f"  Linear fit failed: {e}")
    else:
        c1_inf_val = 0.16

    # ═══════════════════════════════════════════════════════════════
    # PART D: REGRESSION METHOD AT LARGE D
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: REGRESSION c₁ VIA ln(R) AT SELECTED BIT SIZES")
    pr(f"{'═' * 72}\n")

    sigma_vals = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    test_l = 47

    for bits in [10, 14, 18, 22, 26, 30, 34, 38, 42]:
        semiprimes = []
        for _ in range(8000):
            p = random_prime(bits)
            q = random_prime(bits)
            if p != q:
                semiprimes.append((p, q))

        R_data = []
        for sigma in sigma_vals:
            R = measure_R(test_l, sigma, semiprimes)
            if R is not None and R > 0 and not math.isnan(R):
                R_data.append((sigma, R))

        if len(R_data) < 4:
            pr(f"  bits={bits}: insufficient data")
            continue

        sigmas = np.array([v[0] for v in R_data])
        lnR = np.array([math.log(v[1]) for v in R_data])
        A = np.column_stack([test_l ** (-k * sigmas) for k in range(1, 4)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, lnR, rcond=None)
            pr(f"  bits={bits:3d} (D≈{2*bits}): c₁ = {coeffs[0]:.6f}, "
               f"c₂ = {coeffs[1]:.6f}")
        except:
            pr(f"  bits={bits}: fit failed")

    # ═══════════════════════════════════════════════════════════════
    # PART E: IDENTIFY c₁(∞)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: IDENTIFY c₁(∞)")
    pr(f"{'═' * 72}\n")

    candidates = {
        '1/6': 1.0/6,
        '(1-ln2)/2': (1 - math.log(2)) / 2,
        '3/(2π²)': 3.0 / (2 * math.pi**2),
        'π²/60': math.pi**2 / 60,
        '1/5': 0.2,
        'γ/4': 0.5772156649 / 4,
        'ln(2)/4': math.log(2) / 4,
        '(2-√3)/2': (2 - math.sqrt(3)) / 2,
        '3·ln2/(4π)': 3*math.log(2)/(4*math.pi),
        '1/(2π)': 1.0/(2*math.pi),
        '(e-2)/4': (math.e - 2) / 4,
    }

    pr(f"  c₁(∞) estimate = {c1_inf_val:.6f}\n")
    scored = []
    for name, val in candidates.items():
        dist = abs(c1_inf_val - val)
        scored.append((dist, name, val))
    scored.sort()
    for i, (dist, name, val) in enumerate(scored[:8]):
        pr(f"    {i+1}. {name:<15s} = {val:.6f}, |Δ| = {dist:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS — ASYMPTOTIC TRACE ANOMALY")
    pr(f"{'═' * 72}")
    pr(f"""
  The trace anomaly c₁ = -⟨Tr(M)⟩ - 1 has two components:

  1. When c_{{D-1}} = 2 (D = 2d-1, product has odd # of bits):
     Tr(M) = -c_{{D-2}}/2
     c₁ = c_{{D-2}}/2 - 1

  2. When c_{{D-1}} = 3 (D = 2d, product has even # of bits):
     Tr(M) = -c_{{D-2}}/3
     c₁ = c_{{D-2}}/3 - 1

  As D → ∞, the distribution of c_{{D-2}} converges to a
  steady-state determined by the carry Markov chain boundary.
  The weighted average over both cases gives c₁(∞).

  The key question: is c₁(∞) a recognizable constant, or
  is it a non-elementary number arising from the carry chain?
""")

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
