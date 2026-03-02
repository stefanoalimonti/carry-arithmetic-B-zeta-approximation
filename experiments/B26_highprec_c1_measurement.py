#!/usr/bin/env python3
"""
B26: High-Precision Measurement of c₁(∞)

c₁ = ⟨c_{top-1}⟩ - 1 where c_{top-1} = -Q[-2] and c_top = -Q[-1] = 1.

Strategy: compute c_{top-1} DIRECTLY from Q (no eigenvalue computation),
enabling very high statistics (50,000+ samples per bit size).

Measure at bit sizes 20-60 and extrapolate to D → ∞.
Compare c₁(∞) to ln(2)/4 = 0.173286795... at 4+ decimal precision.

Also: analytical decomposition by D parity and top-digit structure
to see if c₁(∞) has a closed-form expression.
"""

import sys, os, time, random, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, carry_poly_int, quotient_poly_int, to_digits

random.seed(42)
np.random.seed(42)

LN2_OVER_4 = math.log(2) / 4


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def get_ctop1(p, q, base=2):
    """Fast extraction of c_{top-1} from Q. Returns (c_top, c_top1, D)."""
    N = p * q
    C = carry_poly_int(p, q, base)
    Q = quotient_poly_int(C, base)
    if len(Q) < 3:
        return None
    c_top = -Q[-1]
    c_top1 = -Q[-2]
    D = len(to_digits(N, base))
    return c_top, c_top1, D


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("P3-02b: HIGH-PRECISION c₁(∞) — TARGET: ln(2)/4 ?")
    pr("=" * 72)
    pr(f"\n  ln(2)/4 = {LN2_OVER_4:.10f}")

    # ═══════════════════════════════════════════════════════════════
    # PART A: MASSIVE STATISTICS AT KEY BIT SIZES
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: HIGH-STATISTICS ⟨c_{top-1}⟩ AT MANY BIT SIZES")
    pr(f"{'═' * 72}\n")

    n_samp = 50000

    results = []
    pr(f"  {'bits':>5s}  {'D̄':>7s}  {'⟨c_top-1⟩':>11s}  {'±':>10s}  "
       f"{'c₁':>11s}  {'±':>10s}  {'ln2/4−c₁':>10s}")
    pr(f"  {'─'*5}  {'─'*7}  {'─'*11}  {'─'*10}  "
       f"{'─'*11}  {'─'*10}  {'─'*10}")

    for bits in [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                 32, 36, 40, 44, 48, 52, 56, 60]:
        ctop1_vals = []
        Ds = []

        for _ in range(n_samp):
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            result = get_ctop1(p, q)
            if result is None:
                continue
            ct, ct1, D = result
            if ct == 1:
                ctop1_vals.append(ct1)
                Ds.append(D)

        if not ctop1_vals:
            continue

        arr = np.array(ctop1_vals, dtype=float)
        D_mean = np.mean(Ds)
        mean = arr.mean()
        se = arr.std() / math.sqrt(len(arr))
        c1 = mean - 1
        c1_se = se
        delta = LN2_OVER_4 - c1

        results.append((bits, D_mean, c1, c1_se))
        pr(f"  {bits:5d}  {D_mean:7.1f}  {mean:11.6f}  {se:10.6f}  "
           f"{c1:11.6f}  {c1_se:10.6f}  {delta:+10.6f}")

    # ═══════════════════════════════════════════════════════════════
    # PART B: CONDITIONAL BY D PARITY
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: CONDITIONAL c₁ BY D PARITY (EVEN vs ODD)")
    pr(f"{'═' * 72}\n")

    for bits in [20, 30, 40, 52]:
        ctop1_even = []
        ctop1_odd = []

        for _ in range(n_samp):
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            result = get_ctop1(p, q)
            if result is None:
                continue
            ct, ct1, D = result
            if ct != 1:
                continue
            if D % 2 == 0:
                ctop1_even.append(ct1)
            else:
                ctop1_odd.append(ct1)

        total = len(ctop1_even) + len(ctop1_odd)
        frac_even = len(ctop1_even) / total

        arr_e = np.array(ctop1_even, dtype=float)
        arr_o = np.array(ctop1_odd, dtype=float)
        c1_e = arr_e.mean() - 1
        c1_o = arr_o.mean() - 1
        c1_total = frac_even * arr_e.mean() + (1 - frac_even) * arr_o.mean() - 1

        pr(f"  bits={bits}: frac(D even)={frac_even:.4f}")
        pr(f"    D even: ⟨c_top-1⟩={arr_e.mean():.6f}, c₁={c1_e:.6f}, n={len(ctop1_even)}")
        pr(f"    D odd:  ⟨c_top-1⟩={arr_o.mean():.6f}, c₁={c1_o:.6f}, n={len(ctop1_odd)}")
        pr(f"    Weighted c₁ = {c1_total:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # PART C: ANALYTICAL PREDICTION FOR D = 2d (EVEN)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: ANALYTICAL PREDICTION FOR D = 2d CASE")
    pr(f"{'═' * 72}")

    def prob_XY_geq(t):
        """P(XY ≥ t) for X,Y ~ Uniform[1/2,1). Valid for 1/4 ≤ t ≤ 1."""
        if t <= 1/4:
            return 1.0
        if t >= 1:
            return 0.0
        if t >= 1/2:
            val = 1 + 2*t*(math.log(t) - 1) + 1
            return 2*(1-t) + 2*t*math.log(1/t)
        y_low = max(1/2, t / 1)
        y_high = min(1, t / (1/2))
        return 0

    p_D_even = 2 * (1 - math.log(2))
    p_D_odd = 1 - p_D_even
    p_XY_3_4 = 1 + 3 * math.log(3/4)

    p_f1_given_Deven = p_XY_3_4 / p_D_even

    E_ctop1_Deven = 1 + p_f1_given_Deven
    c1_Deven = E_ctop1_Deven - 1

    contrib_Deven = p_D_even * E_ctop1_Deven

    pr(f"""
  For D = 2d: m = 2d-1, conv_{{m-1}} = 1, f_{{m-1}} = I(N ≥ 3·2^{{2d-2}}).
  c_{{top-1}} = 1 + f_{{m-1}}.

  P(D even) = 2(1 - ln2) = {p_D_even:.10f}
  P(D odd)  = 2ln2 - 1   = {p_D_odd:.10f}
  P(XY ≥ 3/4) = 1 + 3·ln(3/4) = {p_XY_3_4:.10f}
  P(f=1 | D even) = {p_f1_given_Deven:.10f}

  E[c_{{top-1}} | D even] = 1 + {p_f1_given_Deven:.10f} = {E_ctop1_Deven:.10f}
  c₁(D even) = {c1_Deven:.10f}

  Contribution to overall: p_D_even · E[c_{{top-1}} | D even] = {contrib_Deven:.10f}

  For overall c₁ = ln(2)/4:
    Need p_D_odd · E[c_{{top-1}} | D odd] = 1 + ln(2)/4 - {contrib_Deven:.6f}
    = {1 + LN2_OVER_4 - contrib_Deven:.10f}
    ⟹ E[c_{{top-1}} | D odd] = {(1 + LN2_OVER_4 - contrib_Deven)/p_D_odd:.10f}
""")

    target_Eodd = (1 + LN2_OVER_4 - contrib_Deven) / p_D_odd
    pr(f"  Target ⟨c_{{top-1}}⟩(D odd) for c₁=ln(2)/4: {target_Eodd:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # PART D: FIT WITH CORRECTED DATA
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: EXTRAPOLATION c₁(D → ∞)")
    pr(f"{'═' * 72}\n")

    if len(results) >= 8:
        Ds = np.array([r[1] for r in results])
        c1s = np.array([r[2] for r in results])
        ses = np.array([r[3] for r in results])

        large = Ds > 35
        if np.sum(large) >= 5:
            c1_large = c1s[large]
            se_large = ses[large]
            w = 1.0 / se_large**2
            c1_wmean = np.sum(w * c1_large) / np.sum(w)
            c1_wse = 1.0 / math.sqrt(np.sum(w))

            pr(f"  Weighted mean of c₁ for D > 35:")
            pr(f"    c₁ = {c1_wmean:.8f} ± {c1_wse:.8f}")
            pr(f"    ln(2)/4 = {LN2_OVER_4:.8f}")
            pr(f"    Δ = {c1_wmean - LN2_OVER_4:+.8f}")
            pr(f"    |Δ|/σ = {abs(c1_wmean - LN2_OVER_4)/c1_wse:.2f}")

        from scipy.optimize import curve_fit

        def model_1overD(D, c_inf, a):
            return c_inf + a / D

        def model_power(D, c_inf, a, b):
            return c_inf + a / D**b

        weights = 1.0 / ses**2

        try:
            popt, pcov = curve_fit(model_1overD, Ds, c1s, sigma=ses,
                                   p0=[0.17, -1])
            perr = np.sqrt(np.diag(pcov))
            pr(f"\n  1/D fit: c₁(∞) = {popt[0]:.8f} ± {perr[0]:.8f}")
            pr(f"    a = {popt[1]:.6f} ± {perr[1]:.6f}")
            pr(f"    Δ from ln(2)/4 = {popt[0] - LN2_OVER_4:+.8f}")
        except Exception as e:
            pr(f"  1/D fit failed: {e}")

        try:
            popt2, pcov2 = curve_fit(model_power, Ds, c1s, sigma=ses,
                                     p0=[0.17, -1, 1],
                                     bounds=([0, -50, 0.1], [0.3, 50, 5]))
            perr2 = np.sqrt(np.diag(pcov2))
            pr(f"\n  Power fit: c₁(∞) = {popt2[0]:.8f} ± {perr2[0]:.8f}")
            pr(f"    a = {popt2[1]:.6f} ± {perr2[1]:.6f}")
            pr(f"    β = {popt2[2]:.6f} ± {perr2[2]:.6f}")
            pr(f"    Δ from ln(2)/4 = {popt2[0] - LN2_OVER_4:+.8f}")
        except Exception as e:
            pr(f"  Power fit failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # PART E: OTHER CANDIDATES — FINE DISCRIMINATION
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: FINE DISCRIMINATION AMONG TOP CANDIDATES")
    pr(f"{'═' * 72}\n")

    candidates = {
        'ln(2)/4':          math.log(2) / 4,
        '1/6':              1.0 / 6,
        'π²/60':            math.pi**2 / 60,
        '(e-2)/4':          (math.e - 2) / 4,
        '3·ln2/(4π)':       3*math.log(2)/(4*math.pi),
        '1/(2π)':           1.0/(2*math.pi),
        '(2-√3)/2':         (2 - math.sqrt(3))/2,
        '1/2 - 1/π':        0.5 - 1/math.pi,
        'ln(2)·γ/π':        math.log(2)*0.5772156649/math.pi,
        '(π-3)/2':          (math.pi - 3)/2,
        'ln(2)/(2+ln2)':    math.log(2)/(2+math.log(2)),
        '(1-ln2)/2':        (1 - math.log(2))/2,
        '2·ln2/π-1/π':      2*math.log(2)/math.pi - 1/math.pi,
    }

    if len(results) >= 8:
        Ds_arr = np.array([r[1] for r in results])
        c1_arr = np.array([r[2] for r in results])
        se_arr = np.array([r[3] for r in results])

        large_mask = Ds_arr > 30
        if np.sum(large_mask) >= 3:
            c1_lg = c1_arr[large_mask]
            se_lg = se_arr[large_mask]
            w_lg = 1.0 / se_lg**2
            c1_best = np.sum(w_lg * c1_lg) / np.sum(w_lg)
            c1_best_se = 1.0 / math.sqrt(np.sum(w_lg))
        else:
            c1_best = np.mean(c1_arr)
            c1_best_se = 0.01

        pr(f"  Best estimate: c₁ = {c1_best:.8f} ± {c1_best_se:.8f}\n")

        scored = []
        for name, val in candidates.items():
            delta = abs(c1_best - val)
            nsigma = delta / c1_best_se if c1_best_se > 0 else 999
            scored.append((nsigma, name, val, delta))
        scored.sort()

        for i, (ns, name, val, delta) in enumerate(scored[:10]):
            marker = " ✓" if ns < 2 else ""
            pr(f"    {i+1:2d}. {name:<20s} = {val:.10f}, "
               f"|Δ| = {delta:.8f}, {ns:.1f}σ{marker}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS")
    pr(f"{'═' * 72}")
    pr(f"""
  The trace anomaly c₁ = ⟨c_{{top-1}}⟩ - 1 has been measured to
  high precision across bit sizes D = 20 to D = 120.

  c₁ is controlled by two exact distributions:
    • D = 2d (fraction 2(1-ln2) ≈ 0.614):
      c_{{top-1}} = 1 + I(pq ≥ 3·2^{{2d-2}}) → exact analytical formula
    • D = 2d-1 (fraction 2ln2-1 ≈ 0.386):
      c_{{top-1}} depends on carry chain boundary

  If c₁(∞) = ln(2)/4: this would mean the trace anomaly encodes
  the entropy of the binary carry chain: ln(2) is the bit entropy
  and 1/4 = 1/b² is the inverse-square of the base.

  Alternative: c₁(∞) may be a non-elementary constant defined
  by the stationary distribution of the carry Markov chain.
""")

    pr(f"  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
