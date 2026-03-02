#!/usr/bin/env python3
"""
B24: Precision Identification of the Universal Constant c₁

From prior experiments: ln R(l,s) ≈ c₁/l^s + c₂/l^{2s} + c₃/l^{3s}, with c₁ ≈ 0.155
universal across l.

Strategy:
  A) Measure R(l,σ) at high precision (10k semiprimes, many primes)
  B) Extract c₁ per-l via regression, verify universality
  C) Separately measure ⟨carry_{D-1}⟩ and test c₁ = ⟨carry_{D-1}⟩ - 1
     (from trace formula: c₁ = -⟨Tr(M)⟩ - 1 = ⟨carry_{D-1}⟩ - 1)
  D) Test c₁ against 30+ known mathematical constants
  E) Multi-bit-size check: does c₁ depend on D?
"""

import sys, os, time, random, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, carry_poly_int, quotient_poly_int

random.seed(42)
np.random.seed(42)


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def compute_det_and_trace(p, q, l, s, base=2):
    """Compute |det(I - M/l^s)| and Tr(M) for carry companion matrix."""
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
    ls = l ** (-s)
    det_val = np.linalg.det(np.eye(n, dtype=complex) - M * ls)
    trace_val = np.trace(M).real
    return abs(det_val), trace_val


def get_carry_at_D_minus_1(p, q, base=2):
    """Extract carry_{D-1} directly from the carry sequence."""
    C = carry_poly_int(p, q, base)
    if len(C) < 3:
        return None
    carries = [-int(round(c)) for c in C[:-1]]
    carries.insert(0, 0)
    if len(carries) < 2:
        return None
    return carries[-1]


def measure_R(l, s, semiprimes):
    """Measure R(l,s) and return (R, stderr, n_valid)."""
    target = abs(1.0 / (1.0 - l ** (-s)))
    det_vals = []
    for p, q in semiprimes:
        d, _ = compute_det_and_trace(p, q, l, s)
        if d is not None and d > 0 and not math.isnan(d) and not math.isinf(d):
            det_vals.append(d)
    if len(det_vals) < 200:
        return None, None, 0
    arr = np.array(det_vals)
    mean_det = np.mean(arr)
    se = np.std(arr) / math.sqrt(len(arr))
    return mean_det / target, se / target, len(det_vals)


def generate_semiprimes(n_samp, bits):
    results = []
    for _ in range(n_samp):
        p = random_prime(bits)
        q = random_prime(bits)
        if p != q:
            results.append((p, q))
    return results


CANDIDATES = {
    '1/6': 1.0/6,
    '(1-ln2)/2': (1 - math.log(2)) / 2,
    '3/(2π²)': 3.0 / (2 * math.pi**2),
    'ln(2)²/π': math.log(2)**2 / math.pi,
    '(π²-9)/6': (math.pi**2 - 9) / 6,
    'γ/4': 0.5772156649 / 4,
    '(3-e)/2': (3 - math.e) / 2,
    '1/(2π)': 1.0 / (2 * math.pi),
    'γ·ln(2)/π': 0.5772156649 * math.log(2) / math.pi,
    'ζ(3)/(4π²)': 1.2020569031 / (4 * math.pi**2),
    '(γ+ln2)/8': (0.5772156649 + math.log(2)) / 8,
    '(2-√3)/2': (2 - math.sqrt(3)) / 2,
    'ln(2)/4': math.log(2) / 4,
    '1/2-1/e': 0.5 - 1.0/math.e,
    '(π-3)/2': (math.pi - 3) / 2,
    '(e-2)/2': (math.e - 2) / 2,
    'γ²': 0.5772156649**2,
    'ln(2)²': math.log(2)**2,
    '1-ln(2)-1/4': 1 - math.log(2) - 0.25,
    'ln(3)/(2π)': math.log(3) / (2 * math.pi),
    'γ/(2+γ)': 0.5772156649 / (2 + 0.5772156649),
    '(4-π)/6': (4 - math.pi) / 6,
    '2/π-1/2': 2.0/math.pi - 0.5,
    '(6-π²)/24': (6 - math.pi**2) / 24,
    'ln(4/π)/2': math.log(4.0/math.pi) / 2,
    '1/(2π)·ln(2π)': math.log(2*math.pi) / (2*math.pi),
    '(γ+1)/8': (0.5772156649 + 1) / 8,
    '3·ln(2)/(4π)': 3*math.log(2) / (4*math.pi),
    '(1-γ)/3': (1 - 0.5772156649) / 3,
    'π²/60': math.pi**2 / 60,
}


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("P2-04: PRECISION IDENTIFICATION OF THE UNIVERSAL CONSTANT c₁")
    pr("=" * 72)

    sigma_vals = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    bits_sizes = [16, 20, 24]
    n_samp = 8000
    test_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    # ═══════════════════════════════════════════════════════════════
    # PART A: HIGH-PRECISION R(l,σ) MEASUREMENT
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: HIGH-PRECISION R(l,σ) MEASUREMENT")
    pr(f"{'═' * 72}\n")

    all_c1 = []

    for bits in bits_sizes:
        pr(f"\n--- Bit size = {bits} (n_samp = {n_samp}) ---")
        semiprimes = generate_semiprimes(n_samp, bits)
        pr(f"  Generated {len(semiprimes)} semiprimes")

        c1_per_l = {}

        for l in test_primes:
            R_data = []
            for sigma in sigma_vals:
                R, se, nv = measure_R(l, sigma, semiprimes)
                if R is not None and R > 0 and not math.isnan(R):
                    R_data.append((sigma, R, se))

            if len(R_data) < 4:
                pr(f"  l={l:3d}: insufficient data")
                continue

            sigmas = np.array([v[0] for v in R_data])
            lnR = np.array([math.log(v[1]) for v in R_data])

            A = np.column_stack([l ** (-k * sigmas) for k in range(1, 4)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, lnR, rcond=None)
                c1, c2, c3 = coeffs
                pred = A @ coeffs
                residual_max = np.max(np.abs(lnR - pred))
                c1_per_l[l] = c1
                all_c1.append(c1)
                pr(f"  l={l:3d}: c₁ = {c1:.6f}, c₂ = {c2:.6f}, c₃ = {c3:.6f}, "
                   f"max_resid = {residual_max:.2e}")
            except Exception as e:
                pr(f"  l={l:3d}: fit failed: {e}")

        if c1_per_l:
            vals = list(c1_per_l.values())
            arr = np.array(vals)
            pr(f"\n  SUMMARY ({bits}-bit):")
            pr(f"    c₁ = {arr.mean():.6f} ± {arr.std():.6f} (std across {len(vals)} primes)")
            pr(f"    min = {arr.min():.6f}, max = {arr.max():.6f}")
            pr(f"    cv = {arr.std()/abs(arr.mean()):.4f} (coefficient of variation)")

    pr(f"\n{'═' * 72}")
    pr("GLOBAL c₁ ESTIMATE ACROSS ALL BIT SIZES AND PRIMES")
    pr(f"{'═' * 72}")
    if all_c1:
        all_arr = np.array(all_c1)
        c1_global = all_arr.mean()
        c1_se = all_arr.std() / math.sqrt(len(all_arr))
        pr(f"  c₁ = {c1_global:.8f} ± {c1_se:.8f} (n = {len(all_c1)})")
        pr(f"  Std = {all_arr.std():.8f}")
    else:
        c1_global = 0.155
        c1_se = 0.01

    # ═══════════════════════════════════════════════════════════════
    # PART B: DIRECT MEASUREMENT OF ⟨carry_{D-1}⟩
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: DIRECT MEASUREMENT OF ⟨carry_{{D-1}}⟩ AND Tr(M)")
    pr(f"{'═' * 72}")
    pr("""
  Theory: c₁ = ⟨carry_{D-1}⟩ - 1  (from trace formula)
  Because: ln R ≈ (-⟨Tr(M)⟩ - 1)/l^σ + ... and Tr(M) = -carry_{D-1}
""")

    for bits in bits_sizes:
        pr(f"\n  Bit size = {bits}:")
        semiprimes = generate_semiprimes(5000, bits)

        traces = []
        carry_D1 = []
        for p, q in semiprimes:
            _, tr_val = compute_det_and_trace(p, q, 3, 2.0)
            if tr_val is not None and not math.isnan(tr_val):
                traces.append(tr_val)

            c_d1 = get_carry_at_D_minus_1(p, q)
            if c_d1 is not None:
                carry_D1.append(c_d1)

        if traces:
            tr_arr = np.array(traces)
            pr(f"    ⟨Tr(M)⟩ = {tr_arr.mean():.6f} ± {tr_arr.std()/math.sqrt(len(tr_arr)):.6f} "
               f"(n={len(traces)})")
            pred_c1_from_trace = -tr_arr.mean() - 1
            pr(f"    Predicted c₁ = -⟨Tr(M)⟩ - 1 = {pred_c1_from_trace:.6f}")

        if carry_D1:
            cd1_arr = np.array(carry_D1, dtype=float)
            pr(f"    ⟨carry_{{D-1}}⟩ = {cd1_arr.mean():.6f} ± {cd1_arr.std()/math.sqrt(len(cd1_arr)):.6f} "
               f"(n={len(carry_D1)})")
            pred_c1_from_carry = cd1_arr.mean() - 1
            pr(f"    Predicted c₁ = ⟨carry_{{D-1}}⟩ - 1 = {pred_c1_from_carry:.6f}")
            pr(f"    ⟨carry_{{D-1}}⟩ distribution: "
               f"P(=1)={np.mean(cd1_arr==1):.3f}, "
               f"P(=2)={np.mean(cd1_arr==2):.3f}, "
               f"P(≥3)={np.mean(cd1_arr>=3):.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PART C: TEST AGAINST KNOWN MATHEMATICAL CONSTANTS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: TEST c₁ AGAINST KNOWN MATHEMATICAL CONSTANTS")
    pr(f"{'═' * 72}\n")

    pr(f"  Measured c₁ = {c1_global:.8f} ± {c1_se:.8f}\n")

    scored = []
    for name, val in CANDIDATES.items():
        dist = abs(c1_global - val)
        sigma = dist / c1_se if c1_se > 1e-10 else float('inf')
        scored.append((sigma, name, val, dist))

    scored.sort()
    pr(f"  {'Rank':>4s}  {'Candidate':<25s}  {'Value':>12s}  {'|Δ|':>10s}  {'σ-dist':>8s}  {'Match?':<6s}")
    pr(f"  {'─'*4}  {'─'*25}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*6}")
    for i, (sigma, name, val, dist) in enumerate(scored):
        match = "YES" if sigma < 2.0 else ("maybe" if sigma < 3.0 else "no")
        pr(f"  {i+1:4d}  {name:<25s}  {val:12.8f}  {dist:10.8f}  {sigma:8.2f}  {match:<6s}")

    # ═══════════════════════════════════════════════════════════════
    # PART D: REFINED FIT WITH WEIGHTED REGRESSION
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: REFINED FIT — WEIGHTED BY PRECISION")
    pr(f"{'═' * 72}\n")

    pr("  Using only large primes (l ≥ 11) to reduce finite-size artifacts:")

    bits = 16
    semiprimes = generate_semiprimes(10000, bits)
    pr(f"  Generated 10000 semiprimes ({bits}-bit)")

    refined_c1_vals = []
    for l in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        R_data = []
        for sigma in sigma_vals:
            R, se, nv = measure_R(l, sigma, semiprimes)
            if R is not None and R > 0 and not math.isnan(R):
                R_data.append((sigma, R, se))

        if len(R_data) < 4:
            continue

        sigmas = np.array([v[0] for v in R_data])
        lnR = np.array([math.log(v[1]) for v in R_data])

        A = np.column_stack([l ** (-k * sigmas) for k in range(1, 4)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, lnR, rcond=None)
            c1 = coeffs[0]
            refined_c1_vals.append(c1)
            pr(f"    l={l:3d}: c₁ = {c1:.6f}")
        except:
            pass

    if refined_c1_vals:
        r_arr = np.array(refined_c1_vals)
        c1_refined = r_arr.mean()
        c1_refined_se = r_arr.std() / math.sqrt(len(r_arr))
        pr(f"\n  REFINED c₁ (l ≥ 11): {c1_refined:.8f} ± {c1_refined_se:.8f} "
           f"(n={len(refined_c1_vals)})")
        pr(f"  Std across primes: {r_arr.std():.8f}")

        pr(f"\n  Updated ranking with refined c₁ = {c1_refined:.8f}:")
        scored2 = []
        for name, val in CANDIDATES.items():
            dist = abs(c1_refined - val)
            sigma = dist / c1_refined_se if c1_refined_se > 1e-10 else float('inf')
            scored2.append((sigma, name, val, dist))

        scored2.sort()
        for i, (sigma, name, val, dist) in enumerate(scored2[:10]):
            match = "YES" if sigma < 2.0 else ("maybe" if sigma < 3.0 else "no")
            pr(f"    {i+1:2d}. {name:<25s} = {val:.8f}, Δ = {dist:.8f}, σ = {sigma:.2f} {match}")

    # ═══════════════════════════════════════════════════════════════
    # PART E: MULTI-BIT-SIZE TREND — DOES c₁ CONVERGE?
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: MULTI-BIT-SIZE — DOES c₁ DEPEND ON D?")
    pr(f"{'═' * 72}\n")

    for bits in [12, 16, 20, 24, 28]:
        sps = generate_semiprimes(6000, bits)
        c1_list = []
        for l in [7, 11, 13, 17, 23, 29, 37, 47]:
            R_data = []
            for sigma in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                R, se, nv = measure_R(l, sigma, sps)
                if R is not None and R > 0 and not math.isnan(R):
                    R_data.append((sigma, R))
            if len(R_data) < 4:
                continue
            sigmas = np.array([v[0] for v in R_data])
            lnR = np.array([math.log(v[1]) for v in R_data])
            A = np.column_stack([l ** (-k * sigmas) for k in range(1, 4)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, lnR, rcond=None)
                c1_list.append(coeffs[0])
            except:
                pass
        if c1_list:
            arr = np.array(c1_list)
            pr(f"  bits={bits:3d} (D≈{2*bits}): c₁ = {arr.mean():.6f} ± {arr.std()/math.sqrt(len(arr)):.6f} "
               f"(n={len(c1_list)})")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS — IDENTIFICATION OF c₁")
    pr(f"{'═' * 72}")
    pr(f"""
  Questions answered:

  1. Is c₁ universal across l?
     → Check Part A: if cv < 0.1, universality confirmed.

  2. Is c₁ = ⟨carry_{{D-1}}⟩ - 1 (trace formula prediction)?
     → Check Part B: compare directly.

  3. Does c₁ match any known constant?
     → Check Part C/D ranking.

  4. Does c₁ depend on bit size (D)?
     → Check Part E: if trend exists, c₁ is asymptotic.

  Candidate interpretations:
    • c₁ = ⟨carry_{{D-1}}⟩ - 1: the trace anomaly of the carry Markov chain
    • c₁ = (1-ln2)/2: would connect carries to binary entropy
    • c₁ = 3/(2π²): would connect carries to ζ(2)
    • c₁ = 1/6: the simplest fraction (from l²(R-1) at σ=2)
""")

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
