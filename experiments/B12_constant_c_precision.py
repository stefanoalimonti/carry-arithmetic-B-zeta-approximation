#!/usr/bin/env python3
"""
B12: High-precision measurement of the constant c in (b-1)/b + c/l.

Two independent measurements:
  A. Per-factor identity: h(l) = ⟨det(I - M/l)⟩ * (1 - 1/l) = 1 + c/l + O(1/l²)
  B. Anti-correlation ratio: ⟨ratio(N,l)⟩ = 1/2 + c'/l + O(1/l²)

Candidates:  π²/36 ≈ 0.27416,  1/4 = 0.25000,  ln(2)/2 ≈ 0.34657

Method: polynomial Horner evaluation (exact integer → float64 for 32/64-bit),
        weighted least-squares fit with 1/l and 1/l² terms.
"""

import sys, os, time, random, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import (random_prime, carry_poly_int, quotient_poly_int,
                          eval_poly_mod, primes_up_to)

random.seed(12345)
np.random.seed(12345)


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def horner_det(Q, l):
    """Compute det(I - M/l) where M is companion matrix of Q(x).

    Uses: det(I - M/l) = Q(l) / (Q_lead * l^{D-1})
    Evaluated via Horner scheme on the monic polynomial for numerical stability.
    """
    D = len(Q)
    if D < 2:
        return 1.0
    lead = float(Q[-1])
    if abs(lead) < 1e-30:
        return float('nan')

    # det(I-M*y) = 1 + a_{D-2}*y + a_{D-3}*y² + ... + a_0*y^{D-1}
    # where a_k = Q[k]/lead, and a_{D-1} = 1 (monic).
    # Horner from the highest power of y (= 1/l):
    val = Q[0] / lead
    for k in range(1, D - 1):
        val = Q[k] / lead + val / l
    val = 1.0 + val / l
    return val


def generate_semiprime_data(bits, n_samples):
    """Generate semiprimes and extract quotient polynomials."""
    data = []
    attempts = 0
    while len(data) < n_samples and attempts < n_samples * 10:
        attempts += 1
        p = random_prime(bits)
        q = random_prime(bits)
        if p == q:
            continue
        C = carry_poly_int(p, q, 2)
        Q = quotient_poly_int(C, 2)
        if len(Q) < 3:
            continue
        data.append({'Q': Q, 'p': p, 'q': q})
    return data


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("B12: HIGH-PRECISION MEASUREMENT OF CONSTANT c")
    pr("=" * 72)

    # ═══════════════════════════════════════════════════════════════
    # DATA GENERATION
    # ═══════════════════════════════════════════════════════════════
    configs = [
        (16, 30000),
        (24, 25000),
        (32, 20000),
        (48, 10000),
        (64, 5000),
    ]

    all_data = {}
    for bits, n_target in configs:
        pr(f"\n  Generating {n_target} semiprimes at {bits}-bit...")
        data = generate_semiprime_data(bits, n_target)
        all_data[bits] = data
        Ds = [len(d['Q']) for d in data]
        pr(f"    Got {len(data)}, D ≈ {np.mean(Ds):.0f}")

    test_primes = primes_up_to(500)
    test_primes = [l for l in test_primes if l >= 3]
    pr(f"\n  Test primes: {len(test_primes)} primes from 3 to {test_primes[-1]}")

    # ═══════════════════════════════════════════════════════════════
    # PART A: PER-FACTOR IDENTITY — h(l) = ⟨det(I-M/l)⟩ * (1-1/l)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: PER-FACTOR IDENTITY h(l) = ⟨det(I-M/l)⟩ × (1 - 1/l)")
    pr(f"{'═' * 72}")

    for bits in [16, 32, 64]:
        data = all_data.get(bits)
        if not data:
            continue

        h_means = []
        h_stderrs = []
        l_vals = []

        for l in test_primes:
            h_vals = []
            for d in data:
                det_val = horner_det(d['Q'], l)
                if not math.isfinite(det_val):
                    continue
                h = abs(det_val) * (1.0 - 1.0 / l)
                h_vals.append(h)

            if len(h_vals) < 100:
                continue
            mean_h = np.mean(h_vals)
            stderr = np.std(h_vals) / math.sqrt(len(h_vals))
            h_means.append(mean_h)
            h_stderrs.append(stderr)
            l_vals.append(l)

        h_means = np.array(h_means)
        h_stderrs = np.array(h_stderrs)
        l_arr = np.array(l_vals, dtype=float)

        pr(f"\n  {bits}-bit ({len(data)} semiprimes, {len(l_vals)} primes):")

        # Show h(l) for a few small primes
        for i in range(min(8, len(l_vals))):
            l = l_vals[i]
            pr(f"    l={l:4d}: h(l) = {h_means[i]:.8f} ± {h_stderrs[i]:.6f}  "
               f"(h-1 = {h_means[i]-1:.8f})")

        # Fit: h(l) - 1 = c/l + c2/l²
        y = h_means - 1.0
        X = np.column_stack([1.0 / l_arr, 1.0 / l_arr**2])
        w = 1.0 / (h_stderrs**2 + 1e-20)

        # Weighted least squares
        Xw = X * np.sqrt(w[:, None])
        yw = y * np.sqrt(w)
        coeffs, residuals, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        c_fit = coeffs[0]
        c2_fit = coeffs[1]

        # Bootstrap error on c
        n_boot = 2000
        c_boot = []
        for _ in range(n_boot):
            idx = np.random.choice(len(y), len(y), replace=True)
            Xb = Xw[idx]
            yb = yw[idx]
            cb, _, _, _ = np.linalg.lstsq(Xb, yb, rcond=None)
            c_boot.append(cb[0])
        c_err = np.std(c_boot)

        pr(f"\n    Fit: h(l) - 1 = c/l + c₂/l²")
        pr(f"    c  = {c_fit:.6f} ± {c_err:.6f}")
        pr(f"    c₂ = {c2_fit:.4f}")
        pr(f"\n    Candidates:")
        pr(f"      π²/36   = {math.pi**2/36:.6f}  "
           f"  Δ = {abs(c_fit - math.pi**2/36):.6f}  "
           f"  ({abs(c_fit - math.pi**2/36)/c_err:.1f}σ)")
        pr(f"      1/4     = {0.25:.6f}  "
           f"  Δ = {abs(c_fit - 0.25):.6f}  "
           f"  ({abs(c_fit - 0.25)/c_err:.1f}σ)")
        pr(f"      ln(2)/2 = {math.log(2)/2:.6f}  "
           f"  Δ = {abs(c_fit - math.log(2)/2):.6f}  "
           f"  ({abs(c_fit - math.log(2)/2)/c_err:.1f}σ)")

        # Also fit with 3 terms: c/l + c2/l² + c3/l³
        X3 = np.column_stack([1.0/l_arr, 1.0/l_arr**2, 1.0/l_arr**3])
        X3w = X3 * np.sqrt(w[:, None])
        coeffs3, _, _, _ = np.linalg.lstsq(X3w, yw, rcond=None)
        pr(f"\n    3-term fit: c={coeffs3[0]:.6f}, c₂={coeffs3[1]:.4f}, "
           f"c₃={coeffs3[2]:.2f}")

        # Fit restricted to large l only (l > 20)
        mask = l_arr > 20
        if np.sum(mask) > 5:
            Xm = Xw[mask]
            ym = yw[mask]
            cm, _, _, _ = np.linalg.lstsq(Xm, ym, rcond=None)
            pr(f"    Large-l fit (l>20): c = {cm[0]:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # PART B: ANTI-CORRELATION RATIO (modular arithmetic)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: ANTI-CORRELATION RATIO ⟨ratio⟩ = 1/2 + c'/l + O(1/l²)")
    pr(f"{'═' * 72}")

    small_primes = [l for l in test_primes if l <= 101]

    for bits in [32, 64]:
        data = all_data.get(bits)
        if not data:
            continue

        ratio_means = []
        ratio_stderrs = []
        l_vals_b = []

        for l in small_primes:
            ratios = []
            for d in data:
                Q = d['Q']
                pm = d['p'] % l
                qm = d['q'] % l

                # Count roots of Q mod l
                n_roots = 0
                for x in range(l):
                    if eval_poly_mod(Q, x, l) == 0:
                        n_roots += 1

                if n_roots == 0:
                    continue

                hits = 0
                if eval_poly_mod(Q, pm, l) == 0:
                    hits += 1
                if qm != pm:
                    if eval_poly_mod(Q, qm, l) == 0:
                        hits += 1
                    n_tested = 2
                else:
                    n_tested = 1

                expected = n_roots / l * n_tested
                if expected > 0:
                    ratios.append(hits / expected)

            if len(ratios) < 100:
                continue

            ratio_means.append(np.mean(ratios))
            ratio_stderrs.append(np.std(ratios) / math.sqrt(len(ratios)))
            l_vals_b.append(l)

        if not l_vals_b:
            continue

        ratio_means = np.array(ratio_means)
        ratio_stderrs = np.array(ratio_stderrs)
        l_arr = np.array(l_vals_b, dtype=float)

        pr(f"\n  {bits}-bit ({len(data)} semiprimes, {len(l_vals_b)} primes ≤ 101):")
        for i in range(min(8, len(l_vals_b))):
            pr(f"    l={l_vals_b[i]:4d}: ratio = {ratio_means[i]:.6f} ± {ratio_stderrs[i]:.6f}")

        # Fit: ratio - 0.5 = c'/l + c'₂/l²
        y = ratio_means - 0.5
        X = np.column_stack([1.0/l_arr, 1.0/l_arr**2])
        w = 1.0 / (ratio_stderrs**2 + 1e-20)
        Xw = X * np.sqrt(w[:, None])
        yw = y * np.sqrt(w)
        coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

        n_boot = 2000
        c_boot = []
        for _ in range(n_boot):
            idx = np.random.choice(len(y), len(y), replace=True)
            cb, _, _, _ = np.linalg.lstsq(Xw[idx], yw[idx], rcond=None)
            c_boot.append(cb[0])
        c_err = np.std(c_boot)

        pr(f"\n    Fit: ratio - 1/2 = c'/l + c'₂/l²")
        pr(f"    c' = {coeffs[0]:.6f} ± {c_err:.6f}")
        pr(f"    c'₂ = {coeffs[1]:.4f}")
        pr(f"    Candidates:")
        pr(f"      π²/36   = {math.pi**2/36:.6f}  "
           f"  ({abs(coeffs[0] - math.pi**2/36)/c_err:.1f}σ)")
        pr(f"      1/4     = {0.25:.6f}  "
           f"  ({abs(coeffs[0] - 0.25)/c_err:.1f}σ)")

    # ═══════════════════════════════════════════════════════════════
    # PART C: DEPENDENCE ON SEMIPRIME SIZE
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: DOES c DEPEND ON THE SEMIPRIME SIZE?")
    pr(f"{'═' * 72}")

    c_by_bits = {}
    for bits in [16, 24, 32, 48, 64]:
        data = all_data.get(bits)
        if not data or len(data) < 1000:
            continue

        h_means_local = []
        l_vals_local = []
        stderrs_local = []

        for l in test_primes:
            if l > 200:
                continue
            hvals = []
            for d in data:
                det_val = horner_det(d['Q'], l)
                if math.isfinite(det_val):
                    hvals.append(abs(det_val) * (1.0 - 1.0/l))
            if len(hvals) < 100:
                continue
            h_means_local.append(np.mean(hvals))
            stderrs_local.append(np.std(hvals) / math.sqrt(len(hvals)))
            l_vals_local.append(l)

        if len(l_vals_local) < 5:
            continue

        y = np.array(h_means_local) - 1.0
        l_arr = np.array(l_vals_local, dtype=float)
        X = np.column_stack([1.0/l_arr, 1.0/l_arr**2])
        w = 1.0 / (np.array(stderrs_local)**2 + 1e-20)
        Xw = X * np.sqrt(w[:, None])
        yw = y * np.sqrt(w)
        coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

        n_boot = 1000
        cb_list = []
        for _ in range(n_boot):
            idx = np.random.choice(len(y), len(y), replace=True)
            cb, _, _, _ = np.linalg.lstsq(Xw[idx], yw[idx], rcond=None)
            cb_list.append(cb[0])
        c_err = np.std(cb_list)

        c_by_bits[bits] = (coeffs[0], c_err)
        mean_D = np.mean([len(d['Q']) for d in data])
        pr(f"  {bits:3d}-bit  D≈{mean_D:4.0f}  "
           f"c = {coeffs[0]:.6f} ± {c_err:.6f}  "
           f"c₂ = {coeffs[1]:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PART D: ADDITIONAL CANDIDATE TEST — EXACT COMBINATORIAL VALUES
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: EXPANDED CANDIDATE LIST")
    pr(f"{'═' * 72}")

    candidates = {
        'π²/36':      math.pi**2 / 36,
        '1/4':        0.25,
        'ln(2)/2':    math.log(2) / 2,
        '1/(2π)':     1.0 / (2 * math.pi),
        'π/12':       math.pi / 12,
        '(3-e)/2':    (3 - math.e) / 2,
        '1/e':        1.0 / math.e,
        '1-ln(2)':    1 - math.log(2),
        'γ/2':        0.5772156649 / 2,
        '2/7':        2.0 / 7,
        '3/11':       3.0 / 11,
    }

    if c_by_bits:
        best_bits = max(c_by_bits.keys())
        c_best, c_best_err = c_by_bits[best_bits]
        pr(f"\n  Best measurement: c = {c_best:.6f} ± {c_best_err:.6f} "
           f"({best_bits}-bit)")
        pr(f"\n  {'Candidate':>12s}  {'Value':>10s}  {'Δ':>10s}  {'σ':>6s}")
        sorted_cands = sorted(candidates.items(),
                              key=lambda x: abs(x[1] - c_best))
        for name, val in sorted_cands:
            delta = abs(val - c_best)
            sigma = delta / c_best_err if c_best_err > 0 else float('inf')
            marker = " ←" if sigma < 2 else ""
            pr(f"  {name:>12s}  {val:10.6f}  {delta:10.6f}  {sigma:6.1f}{marker}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS")
    pr(f"{'═' * 72}")

    if c_by_bits:
        c_vals = [v[0] for v in c_by_bits.values()]
        c_errs = [v[1] for v in c_by_bits.values()]

        # Weighted average over all bit sizes
        w = [1.0/e**2 for e in c_errs]
        c_combined = sum(c*w_ for c, w_ in zip(c_vals, w)) / sum(w)
        c_combined_err = 1.0 / math.sqrt(sum(w))

        pr(f"\n  Combined measurement (weighted average over all bit sizes):")
        pr(f"  c = {c_combined:.6f} ± {c_combined_err:.6f}")
        pr(f"\n  Distance to candidates:")
        pr(f"    π²/36   = {math.pi**2/36:.6f}  → "
           f"{abs(c_combined - math.pi**2/36)/c_combined_err:.1f}σ")
        pr(f"    1/4     = {0.25:.6f}  → "
           f"{abs(c_combined - 0.25)/c_combined_err:.1f}σ")
        pr(f"    ln(2)/2 = {math.log(2)/2:.6f}  → "
           f"{abs(c_combined - math.log(2)/2)/c_combined_err:.1f}σ")

        if abs(c_combined - math.pi**2/36) < abs(c_combined - 0.25):
            pr(f"\n  → π²/36 is the closer candidate")
        else:
            pr(f"\n  → 1/4 is the closer candidate")

        gap_sigma = abs(math.pi**2/36 - 0.25) / c_combined_err
        pr(f"  → Separation between π²/36 and 1/4: {gap_sigma:.1f}σ")
        if gap_sigma >= 5:
            pr(f"  → DISCRIMINABLE at 5σ level!")
        elif gap_sigma >= 3:
            pr(f"  → Marginal discrimination (3-5σ)")
        else:
            pr(f"  → Cannot discriminate ({gap_sigma:.1f}σ < 3σ)")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == "__main__":
    main()
