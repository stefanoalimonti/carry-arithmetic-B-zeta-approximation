#!/usr/bin/env python3
"""
B23: Dirichlet Decomposition of R(l,s)

Tests whether R(l,s) = ⟨|det(I - M_l/l^s)|⟩ / |1 - l^{-s}|^{-1}
can be decomposed using Dirichlet characters, connecting it to L-functions.

Hypothesis: det(I - M_l/l^s) = ∏_{χ mod l} L_carry(s, χ)
If this factorization holds, R(l,s) factors into character-dependent pieces
with known functional equations.

Parts:
  A) Character decomposition of |det| via Dirichlet characters mod l
  B) Phase structure of det(I - M_l/l^s) in the complex plane
  C) R(l,s) decomposition via Möbius / log-expansion coefficients c_k
  D) Connection to ζ(2s): test R(l,s) ≈ (1-l^{-2s})^{-α} or R ≈ 1+β·Re(l^{-2s})
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


def compute_det_complex(p, q, l, s, base=2):
    """Compute det(I - M_l/l^s) as a complex number (not absolute value)."""
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
    return det_val


def dirichlet_characters(l):
    """Compute all Dirichlet characters mod l (prime l).

    Returns a dict: chi[j][n] = χ_j(n) for j=0..l-2, n=0..l-1.
    χ_j(n) = exp(2πi·j·ind_g(n)/(l-1)) for gcd(n,l)=1, else 0.
    """
    phi = l - 1
    g = _primitive_root(l)
    log_table = {}
    power = 1
    for k in range(phi):
        log_table[power] = k
        power = (power * g) % l

    chars = {}
    for j in range(phi):
        chi = {}
        chi[0] = 0.0 + 0.0j
        for n in range(1, l):
            idx = log_table[n]
            chi[n] = cmath.exp(2j * cmath.pi * j * idx / phi)
        chars[j] = chi
    return chars


def _primitive_root(l):
    """Find primitive root mod prime l."""
    if l == 2:
        return 1
    phi = l - 1
    factors = set()
    n = phi
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, l):
        if all(pow(g, phi // f, l) != 1 for f in factors):
            return g
    return None


def generate_semiprimes(n_samp, bits=16):
    """Generate semiprimes as (p, q, N=p*q) tuples."""
    results = []
    for _ in range(n_samp):
        p = random_prime(bits)
        q = random_prime(bits)
        if p == q:
            continue
        results.append((p, q, p * q))
    return results


def measure_R(l, s, semiprimes):
    """Measure R(l,s) using precomputed semiprimes. Returns (R, std_err)."""
    target = abs(1.0 / (1.0 - l ** (-s)))
    det_vals = []
    for p, q, N in semiprimes:
        d = compute_det_complex(p, q, l, s)
        if d is not None:
            ad = abs(d)
            if ad > 0 and not math.isnan(ad) and not math.isinf(ad):
                det_vals.append(ad)
    if len(det_vals) < 100:
        return None, None
    mean_det = np.mean(det_vals)
    se = np.std(det_vals) / math.sqrt(len(det_vals))
    return mean_det / target, se / target


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("P2-03: DIRICHLET DECOMPOSITION OF R(l,s)")
    pr("=" * 72)

    test_primes = [3, 5, 7]
    n_samp = 3000
    bits = 16

    pr(f"\nGenerating {n_samp} semiprimes ({bits}-bit factors)...")
    semiprimes = generate_semiprimes(n_samp, bits)
    pr(f"  Got {len(semiprimes)} semiprimes.")

    # ═══════════════════════════════════════════════════════════════
    # PART A: CHARACTER DECOMPOSITION OF |det|
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: CHARACTER DECOMPOSITION OF |det|")
    pr(f"{'═' * 72}")
    pr("""
  For each prime l, compute:
    L_carry(s, χ_j) = ⟨det(I - M/l^s) · χ_j(N mod l)⟩
  where χ_j are Dirichlet characters mod l.
  If det decomposes by character, these averages reveal the structure.
""")

    s_test = [complex(2.0, 0), complex(1.5, 0), complex(0.5, 14.13)]

    for l in test_primes:
        pr(f"  l = {l}:")
        chars = dirichlet_characters(l)
        phi = l - 1

        for s in s_test:
            s_label = f"σ={s.real:.1f}" if s.imag == 0 else f"s={s.real:.1f}+{s.imag:.2f}i"
            pr(f"\n    s = {s_label}:")

            det_by_residue = {r: [] for r in range(l)}
            all_dets = []

            for p, q, N in semiprimes:
                d = compute_det_complex(p, q, l, s)
                if d is None:
                    continue
                if abs(d) == 0 or math.isnan(abs(d)) or math.isinf(abs(d)):
                    continue
                r = N % l
                det_by_residue[r].append(d)
                all_dets.append((d, N))

            if len(all_dets) < 100:
                pr("      [insufficient data]")
                continue

            pr(f"      Total valid dets: {len(all_dets)}")
            pr(f"      ⟨|det|⟩ = {np.mean([abs(d) for d, _ in all_dets]):.6f}")

            for j in range(min(phi, 6)):
                chi = chars[j]
                weighted = []
                for d, N in all_dets:
                    r = N % l
                    if r == 0:
                        continue
                    weighted.append(d * chi[r])
                if len(weighted) < 50:
                    continue
                avg = np.mean(weighted)
                pr(f"      χ_{j}: ⟨det·χ_{j}(N)⟩ = {avg.real:+.6f} {avg.imag:+.6f}i, "
                   f"|·| = {abs(avg):.6f}")

            pr(f"      Counts by residue: "
               + ", ".join(f"{r}:{len(det_by_residue[r])}" for r in range(l)))

            for r in range(l):
                vals = det_by_residue[r]
                if len(vals) > 10:
                    mean_abs = np.mean([abs(v) for v in vals])
                    pr(f"      ⟨|det|⟩ for N≡{r} (mod {l}): {mean_abs:.6f} "
                       f"(n={len(vals)})")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART B: PHASE STRUCTURE OF det
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: PHASE STRUCTURE OF det(I - M_l/l^s)")
    pr(f"{'═' * 72}")
    pr("""
  The determinant is complex. Analyze:
    - Distribution of arg(det): uniform or structured?
    - Correlation between arg(det) and N mod l
    - Whether residue class predicts the phase
""")

    for l in test_primes:
        pr(f"  l = {l}:")

        for s in [complex(2.0, 0), complex(0.5, 14.13)]:
            s_label = f"σ={s.real:.1f}" if s.imag == 0 else f"s={s.real:.1f}+{s.imag:.2f}i"
            pr(f"\n    s = {s_label}:")

            phases_by_residue = {r: [] for r in range(l)}
            all_phases = []

            for p, q, N in semiprimes:
                d = compute_det_complex(p, q, l, s)
                if d is None or abs(d) < 1e-30:
                    continue
                if math.isnan(abs(d)) or math.isinf(abs(d)):
                    continue
                phase = cmath.phase(d)
                r = N % l
                phases_by_residue[r].append(phase)
                all_phases.append(phase)

            if len(all_phases) < 100:
                pr("      [insufficient data]")
                continue

            all_arr = np.array(all_phases)
            n_bins = 8
            bin_edges = np.linspace(-math.pi, math.pi, n_bins + 1)
            counts, _ = np.histogram(all_arr, bins=bin_edges)
            expected = len(all_arr) / n_bins
            chi2 = np.sum((counts - expected) ** 2 / expected)
            pr(f"      Phase distribution ({len(all_arr)} samples):")
            pr(f"        Histogram (8 bins): {list(counts)}")
            pr(f"        χ² vs uniform: {chi2:.1f} (df=7, p<0.05 if >14.1)")

            mean_cos = np.mean(np.cos(all_arr))
            mean_sin = np.mean(np.sin(all_arr))
            pr(f"        ⟨cos(arg)⟩ = {mean_cos:.6f}, ⟨sin(arg)⟩ = {mean_sin:.6f}")
            pr(f"        |⟨e^{{iφ}}⟩| = {math.sqrt(mean_cos**2 + mean_sin**2):.6f} "
               f"(0 = uniform, 1 = concentrated)")

            pr(f"      Phase by residue class:")
            for r in range(l):
                vals = phases_by_residue[r]
                if len(vals) > 10:
                    arr = np.array(vals)
                    mc = np.mean(np.cos(arr))
                    ms = np.mean(np.sin(arr))
                    mean_phase = math.atan2(ms, mc)
                    concentration = math.sqrt(mc**2 + ms**2)
                    pr(f"        N≡{r} (mod {l}): n={len(vals)}, "
                       f"mean phase={mean_phase:.4f}, "
                       f"concentration={concentration:.4f}")
        pr()

    # ═══════════════════════════════════════════════════════════════
    # PART C: R(l,s) DECOMPOSITION VIA LOG-EXPANSION
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: R(l,s) DECOMPOSITION — MÖBIUS COEFFICIENTS")
    pr(f"{'═' * 72}")
    pr("""
  Ansatz: R(l,s) = ∏_{k≥1} (1 + c_k/l^{ks})
  Taking log: ln(R) = Σ_k c_k/l^{ks} - Σ_k c_k²/(2l^{2ks}) + ...
  ≈ c_1/l^s + c_2/l^{2s} + ... for small c_k.

  Strategy: measure R at several σ values, then fit c_1, c_2, c_3.
""")

    sigma_vals = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    for l in test_primes:
        pr(f"  l = {l}:")

        R_vals = []
        for sigma in sigma_vals:
            R, R_err = measure_R(l, sigma, semiprimes)
            if R is None:
                R_vals.append(None)
                continue
            R_vals.append(R)
            pr(f"    σ={sigma:.1f}: R = {R:.8f}, ln(R) = {math.log(R):.8f}")

        valid = [(sigma_vals[i], R_vals[i]) for i in range(len(sigma_vals))
                 if R_vals[i] is not None and R_vals[i] > 0]
        if len(valid) < 3:
            pr("    [insufficient data for fitting]")
            continue

        sigmas = np.array([v[0] for v in valid])
        lnR = np.array([math.log(v[1]) for v in valid])

        A = np.column_stack([
            l ** (-k * sigmas) for k in range(1, 4)
        ])
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, lnR, rcond=None)
            c1, c2, c3 = coeffs
            pr(f"    Fit: ln(R) ≈ {c1:.6f}/l^s + {c2:.6f}/l^{{2s}} + {c3:.6f}/l^{{3s}}")
            pr(f"    c_1 = {c1:.6f}, c_2 = {c2:.6f}, c_3 = {c3:.6f}")

            pred = A @ coeffs
            residual = lnR - pred
            pr(f"    Max residual: {np.max(np.abs(residual)):.2e}")

            if abs(c2) > 1e-8:
                pr(f"    Ratio c_1/c_2 = {c1/c2:.4f}")
            if abs(c3) > 1e-8:
                pr(f"    Ratio c_2/c_3 = {c2/c3:.4f}")
        except Exception as e:
            pr(f"    Fit failed: {e}")

        pr()

    pr("  Euler product comparison:")
    pr("  If R = (1-l^{-2s})^{-α}, then ln(R) = α·Σ_{k≥1} l^{-2ks}/k")
    pr("  So c_1 = 0 (no l^{-s} term), c_2 = α, c_3 = 0, etc.\n")

    for l in test_primes:
        pr(f"  l = {l}: checking c_1 ≈ 0 hypothesis:")
        R_vals_check = []
        for sigma in sigma_vals:
            R, _ = measure_R(l, sigma, semiprimes)
            if R is not None and R > 0:
                R_vals_check.append((sigma, R))

        if len(R_vals_check) >= 3:
            sigmas_c = np.array([v[0] for v in R_vals_check])
            lnR_c = np.array([math.log(v[1]) for v in R_vals_check])
            A2 = np.column_stack([l ** (-2 * sigmas_c), l ** (-4 * sigmas_c)])
            try:
                c_even, _, _, _ = np.linalg.lstsq(A2, lnR_c, rcond=None)
                pr(f"    Even-only fit: ln(R) ≈ {c_even[0]:.6f}/l^{{2s}} + {c_even[1]:.6f}/l^{{4s}}")
                pred2 = A2 @ c_even
                resid2 = lnR_c - pred2
                pr(f"    Max residual (even-only): {np.max(np.abs(resid2)):.2e}")
            except Exception as e:
                pr(f"    Even fit failed: {e}")
    pr()

    # ═══════════════════════════════════════════════════════════════
    # PART D: CONNECTION TO ζ(2s)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: CONNECTION TO ζ(2s) — FITTING α AND β")
    pr(f"{'═' * 72}")
    pr("""
  Test 1: R(l,s) ≈ (1 - l^{-2s})^{-α}  — does α depend on s?
  Test 2: R(l,s) ≈ 1 + β·Re(l^{-2s})    — is β ≈ 1/6?
""")

    pr("  Test 1: α(σ) = ln(R)/ln(1-l^{-2σ})^{-1} at various σ:\n")

    for l in test_primes:
        pr(f"  l = {l}:")
        for sigma in [1.5, 2.0, 2.5, 3.0, 3.5]:
            R, _ = measure_R(l, sigma, semiprimes)
            if R is None or R <= 0:
                continue
            denom = -math.log(abs(1 - l ** (-2 * sigma)))
            if abs(denom) < 1e-12:
                continue
            alpha = math.log(R) / denom
            pr(f"    σ={sigma:.1f}: R={R:.8f}, α = {alpha:.6f}")
        pr()

    pr("\n  Test 2: β fitting — R(l,σ) ≈ 1 + β·Re(l^{-2σ})")
    pr("  Fitting β across multiple l at fixed σ:\n")

    for sigma in [1.5, 2.0, 2.5, 3.0]:
        xs = []
        ys = []
        details = []
        for l in test_primes:
            R, R_err = measure_R(l, sigma, semiprimes)
            if R is None:
                continue
            x_val = (l ** (-2 * sigma)).real
            xs.append(x_val)
            ys.append(R - 1)
            details.append((l, R, x_val))

        if len(xs) >= 2:
            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            beta_fit = np.sum(xs_arr * ys_arr) / np.sum(xs_arr ** 2)
            pr(f"    σ={sigma:.1f}: β = {beta_fit:.6f} (cf 1/6 ≈ {1/6:.6f})")
            for l_val, R_val, x_val in details:
                beta_local = (R_val - 1) / x_val if abs(x_val) > 1e-30 else float('inf')
                pr(f"      l={l_val}: R-1={R_val-1:+.8f}, "
                   f"Re(l^{{-2σ}})={x_val:.8f}, β_local={beta_local:.6f}")
        pr()

    pr("  Test on critical line: s = 0.5 + it")
    for t in [0.0, 5.0, 14.13, 21.02, 30.0]:
        s = complex(0.5, t)
        s_label = f"s=0.5+{t:.2f}i"
        pr(f"\n    {s_label}:")
        for l in test_primes:
            R, _ = measure_R(l, s, semiprimes)
            if R is None or R <= 0:
                continue
            l_minus2s = l ** (-2 * s)
            re_part = l_minus2s.real
            if abs(re_part) > 1e-30:
                beta_local = (R - 1) / re_part
            else:
                beta_local = float('inf')
            pr(f"      l={l}: R={R:.6f}, R-1={R-1:+.6f}, "
               f"Re(l^{{-2s}})={re_part:+.6f}, β_local={beta_local:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS — DIRICHLET DECOMPOSITION ANALYSIS")
    pr(f"{'═' * 72}")
    pr("""
  Questions answered:

  A) Character decomposition:
     - Do the character-weighted averages ⟨det · χ(N)⟩ vanish for χ ≠ χ_0?
     - If yes: det does NOT decompose by character (no L-function product).
     - If no:  specific characters χ contribute → L-function connection.

  B) Phase structure:
     - Is arg(det) uniform? If so, det is "phase-random" → R = ⟨|det|⟩.
     - Does residue class predict phase? If so, character decomposition works.

  C) Möbius coefficients:
     - Are odd coefficients c_1, c_3 ≈ 0? If so: only even powers l^{-2ks}.
     - This would mean R is a function of l^{-2s}, linking to ζ(2s).

  D) ζ(2s) connection:
     - Is β ≈ 1/6? If constant across l and σ, then R ≈ 1 + (1/6)·l^{-2σ}.
     - Is α constant? If so, R = (1-l^{-2s})^{-α} exactly.
""")

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
