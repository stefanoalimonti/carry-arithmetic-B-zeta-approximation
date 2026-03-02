#!/usr/bin/env python3
"""Analysis of B30 exact enumeration data: K=3..19 (+ K=20 when available).

Extends the convergence model with K=17,18,19 data from overnight enumeration.
Tests:
  1. Polynomial-geometric convergence model: c1(K) = π/18 + A·K·(1/2)^K
  2. Richardson extrapolation with new data
  3. Precision improvement: how many digits do we get at K=19?
  4. PSLQ on extrapolated values
  5. Σ_even convergence (should be pure geometric)
"""

import mpmath
from mpmath import mpf, mp, log, pi, sqrt

mp.dps = 60

# ══════════════════════════════════════════════════════════════════
# EXACT DATA: So_num/Nt and Se_num/Nt from prior experiments enumeration
# Nt(K) = 4^(K-1) for K-bit factors (2^{K-1} to 2^K - 1)
# ══════════════════════════════════════════════════════════════════

DATA = {
    3:  {'So_num':         -1, 'Se_num':          1, 'Nt': 4**2},
    4:  {'So_num':         -6, 'Se_num':          6, 'Nt': 4**3},
    5:  {'So_num':        -18, 'Se_num':         28, 'Nt': 4**4},
    6:  {'So_num':        -51, 'Se_num':        126, 'Nt': 4**5},
    7:  {'So_num':       -116, 'Se_num':        531, 'Nt': 4**6},
    8:  {'So_num':       -100, 'Se_num':       2182, 'Nt': 4**7},
    9:  {'So_num':        600, 'Se_num':       8848, 'Nt': 4**8},
    10: {'So_num':       5140, 'Se_num':      35651, 'Nt': 4**9},
    11: {'So_num':      27926, 'Se_num':     143086, 'Nt': 4**10},
    12: {'So_num':     130210, 'Se_num':     573404, 'Nt': 4**11},
    13: {'So_num':     566916, 'Se_num':    2295631, 'Nt': 4**12},
    14: {'So_num':    2377258, 'Se_num':    9186707, 'Nt': 4**13},
    15: {'So_num':    9768054, 'Se_num':   36755074, 'Nt': 4**14},
    16: {'So_num':   39664988, 'Se_num':  147036644, 'Nt': 4**15},
    # NEW from overnight B30 run:
    17: {'So_num':  160017931, 'Se_num':  588179206, 'Nt': 4**16},
    18: {'So_num':  643095621, 'Se_num': 2352782572, 'Nt': 4**17},
    19: {'So_num': 2579119971, 'Se_num': 9411261235, 'Nt': 4**18},
}

TARGET_C1 = pi / 18
TARGET_SO = pi / 18 - 1 + 6 * log(2) - 3 * log(3)
TARGET_SE = 1 + 3 * log(mpf(3) / 4)

for K in sorted(DATA):
    d = DATA[K]
    d['So'] = mpf(d['So_num']) / d['Nt']
    d['Se'] = mpf(d['Se_num']) / d['Nt']
    d['c1'] = d['So'] + d['Se']


def pr(*a, **kw):
    print(*a, **kw)


# ══════════════════════════════════════════════════════════════════
# 1. RAW CONVERGENCE
# ══════════════════════════════════════════════════════════════════
pr("=" * 72)
pr("1. RAW CONVERGENCE: c₁(K) vs π/18")
pr("=" * 72)

Ks = sorted(DATA)
pr(f"{'K':>3} {'c₁(K)':>22} {'Δ(π/18)':>14} {'digits':>6} {'ρ(K)':>8}")

prev_delta = None
for K in Ks:
    c1 = DATA[K]['c1']
    delta = float(c1 - TARGET_C1)
    digits = -mpmath.log10(abs(c1 - TARGET_C1)) if c1 != TARGET_C1 else 99
    rho = ""
    if prev_delta is not None and prev_delta != 0:
        rho = f"{abs(delta / prev_delta):.4f}"
    pr(f"{K:3d} {float(c1):22.18f} {delta:+14.8e} {float(digits):6.2f} {rho:>8}")
    prev_delta = delta

# ══════════════════════════════════════════════════════════════════
# 2. POLYNOMIAL-GEOMETRIC MODEL: c1(K) = π/18 + (A₀K + A₁)(1/2)^K
# ══════════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("2. POLYNOMIAL-GEOMETRIC MODEL: Δ(K) = c₁(K) - π/18")
pr("   Testing: Δ(K) · 2^K = A₀·K + A₁")
pr("=" * 72)

for K in Ks:
    c1 = DATA[K]['c1']
    delta = c1 - TARGET_C1
    f_K = float(delta * mpf(2)**K)
    pr(f"  K={K:2d}: Δ·2^K = {f_K:12.4f}")

pr("\n  Linear fit f(K) = A₀·K + A₁ using K=15..19:")
use_Ks = [K for K in Ks if K >= 15]
fvals = [float((DATA[K]['c1'] - TARGET_C1) * mpf(2)**K) for K in use_Ks]
n = len(use_Ks)
sx = sum(use_Ks)
sy = sum(fvals)
sxx = sum(k*k for k in use_Ks)
sxy = sum(k*f for k, f in zip(use_Ks, fvals))
A0 = (n * sxy - sx * sy) / (n * sxx - sx**2)
A1 = (sy - A0 * sx) / n
pr(f"  A₀ = {A0:.6f}, A₁ = {A1:.4f}")
pr(f"  Model: c₁(K) ≈ π/18 + ({A0:.4f}·K + {A1:.2f})·(1/2)^K")

pr("\n  Residuals:")
for K in use_Ks:
    c1 = DATA[K]['c1']
    model = TARGET_C1 + (A0 * K + A1) * mpf(1) / mpf(2)**K
    res = float(c1 - model)
    pr(f"    K={K}: residual = {res:+.4e}")

# ══════════════════════════════════════════════════════════════════
# 3. Σ_even CONVERGENCE (should be pure geometric, no K prefactor)
# ══════════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("3. Σ_even CONVERGENCE: Se(K) vs C₁E = 1+3ln(3/4)")
pr("=" * 72)

pr(f"{'K':>3} {'Se(K)':>22} {'Δ(C₁E)':>14} {'digits':>6} {'Δ·2^K':>10}")
for K in Ks:
    se = DATA[K]['Se']
    delta = float(se - TARGET_SE)
    digits = -mpmath.log10(abs(se - TARGET_SE)) if se != TARGET_SE else 99
    f_K = float((se - TARGET_SE) * mpf(2)**K)
    pr(f"{K:3d} {float(se):22.18f} {delta:+14.8e} {float(digits):6.2f} {f_K:10.4f}")

# ══════════════════════════════════════════════════════════════════
# 4. Σ_odd CONVERGENCE
# ══════════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("4. Σ_odd CONVERGENCE: So(K) vs π/18 - 1 + 6ln2 - 3ln3")
pr("=" * 72)

pr(f"{'K':>3} {'So(K)':>22} {'Δ(target)':>14} {'digits':>6} {'Δ·2^K':>10}")
for K in Ks:
    so = DATA[K]['So']
    delta = float(so - TARGET_SO)
    digits = -mpmath.log10(abs(so - TARGET_SO)) if so != TARGET_SO else 99
    f_K = float((so - TARGET_SO) * mpf(2)**K)
    pr(f"{K:3d} {float(so):22.18f} {delta:+14.8e} {float(digits):6.2f} {f_K:10.4f}")

# ══════════════════════════════════════════════════════════════════
# 5. RICHARDSON EXTRAPOLATION (multi-rate)
# ══════════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("5. RICHARDSON EXTRAPOLATION")
pr("=" * 72)


def richardson(Ks, vals, rate, n_terms):
    A = mpmath.matrix(n_terms, n_terms)
    b = mpmath.matrix(n_terms, 1)
    for i in range(n_terms):
        A[i, 0] = 1
        for j in range(1, n_terms):
            A[i, j] = mpf(rate) ** (j * Ks[i])
        b[i] = vals[i]
    return mpmath.lu_solve(A, b)[0]


def poly_geo_richardson(Ks_in, vals_in, rate, n_terms):
    """Richardson for c(K) = c∞ + (a₀K + a₁)·r^K + a₂·r^{2K} + ..."""
    n = n_terms
    Ks = Ks_in[-n:]
    vals = vals_in[-n:]
    A = mpmath.matrix(n, n)
    b_vec = mpmath.matrix(n, 1)
    for i in range(n):
        K = Ks[i]
        rK = mpf(rate) ** K
        A[i, 0] = 1
        A[i, 1] = K * rK
        A[i, 2] = rK
        for j in range(3, n):
            A[i, j] = mpf(rate) ** ((j - 1) * K)
        b_vec[i] = vals[i]
    return mpmath.lu_solve(A, b_vec)[0]


vals_c1 = [DATA[K]['c1'] for K in Ks]
vals_so = [DATA[K]['So'] for K in Ks]
vals_se = [DATA[K]['Se'] for K in Ks]

for label, vals, target, tgt_name in [
    ("c₁", vals_c1, TARGET_C1, "π/18"),
    ("Σ_odd", vals_so, TARGET_SO, "π/18-1+6ln2-3ln3"),
    ("Σ_even", vals_se, TARGET_SE, "1+3ln(3/4)"),
]:
    pr(f"\n  --- {label} ---")
    for n_terms in [3, 4, 5, 6]:
        if n_terms > len(Ks):
            continue
        try:
            ext = richardson(Ks, vals, mpf(1)/2, n_terms)
            delta = float(ext - target)
            dig = -mpmath.log10(abs(ext - target)) if ext != target else 99
            pr(f"    rate=1/2, {n_terms} terms: {float(ext):.18f} "
               f"Δ={delta:+.4e} ({float(dig):.1f} digits)")
        except Exception:
            pass

    for n_terms in [4, 5, 6]:
        if n_terms > len(Ks):
            continue
        try:
            ext = poly_geo_richardson(Ks, vals, mpf(1)/2, n_terms)
            delta = float(ext - target)
            dig = -mpmath.log10(abs(ext - target)) if ext != target else 99
            pr(f"    poly-geo, {n_terms} terms: {float(ext):.18f} "
               f"Δ={delta:+.4e} ({float(dig):.1f} digits)")
        except Exception as e:
            pr(f"    poly-geo, {n_terms} terms: FAILED ({e})")

# ══════════════════════════════════════════════════════════════════
# 6. EFFECTIVE CONVERGENCE RATE ρ(K)
# ══════════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("6. EFFECTIVE CONVERGENCE RATE ρ(K) = |Δ(K)/Δ(K-1)|^{1/1}")
pr("=" * 72)

pr(f"{'K':>3} {'ρ_c1':>8} {'ρ_So':>8} {'ρ_Se':>8}")
for i in range(1, len(Ks)):
    K = Ks[i]
    Kp = Ks[i-1]
    dc1 = DATA[K]['c1'] - TARGET_C1
    dc1p = DATA[Kp]['c1'] - TARGET_C1
    dso = DATA[K]['So'] - TARGET_SO
    dsop = DATA[Kp]['So'] - TARGET_SO
    dse = DATA[K]['Se'] - TARGET_SE
    dsep = DATA[Kp]['Se'] - TARGET_SE

    rc1 = float(abs(dc1 / dc1p)) if dc1p != 0 else 0
    rso = float(abs(dso / dsop)) if dsop != 0 else 0
    rse = float(abs(dse / dsep)) if dsep != 0 else 0
    pr(f"{K:3d} {rc1:8.4f} {rso:8.4f} {rse:8.4f}")

pr("\n  Note: ρ → 1/2 confirms geometric convergence at Diaconis-Fulman rate.")
pr("  K·(1/2)^K prefactor makes ρ_eff = K/(K-1) · 1/2 → 1/2⁺")

# ══════════════════════════════════════════════════════════════════
# 7. PSLQ (if enough precision)
# ══════════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("7. PSLQ ON EXTRAPOLATED VALUES")
pr("=" * 72)

try:
    ext_c1 = poly_geo_richardson(Ks, vals_c1, mpf(1)/2, 6)
    ext_so = poly_geo_richardson(Ks, vals_so, mpf(1)/2, 6)
    ext_se = poly_geo_richardson(Ks, vals_se, mpf(1)/2, 6)

    basis_names = ["val", "1", "ln2", "ln3", "π"]
    basis = lambda v: [v, mpf(1), log(2), log(3), pi]

    for label, val, target in [
        ("c₁(∞)", ext_c1, TARGET_C1),
        ("Σ_odd(∞)", ext_so, TARGET_SO),
        ("Σ_even(∞)", ext_se, TARGET_SE),
    ]:
        pr(f"\n  {label} = {float(val):.18f}  (target = {float(target):.18f})")
        bv = basis(val)
        rel = mpmath.pslq(bv, maxcoeff=200, maxsteps=20000)
        if rel is not None:
            terms = [f"{c}·{n}" for c, n in zip(rel, basis_names) if c != 0]
            pr(f"    PSLQ: {' + '.join(terms)} = 0")
            if rel[0] != 0:
                rec = -sum(mpf(rel[i]) * bv[i] for i in range(1, len(rel))) / rel[0]
                delta = float(rec - target)
                pr(f"    Recovered: {float(rec):.18f}, Δ(target) = {delta:+.4e}")
        else:
            pr(f"    PSLQ: no relation found (precision insufficient)")
except Exception as e:
    pr(f"  Richardson/PSLQ failed: {e}")

# ══════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ══════════════════════════════════════════════════════════════════
pr("\n" + "=" * 72)
pr("SUMMARY")
pr("=" * 72)
c1_19 = DATA[19]['c1']
dig_19 = float(-mpmath.log10(abs(c1_19 - TARGET_C1)))
pr(f"  K=19 direct: c₁(19) = {float(c1_19):.18f}")
pr(f"  Δ(π/18) = {float(c1_19 - TARGET_C1):+.8e} → {dig_19:.1f} digits")
pr(f"  K=20 (running): will give Nt = 4^19 ≈ 2.7×10¹¹ pairs")
pr(f"  Expected: ~{dig_19 + 0.3:.1f} digits (gain ~0.3 per K step)")
pr(f"  For PSLQ-ready precision (~8 digits): need K ≈ {19 + int((8-dig_19)/0.3)}")
