#!/usr/bin/env python3
"""
B06: Three Paths to Fix the Phantom Problem.

prior analysis proved phantoms GROW with L_max. Can we fix this?

PATH A — BASELINE COMPARISON
  Compare carry product vs pure truncated ζ Euler product.
  How many phantoms come from truncation alone vs carry noise?

PATH B — FILTERED EIGENVALUE PRODUCT
  Remove outlier eigenvalues (|λ| far from 1).
  Does filtering to the "physical" unit-circle eigenvalues reduce phantoms?

PATH C — DEPTH-BASED CLASSIFICATION
  Are true-zero minima systematically deeper than phantoms?
  Can depth filtering recover all true zeros with zero phantoms?
"""

import sys
import os
import math
import random
import time
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, carry_poly_int, quotient_poly_int, primes_up_to

BASE = 2
random.seed(42)
np.random.seed(42)
pr = lambda *a, **kw: (print(*a, **kw), sys.stdout.flush())

try:
    import mpmath
    mpmath.mp.dps = 30
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

# Fallback values from LMFDB; primary source is mpmath.zetazero()
RIEMANN_HARDCODED = [
    14.134725142, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
]


def get_riemann_zeros(n):
    if HAS_MPMATH:
        return [float(mpmath.zetazero(k).imag) for k in range(1, n + 1)]
    return RIEMANN_HARDCODED[:n]


def companion_eigvals(Q_int):
    d = len(Q_int) - 1
    if d < 1:
        return None
    M = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        M[i + 1, i] = 1.0
    lead = float(Q_int[-1])
    if abs(lead) < 1e-30:
        return None
    for i in range(d):
        M[i, d - 1] = -float(Q_int[i]) / lead
    if not np.all(np.isfinite(M)):
        return None
    try:
        ev = np.linalg.eigvals(M)
        if not np.all(np.isfinite(ev)):
            return None
        return ev
    except Exception:
        return None


def build_ensemble(half_bits, n_target):
    ops = []
    for _ in range(n_target * 5):
        if len(ops) >= n_target:
            break
        p = random_prime(half_bits)
        q = random_prime(half_bits)
        while q == p:
            q = random_prime(half_bits)
        C = carry_poly_int(p, q, BASE)
        Q = quotient_poly_int(C, BASE)
        ev = companion_eigvals(Q)
        if ev is not None:
            ops.append(ev)
    return ops


def find_local_minima(t_vals, L_vals):
    mt, mv = [], []
    for i in range(1, len(L_vals) - 1):
        if L_vals[i] < L_vals[i - 1] and L_vals[i] < L_vals[i + 1]:
            mt.append(t_vals[i])
            mv.append(L_vals[i])
    return np.array(mt), np.array(mv)


def match_zeros(pseudo_t, true_t, threshold):
    matched_true = set()
    matched_pseudo = set()
    dists = []
    for j, pz in enumerate(pseudo_t):
        best_d, best_k = float('inf'), -1
        for k, rz in enumerate(true_t):
            d = abs(pz - rz)
            if d < best_d:
                best_d, best_k = d, k
        if best_d <= threshold:
            matched_true.add(best_k)
            matched_pseudo.add(j)
            dists.append(best_d)
    return len(matched_true), len(matched_pseudo), dists


def compute_depths(t_vals, L_vals, minima_t):
    """Compute depth of each minimum relative to its neighbors."""
    depths = []
    for mt in minima_t:
        idx = np.argmin(np.abs(t_vals - mt))
        val = L_vals[idx]
        left = max(0, idx - 20)
        right = min(len(L_vals) - 1, idx + 20)
        neighbor_max = max(np.max(L_vals[left:idx]) if idx > left else val,
                          np.max(L_vals[idx + 1:right + 1]) if idx < right else val)
        depths.append(neighbor_max - val)
    return np.array(depths)


# ════════════════════════════════════════════════════════════════
# Product computations
# ════════════════════════════════════════════════════════════════

def product_carry_standard(ops, test_primes, t_vals):
    """Standard carry product with constant R(l)."""
    n_t = len(t_vals)
    n_semi = len(ops)
    alpha = math.pi ** 2 / 3.0
    s_arr = 0.5 + 1j * t_vals
    L = np.zeros(n_t)
    for l in test_primes:
        z_arr = 1.0 / l ** s_arr
        log_R = -alpha * math.log(1 - 1.0 / l)
        det_sum = np.zeros(n_t)
        for ev in ops:
            factors = 1.0 - np.outer(ev, z_arr)
            log_det = np.sum(np.log(np.abs(factors) + 1e-300), axis=0)
            det_sum += np.exp(log_det)
        L += np.log(np.maximum(det_sum / n_semi, 1e-300)) - log_R
    return L


def product_zeta_truncated(test_primes, t_vals):
    """Pure truncated ζ Euler product — no carry matrices at all."""
    n_t = len(t_vals)
    s_arr = 0.5 + 1j * t_vals
    L = np.zeros(n_t)
    for l in test_primes:
        euler_factor = 1.0 / np.abs(1.0 - l ** (-s_arr))
        L += np.log(np.maximum(euler_factor, 1e-300))
    return L


def product_carry_filtered(ops, test_primes, t_vals, radius_tol=0.3):
    """Carry product using only eigenvalues near |z|=1."""
    n_t = len(t_vals)
    alpha = math.pi ** 2 / 3.0
    s_arr = 0.5 + 1j * t_vals
    L = np.zeros(n_t)

    # Filter eigenvalues
    ops_filt = []
    total_kept = 0
    total_orig = 0
    for ev in ops:
        mods = np.abs(ev)
        mask = (mods >= 1.0 - radius_tol) & (mods <= 1.0 + radius_tol)
        filt = ev[mask]
        ops_filt.append(filt)
        total_kept += len(filt)
        total_orig += len(ev)

    frac_kept = total_kept / total_orig if total_orig > 0 else 0
    n_semi = len(ops_filt)

    for l in test_primes:
        z_arr = 1.0 / l ** s_arr
        # R(l) needs adjustment for fewer eigenvalues
        # Use empirical R: measure at s=1 from filtered ensemble
        det_sum = np.zeros(n_t)
        for ev in ops_filt:
            if len(ev) == 0:
                det_sum += 1.0
                continue
            factors = 1.0 - np.outer(ev, z_arr)
            log_det = np.sum(np.log(np.abs(factors) + 1e-300), axis=0)
            det_sum += np.exp(log_det)
        mean_det = det_sum / n_semi

        # Empirical renormalization: measure R_filt(l) at s=1
        z_real = 1.0 / float(l)
        det_sum_real = 0.0
        for ev in ops_filt:
            if len(ev) == 0:
                det_sum_real += 1.0
                continue
            det_sum_real += float(np.prod(np.abs(1.0 - ev * z_real)))
        mean_det_real = det_sum_real / n_semi
        root_count = sum(1 for ev_set in ops for ev_val in ev_set
                         if abs(abs(ev_val) - 1) < 0.5)
        # Simple normalization: divide mean_det by its value at a reference t
        # Actually, use the same R(l) formula but adjusted
        log_R = -alpha * frac_kept * math.log(1 - 1.0 / l)

        L += np.log(np.maximum(mean_det, 1e-300)) - log_R

    return L, frac_kept


def product_carry_phase_corrected(ops, test_primes, t_vals):
    """Carry product with per-(l,t) empirical phase correction.
    Measure R(l,t) from the ensemble, then smooth and apply.
    """
    n_t = len(t_vals)
    n_semi = len(ops)
    s_arr = 0.5 + 1j * t_vals
    L = np.zeros(n_t)

    for l in test_primes:
        z_arr = 1.0 / l ** s_arr

        # Compute <|det|> for all t
        det_sum = np.zeros(n_t)
        for ev in ops:
            factors = 1.0 - np.outer(ev, z_arr)
            log_det = np.sum(np.log(np.abs(factors) + 1e-300), axis=0)
            det_sum += np.exp(log_det)
        mean_det = det_sum / n_semi

        # Compute exact ζ factor
        zeta_factor = 1.0 / np.abs(1.0 - l ** (-s_arr))

        # R(l,t) = <|det|> / |1-l^{-s}|^{-1}
        R_lt = mean_det / (zeta_factor + 1e-300)

        # Smooth R(l,t) with a rolling window to reduce noise
        window = min(51, n_t // 10)
        if window % 2 == 0:
            window += 1
        R_smooth = np.convolve(R_lt, np.ones(window) / window, mode='same')

        # Corrected product: divide by smoothed R(l,t) instead of constant R(l)
        L += np.log(np.maximum(mean_det / (R_smooth + 1e-300), 1e-300))

    return L


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    pr("=" * 72)
    pr("B06: THREE PATHS TO FIX THE PHANTOM PROBLEM")
    pr("=" * 72)

    riemann_t = np.array(get_riemann_zeros(10))
    T_MIN, T_MAX = 10.0, 52.0
    rz = riemann_t[(riemann_t >= T_MIN) & (riemann_t <= T_MAX)]
    n_true = len(rz)
    pr(f"  {n_true} Riemann zeros in [{T_MIN}, {T_MAX}]")

    t_vals = np.linspace(T_MIN, T_MAX, 4000)
    pr(f"  Resolution: Δt = {t_vals[1] - t_vals[0]:.4f}")

    HALF_BITS = 16
    N_SEMI = 300
    pr(f"\nBuilding ensemble: {HALF_BITS}-bit, {N_SEMI} semiprimes...")
    ops = build_ensemble(HALF_BITS, N_SEMI)
    pr(f"  {len(ops)} operators built")

    L_MAX_CONFIGS = [
        (200, "~200"),
        (500, "~500"),
        (1000, "~1k"),
        (2000, "~2k"),
        (5000, "~5k"),
    ]

    all_primes = [l for l in primes_up_to(5000) if l > 2]

    summary = []

    for prime_limit, label in L_MAX_CONFIGS:
        test_primes = [l for l in all_primes if l <= prime_limit]
        n_primes = len(test_primes)

        pr(f"\n{'═' * 72}")
        pr(f"  L_max = {n_primes} primes (≤ {prime_limit})")
        pr(f"{'═' * 72}")

        methods = {}

        # ── Method 1: Standard carry product ──
        t0 = time.time()
        L_std = product_carry_standard(ops, test_primes, t_vals)
        el = time.time() - t0
        pt, pv = find_local_minima(t_vals, L_std)
        n_mt, n_mp, dists = match_zeros(pt, rz, 0.3)
        methods['carry_std'] = {
            'L': L_std, 'pseudo_t': pt, 'pseudo_v': pv,
            'n_pseudo': len(pt), 'matched': n_mt,
            'phantoms': len(pt) - n_mp, 'time': el,
            'med_d': np.median(dists) if dists else float('inf'),
        }
        pr(f"\n  [1] STANDARD CARRY PRODUCT ({el:.1f}s)")
        pr(f"      pseudo={len(pt)}  matched={n_mt}/{n_true}  "
           f"phantoms={len(pt) - n_mp}  medΔ={methods['carry_std']['med_d']:.4f}")

        # ── Method 2: Pure truncated ζ product ──
        t0 = time.time()
        L_zeta = product_zeta_truncated(test_primes, t_vals)
        el = time.time() - t0
        pt, pv = find_local_minima(t_vals, L_zeta)
        n_mt, n_mp, dists = match_zeros(pt, rz, 0.3)
        methods['zeta_trunc'] = {
            'L': L_zeta, 'pseudo_t': pt, 'pseudo_v': pv,
            'n_pseudo': len(pt), 'matched': n_mt,
            'phantoms': len(pt) - n_mp, 'time': el,
            'med_d': np.median(dists) if dists else float('inf'),
        }
        pr(f"\n  [2] PURE ζ TRUNCATED PRODUCT ({el:.1f}s)")
        pr(f"      pseudo={len(pt)}  matched={n_mt}/{n_true}  "
           f"phantoms={len(pt) - n_mp}  medΔ={methods['zeta_trunc']['med_d']:.4f}")

        # ── Method 3: Filtered eigenvalue product ──
        for tol in [0.2, 0.1, 0.05]:
            t0 = time.time()
            L_filt, frac = product_carry_filtered(ops, test_primes, t_vals, radius_tol=tol)
            el = time.time() - t0
            pt, pv = find_local_minima(t_vals, L_filt)
            n_mt, n_mp, dists = match_zeros(pt, rz, 0.3)
            key = f'filtered_{tol}'
            methods[key] = {
                'L': L_filt, 'pseudo_t': pt, 'pseudo_v': pv,
                'n_pseudo': len(pt), 'matched': n_mt,
                'phantoms': len(pt) - n_mp, 'time': el,
                'frac_kept': frac,
                'med_d': np.median(dists) if dists else float('inf'),
            }
            pr(f"\n  [3] FILTERED |λ-1|<{tol} ({frac * 100:.0f}% kept, {el:.1f}s)")
            pr(f"      pseudo={len(pt)}  matched={n_mt}/{n_true}  "
               f"phantoms={len(pt) - n_mp}  medΔ={methods[key]['med_d']:.4f}")

        # ── Method 4: Phase-corrected carry product ──
        t0 = time.time()
        L_phase = product_carry_phase_corrected(ops, test_primes, t_vals)
        el = time.time() - t0
        pt, pv = find_local_minima(t_vals, L_phase)
        n_mt, n_mp, dists = match_zeros(pt, rz, 0.3)
        methods['phase_corr'] = {
            'L': L_phase, 'pseudo_t': pt, 'pseudo_v': pv,
            'n_pseudo': len(pt), 'matched': n_mt,
            'phantoms': len(pt) - n_mp, 'time': el,
            'med_d': np.median(dists) if dists else float('inf'),
        }
        pr(f"\n  [4] PHASE-CORRECTED CARRY ({el:.1f}s)")
        pr(f"      pseudo={len(pt)}  matched={n_mt}/{n_true}  "
           f"phantoms={len(pt) - n_mp}  medΔ={methods['phase_corr']['med_d']:.4f}")

        # ── Path C: Depth analysis on standard product ──
        pr(f"\n  [C] DEPTH ANALYSIS (standard carry product)")
        std = methods['carry_std']
        if len(std['pseudo_t']) > 0:
            depths = compute_depths(t_vals, std['L'], std['pseudo_t'])
            n_mt_all, n_mp_all, _ = match_zeros(std['pseudo_t'], rz, 0.3)
            is_true = np.zeros(len(std['pseudo_t']), dtype=bool)
            for j, pz in enumerate(std['pseudo_t']):
                for rz_val in rz:
                    if abs(pz - rz_val) <= 0.3:
                        is_true[j] = True
                        break

            true_depths = depths[is_true]
            phantom_depths = depths[~is_true]

            if len(true_depths) > 0 and len(phantom_depths) > 0:
                pr(f"      True zero depth:    mean={np.mean(true_depths):.4f}  "
                   f"min={np.min(true_depths):.4f}")
                pr(f"      Phantom depth:      mean={np.mean(phantom_depths):.4f}  "
                   f"max={np.max(phantom_depths):.4f}")
                ratio = np.mean(true_depths) / np.mean(phantom_depths) if np.mean(phantom_depths) > 0 else float('inf')
                pr(f"      Depth ratio (true/phantom): {ratio:.2f}x")

                # Test depth-based filtering at various thresholds
                pr(f"      Depth-based filtering:")
                for pct in [90, 80, 70, 50, 30]:
                    threshold_depth = np.percentile(depths, 100 - pct)
                    kept = depths >= threshold_depth
                    kept_t = std['pseudo_t'][kept]
                    n_mt_k, n_mp_k, _ = match_zeros(kept_t, rz, 0.3)
                    n_ph_k = len(kept_t) - n_mp_k
                    pr(f"        Top {pct}% deepest: kept={len(kept_t)}  "
                       f"match={n_mt_k}/{n_true}  phantoms={n_ph_k}")

        # Summary row
        summary.append({
            'prime_limit': prime_limit,
            'n_primes': n_primes,
            'carry_ph': methods['carry_std']['phantoms'],
            'zeta_ph': methods['zeta_trunc']['phantoms'],
            'filt02_ph': methods.get('filtered_0.2', {}).get('phantoms', -1),
            'filt01_ph': methods.get('filtered_0.1', {}).get('phantoms', -1),
            'phase_ph': methods['phase_corr']['phantoms'],
            'carry_match': methods['carry_std']['matched'],
            'zeta_match': methods['zeta_trunc']['matched'],
            'phase_match': methods['phase_corr']['matched'],
        })

    # ════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("GRAND SUMMARY")
    pr(f"{'═' * 72}")

    pr(f"\n  Phantom count by method (matched/10 in parentheses):")
    pr(f"  {'l≤':>6s}  {'carry':>10s}  {'ζ_trunc':>10s}  {'filt.2':>10s}  "
       f"{'filt.1':>10s}  {'phase':>10s}")
    pr(f"  {'─' * 58}")

    for s in summary:
        def fmt(ph, mt):
            return f"{ph:>3d} ({mt}/10)" if ph >= 0 else "   —     "
        pr(f"  {s['prime_limit']:>6d}  "
           f"{fmt(s['carry_ph'], s['carry_match'])}  "
           f"{fmt(s['zeta_ph'], s['zeta_match'])}  "
           f"{fmt(s['filt02_ph'], '?')}  "
           f"{fmt(s['filt01_ph'], '?')}  "
           f"{fmt(s['phase_ph'], s['phase_match'])}")

    pr(f"\n{'═' * 72}")
    pr("CONCLUSIONS")
    pr(f"{'═' * 72}")

    if summary:
        s = summary[-1]
        pr(f"""
  At L_max = {s['n_primes']} primes:
    Carry standard:   {s['carry_ph']} phantoms
    Pure ζ truncated:  {s['zeta_ph']} phantoms
    Filtered (|λ-1|<0.2): {s['filt02_ph']} phantoms
    Phase-corrected:  {s['phase_ph']} phantoms
""")
        if s['zeta_ph'] < s['carry_ph']:
            pr("  → Pure ζ truncated has FEWER phantoms than carry product.")
            pr("    The carry ensemble adds EXTRA oscillations beyond truncation effects.")
            extra = s['carry_ph'] - s['zeta_ph']
            pr(f"    Extra phantoms from carry: {extra}")
        elif s['zeta_ph'] > s['carry_ph']:
            pr("  → Pure ζ truncated has MORE phantoms! Carry product is actually CLEANER.")
        else:
            pr("  → Same phantom count: carry product phantoms = truncation phantoms.")

        if s['phase_ph'] < s['carry_ph']:
            pr(f"\n  → Phase correction REDUCES phantoms: {s['carry_ph']} → {s['phase_ph']}")
        elif s['phase_ph'] == 0:
            pr(f"\n  → Phase correction ELIMINATES all phantoms!")


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
