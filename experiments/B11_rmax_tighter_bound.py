#!/usr/bin/env python3
"""
B11: Tighter bound on r_max — does worst_rmax → 2 as D → ∞?

Key question: the E-K bound gives |λ| ≤ 3, we observe r_max < 2 always.
Is the gap (2 - r_max) bounded away from zero, or does it shrink to 0?

Strategy:
  A. Scaling: worst_rmax(D) across many D values, large samples
  B. Adversarial construction: maximize r_max over valid carry sequences
  C. P(-2) structure: quantitative gap for the characteristic polynomial at z=-2
  D. The quotient polynomial Q(x) = C(x)/(x-2): what bounds its roots?
  E. Tight Fujiwara-type bound exploiting carry_D = 1
"""

import sys, os, time, random, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, to_digits

random.seed(42)
np.random.seed(42)


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def extract_carry_info(p, q, base=2):
    """Extract carry sequence, eigenvalues, and diagnostic quantities."""
    N = p * q
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    fd = to_digits(N, base)

    conv = [0] * (len(gd) + len(hd) - 1)
    for i, a in enumerate(gd):
        for j, b_val in enumerate(hd):
            conv[i + j] += a * b_val

    D_max = max(len(conv), len(fd))
    carries = [0] * (D_max + 2)
    for k in range(D_max):
        conv_k = conv[k] if k < len(conv) else 0
        carries[k + 1] = (conv_k + carries[k]) // base

    D_carry = 0
    for j in range(len(carries) - 1, 0, -1):
        if carries[j] != 0:
            D_carry = j
            break
    if D_carry < 3:
        return None

    carry_seq = carries[1:D_carry + 1]
    D = len(carry_seq)
    if D < 3 or carry_seq[-1] != 1:
        return None

    # Build companion matrix
    M = np.zeros((D, D), dtype=np.float64)
    for i in range(D - 1):
        M[i + 1, i] = 1.0
    for i in range(D):
        M[i, D - 1] = -float(carry_seq[i])

    try:
        ev = np.linalg.eigvals(M)
        if not np.all(np.isfinite(ev)):
            return None
    except Exception:
        return None

    r_max = float(np.max(np.abs(ev)))

    # Evaluate characteristic poly at z = -2
    # p(z) = z^D + carry_seq[D-1]*z^{D-1} + ... + carry_seq[0]
    p_neg2 = 0
    z = -2
    for k in range(D):
        p_neg2 += carry_seq[k] * (z ** k)
    p_neg2 += z ** D

    return {
        'carries': carry_seq,
        'D': D,
        'ev': ev,
        'r_max': r_max,
        'conv': conv[:D_carry + 1],
        'p_neg2': p_neg2,
        'p': p, 'q': q,
    }


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("B11: TIGHTER BOUND ON r_max")
    pr("=" * 72)

    # ═══════════════════════════════════════════════════════════════
    # PART A: SCALING — worst_rmax(D) across many bit sizes
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: SCALING OF WORST r_max WITH POLYNOMIAL DEGREE D")
    pr(f"{'═' * 72}")

    bit_configs = [8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    n_samples = 5000
    scaling_data = []

    for bits in bit_configs:
        rmaxes = []
        Ds = []
        worst = None
        attempts = 0
        while len(rmaxes) < n_samples and attempts < n_samples * 20:
            attempts += 1
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            res = extract_carry_info(p, q)
            if res is None:
                continue
            rmaxes.append(res['r_max'])
            Ds.append(res['D'])
            if worst is None or res['r_max'] > worst['r_max']:
                worst = res

        if not rmaxes:
            continue

        mean_D = np.mean(Ds)
        worst_rmax = np.max(rmaxes)
        p99 = np.percentile(rmaxes, 99)
        p999 = np.percentile(rmaxes, 99.9)
        gap = 2.0 - worst_rmax

        scaling_data.append({
            'bits': bits, 'mean_D': mean_D,
            'worst_rmax': worst_rmax, 'p99': p99, 'p999': p999,
            'gap': gap, 'mean_rmax': np.mean(rmaxes),
            'worst_carries_top5': worst['carries'][-5:],
            'worst_p_neg2': worst['p_neg2'],
        })

        pr(f"  {bits:3d}-bit  D≈{mean_D:5.0f}  "
           f"mean={np.mean(rmaxes):.4f}  "
           f"p99={p99:.4f}  "
           f"worst={worst_rmax:.6f}  "
           f"gap={gap:.6f}  "
           f"P(-2)={worst['p_neg2']}")

    pr(f"\n  Summary: gap = 2 - worst_rmax")
    pr(f"  {'bits':>5s}  {'D':>5s}  {'gap':>10s}  {'gap*D':>10s}  {'gap*sqrt(D)':>12s}")
    for s in scaling_data:
        gD = s['gap'] * s['mean_D']
        gSD = s['gap'] * math.sqrt(s['mean_D'])
        pr(f"  {s['bits']:5d}  {s['mean_D']:5.0f}  {s['gap']:10.6f}  "
           f"{gD:10.4f}  {gSD:12.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PART B: ADVERSARIAL CARRY SEQUENCES
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: ADVERSARIAL CARRY SEQUENCES")
    pr(f"{'═' * 72}")
    pr("  Constructing valid carry sequences that maximize r_max.")
    pr("  Constraint: carry_{k+1} = floor((conv_k + carry_k) / 2)")
    pr("  with carry_0 = 0, carry_D = 1, conv_k ≥ 0.\n")

    # Work backwards from carry_D = 1 to maximize large carries near top
    for D_target in [10, 20, 30, 50]:
        best_rmax = 0
        best_seq = None
        n_trials = 50000

        for _ in range(n_trials):
            # Build backwards: start from carry_D = 1
            carries = [0] * (D_target + 1)
            carries[D_target] = 1
            valid = True

            for k in range(D_target - 1, 0, -1):
                # carry_{k+1} = floor((conv_k + carry_k) / 2)
                # So conv_k + carry_k = 2*carry_{k+1} + r_k where r_k ∈ {0,1}
                c_next = carries[k + 1]
                r_k = random.randint(0, 1)
                total = 2 * c_next + r_k

                # conv_k must be non-negative, and carry_k must be non-negative
                # conv_k is bounded by min(k+1, D-k+1) approximately
                max_conv = min(k + 1, D_target - k + 1)
                max_conv = min(max_conv, total)  # conv can't exceed total

                if total < 0:
                    valid = False
                    break

                # Choose conv to maximize carry_k (adversarial)
                # carry_k = total - conv_k, maximized when conv_k is minimized
                conv_k = random.randint(0, max_conv)
                carries[k] = total - conv_k

                if carries[k] < 0:
                    valid = False
                    break

            if not valid or carries[0] != 0:
                continue

            carry_seq = carries[1:]
            if carry_seq[-1] != 1:
                continue

            D = len(carry_seq)
            M = np.zeros((D, D))
            for i in range(D - 1):
                M[i + 1, i] = 1.0
            for i in range(D):
                M[i, D - 1] = -float(carry_seq[i])

            try:
                ev = np.linalg.eigvals(M)
                rm = float(np.max(np.abs(ev)))
            except Exception:
                continue

            if rm > best_rmax:
                best_rmax = rm
                best_seq = carry_seq[:]

        pr(f"  D={D_target:3d}: best r_max = {best_rmax:.6f}  "
           f"gap = {2 - best_rmax:.6f}"
           f"  top carries = {best_seq[-5:] if best_seq else 'N/A'}")

    # ═══════════════════════════════════════════════════════════════
    # PART C: P(-2) STRUCTURE
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: P(-2) QUANTITATIVE ANALYSIS")
    pr(f"{'═' * 72}")
    pr("  The characteristic polynomial p(z) = z^D + c_{D-1}z^{D-1} + ... + c_0")
    pr("  We know C(-2) ≡ 0 mod 8 for the CARRY polynomial.")
    pr("  What about p(-2) for the COMPANION characteristic polynomial?\n")

    for bits in [16, 24, 32]:
        p_neg2_vals = []
        rmax_vals = []
        count = 0
        attempts = 0
        while count < 5000 and attempts < 50000:
            attempts += 1
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            res = extract_carry_info(p, q)
            if res is None:
                continue
            p_neg2_vals.append(res['p_neg2'])
            rmax_vals.append(res['r_max'])
            count += 1

        abs_vals = [abs(v) for v in p_neg2_vals]
        n_zero = sum(1 for v in p_neg2_vals if v == 0)
        min_abs = min(abs_vals)
        sorted_abs = sorted(abs_vals)

        pr(f"  {bits}-bit ({count} semiprimes):")
        pr(f"    |p(-2)| = 0:  {n_zero} cases")
        pr(f"    min |p(-2)|:  {min_abs}")
        pr(f"    5 smallest:   {sorted_abs[:5]}")
        pr(f"    median:       {np.median(abs_vals):.1f}")

        # Correlation between small |p(-2)| and large r_max
        pairs = sorted(zip(rmax_vals, p_neg2_vals), reverse=True)[:20]
        pr(f"    Top 20 r_max vs p(-2):")
        for rm, pv in pairs[:10]:
            pr(f"      r_max={rm:.6f}  p(-2)={pv}")

    # ═══════════════════════════════════════════════════════════════
    # PART D: THE QUOTIENT POLYNOMIAL Q(x) ROOTS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: QUOTIENT POLYNOMIAL Q(x) = C(x)/(x-2)")
    pr(f"{'═' * 72}")
    pr("  Q(x) is the polynomial whose roots are the carry eigenvalues.")
    pr("  How does its structure relate to r_max?\n")

    for bits in [16, 32]:
        count = 0
        attempts = 0
        q_rmax_data = []
        while count < 3000 and attempts < 30000:
            attempts += 1
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue

            N = p * q
            gd = to_digits(p, 2)
            hd = to_digits(q, 2)
            fd = to_digits(N, 2)

            gh = [0] * (len(gd) + len(hd) - 1)
            for i, a in enumerate(gd):
                for j, b_val in enumerate(hd):
                    gh[i + j] += a * b_val
            mx = max(len(gh), len(fd))
            c_coeffs = []
            for i in range(mx):
                gi = gh[i] if i < len(gh) else 0
                fi = fd[i] if i < len(fd) else 0
                c_coeffs.append(gi - fi)
            while len(c_coeffs) > 1 and c_coeffs[-1] == 0:
                c_coeffs.pop()

            # Synthetic division by (x - 2)
            n_c = len(c_coeffs)
            if n_c <= 2:
                continue
            q_coeffs = [0] * (n_c - 1)
            q_coeffs[-1] = c_coeffs[-1]
            for i in range(n_c - 2, 0, -1):
                q_coeffs[i - 1] = c_coeffs[i] + 2 * q_coeffs[i]

            # Compute roots of Q(x) using numpy
            # q_coeffs is constant-first, numpy wants highest-degree-first
            np_coeffs = list(reversed(q_coeffs))
            if len(np_coeffs) < 2:
                continue
            try:
                roots = np.roots(np_coeffs)
                rm = float(np.max(np.abs(roots)))
            except Exception:
                continue

            if not np.isfinite(rm):
                continue

            # Evaluate Q(-2)
            q_neg2 = sum(q_coeffs[k] * ((-2) ** k) for k in range(len(q_coeffs)))

            q_rmax_data.append({
                'r_max': rm,
                'D': len(q_coeffs),
                'q_neg2': q_neg2,
                'q_coeffs_top': q_coeffs[-5:],
                'q_coeffs_bot': q_coeffs[:3],
            })
            count += 1

        if not q_rmax_data:
            continue

        rmaxes = [d['r_max'] for d in q_rmax_data]
        q_neg2s = [abs(d['q_neg2']) for d in q_rmax_data]

        pr(f"  {bits}-bit ({count} semiprimes):")
        pr(f"    r_max: mean={np.mean(rmaxes):.4f}  "
           f"max={np.max(rmaxes):.6f}  "
           f"gap={2-np.max(rmaxes):.6f}")

        worst = sorted(q_rmax_data, key=lambda d: d['r_max'], reverse=True)[:5]
        for i, w in enumerate(worst):
            pr(f"    Worst #{i+1}: r_max={w['r_max']:.6f}  D={w['D']}  "
               f"Q(-2)={w['q_neg2']}  top={w['q_coeffs_top']}")

    # ═══════════════════════════════════════════════════════════════
    # PART E: FUJIWARA-TYPE BOUND WITH carry_D = 1
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: FUJIWARA BOUND FOR CARRY POLYNOMIALS")
    pr(f"{'═' * 72}")
    pr("""  Fujiwara's bound: |z| ≤ 2 * max(|a_{n-1}/a_n|, |a_{n-2}/a_n|^{1/2},
                               ..., |a_0/a_n|^{1/n})
  Since a_n = 1 (monic) and a_{n-1} = carry_D = 1:
  The Fujiwara bound gives max(2*1, 2*sqrt(carry_{D-1}), ..., 2*carry_1^{1/D})""")

    for bits in [16, 24, 32]:
        fujiwara_bounds = []
        rmaxes = []
        count = 0
        attempts = 0
        while count < 3000 and attempts < 30000:
            attempts += 1
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            res = extract_carry_info(p, q)
            if res is None:
                continue

            D = res['D']
            cs = res['carries']
            # Fujiwara: 2 * max_k (|a_{D-1-k}|)^{1/(k+1)}
            # a_{D-1} = cs[-1] = 1 (already monic), a_{D-2} = cs[-2], etc.
            # The polynomial is z^D + cs[-1]*z^{D-1} + ... + cs[0]
            # So a_{n-1-k} = cs[D-1-k] for k=0..D-1
            max_term = 0
            for k in range(D):
                coeff = cs[D - 1 - k] if (D - 1 - k) >= 0 else 0
                if coeff > 0:
                    term = 2.0 * (coeff ** (1.0 / (k + 1)))
                    max_term = max(max_term, term)

            fujiwara_bounds.append(max_term)
            rmaxes.append(res['r_max'])
            count += 1

        pr(f"\n  {bits}-bit ({count} semiprimes):")
        pr(f"    Fujiwara bound: mean={np.mean(fujiwara_bounds):.4f}  "
           f"max={np.max(fujiwara_bounds):.4f}  "
           f"min={np.min(fujiwara_bounds):.4f}")
        pr(f"    Actual r_max:   mean={np.mean(rmaxes):.4f}  "
           f"max={np.max(rmaxes):.6f}")
        pr(f"    Tightness (bound/r_max): "
           f"mean={np.mean(np.array(fujiwara_bounds)/np.array(rmaxes)):.4f}")

        # Does Fujiwara ever give < 2?
        below_2 = sum(1 for b in fujiwara_bounds if b < 2.0)
        pr(f"    Fujiwara < 2: {below_2}/{count} ({100*below_2/count:.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # PART F: THE CRITICAL OBSERVATION — NEGATIVE REAL AXIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART F: WHERE IS r_max ACHIEVED? (angle analysis)")
    pr(f"{'═' * 72}")

    for bits in [16, 32, 64]:
        angles = []
        rmaxes = []
        count = 0
        attempts = 0
        while count < 3000 and attempts < 30000:
            attempts += 1
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            res = extract_carry_info(p, q)
            if res is None:
                continue

            idx = np.argmax(np.abs(res['ev']))
            max_ev = res['ev'][idx]
            angle = np.angle(max_ev) / np.pi  # in units of pi
            angles.append(angle)
            rmaxes.append(res['r_max'])
            count += 1

        angles = np.array(angles)
        pr(f"\n  {bits}-bit ({count} semiprimes):")
        n_real_neg = sum(1 for a in angles if abs(abs(a) - 1.0) < 0.01)
        n_real_pos = sum(1 for a in angles if abs(a) < 0.01)
        n_complex = count - n_real_neg - n_real_pos
        pr(f"    r_max at negative real (|angle|≈π): {n_real_neg} ({100*n_real_neg/count:.1f}%)")
        pr(f"    r_max at positive real (angle≈0):   {n_real_pos} ({100*n_real_pos/count:.1f}%)")
        pr(f"    r_max at complex angle:             {n_complex} ({100*n_complex/count:.1f}%)")

        # For cases with r_max > 1.3, where is it?
        high_rm = [(a, r) for a, r in zip(angles, rmaxes) if r > 1.3]
        if high_rm:
            n_neg_high = sum(1 for a, r in high_rm if abs(abs(a) - 1.0) < 0.01)
            pr(f"    Among r_max > 1.3 ({len(high_rm)} cases): "
               f"{n_neg_high} ({100*n_neg_high/len(high_rm):.0f}%) on negative real axis")

    # ═══════════════════════════════════════════════════════════════
    # PART G: CARRY RECURSION — BACKWARD PROPAGATION BOUNDS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART G: BACKWARD PROPAGATION FROM carry_D = 1")
    pr(f"{'═' * 72}")
    pr("""  From carry_D = 1, propagate backwards using:
    carry_k = 2*carry_{k+1} + r_k - conv_k  (r_k = digit of N)
  The question: how fast can carry_{D-k} grow going downward?
  The growth rate vs 2^k determines the Rouché sum behavior.\n""")

    for bits in [16, 24, 32]:
        count = 0
        attempts = 0
        # For each distance k from top, track carry_{D-k}
        max_k = 20
        carry_at_k = {k: [] for k in range(max_k)}
        conv_at_k = {k: [] for k in range(max_k)}

        while count < 5000 and attempts < 50000:
            attempts += 1
            p = random_prime(bits)
            q = random_prime(bits)
            if p == q:
                continue
            res = extract_carry_info(p, q)
            if res is None:
                continue

            D = res['D']
            cs = res['carries']
            cv = res['conv']

            for k in range(min(max_k, D)):
                carry_at_k[k].append(cs[D - 1 - k])
                if D - 1 - k < len(cv):
                    conv_at_k[k].append(cv[D - 1 - k])
            count += 1

        pr(f"\n  {bits}-bit ({count} semiprimes):")
        pr(f"  {'k':>3s}  {'mean c_{D-k}':>12s}  {'max c_{D-k}':>12s}  "
           f"{'mean/2^k':>10s}  {'max/2^k':>10s}  {'mean conv':>10s}")
        for k in range(min(max_k, 15)):
            if not carry_at_k[k]:
                continue
            mc = np.mean(carry_at_k[k])
            xc = np.max(carry_at_k[k])
            r_mean = mc / (2 ** k) if 2**k > 0 else 0
            r_max = xc / (2 ** k) if 2**k > 0 else 0
            mcv = np.mean(conv_at_k[k]) if conv_at_k[k] else 0
            pr(f"  {k:3d}  {mc:12.3f}  {xc:12.0f}  "
               f"{r_mean:10.6f}  {r_max:10.6f}  {mcv:10.3f}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS")
    pr(f"{'═' * 72}")

    if scaling_data:
        gaps = [s['gap'] for s in scaling_data]
        Ds = [s['mean_D'] for s in scaling_data]
        pr(f"\n  Gap (2 - worst_rmax) vs D:")
        pr(f"  If gap ∝ 1/D^α, then log(gap) ∝ -α*log(D)")
        if len(Ds) >= 3:
            log_D = np.log(np.array(Ds, dtype=float))
            log_gap = np.log(np.array(gaps, dtype=float))
            # Linear fit in log-log
            valid = np.isfinite(log_gap) & np.isfinite(log_D)
            if np.sum(valid) >= 2:
                coeffs = np.polyfit(log_D[valid], log_gap[valid], 1)
                pr(f"  Fit: gap ∝ D^({coeffs[0]:.3f})")
                pr(f"  If exponent > -1: gap stays bounded → r_max < 2 provable")
                pr(f"  If exponent = -∞: gap → 0 → conjecture may be false")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == "__main__":
    main()
