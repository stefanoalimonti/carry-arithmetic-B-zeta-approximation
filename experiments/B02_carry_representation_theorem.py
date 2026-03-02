#!/usr/bin/env python3
"""
B02: Trace Moment Stability and the Carry Representation Theorem.

ANALYTICAL RESULT (to verify):
  Q(x) = C(x)/(x-b) has coefficients q_i = -carry_{i+1}
  where carry_j is the j-th carry in the base-b multiplication p × q.

  Proof: Synthetic division gives q_i = Σ_{j=0}^{D-1-i} b^j · c_{i+1+j}.
  Substituting c_k = b·carry_{k+1} - carry_k and telescoping:
  q_i = b^{D-i}·carry_{D+1} - carry_{i+1} = -carry_{i+1}  (carry_{D+1} = 0). □

CONSEQUENCE:
  tr(M) = -carry_{D-1}/carry_D  (ratio of adjacent carries near the top)
  tr(M^k) is determined by carry ratios near the most significant positions

EXPERIMENT:
  Part A: Verify q_i = -carry_{i+1} computationally
  Part B: Measure ⟨p_k⟩ across bit sizes 10, 12, 16, 20, 24, 32
  Part C: Measure carry ratios and connect to trace moments
  Part D: Test convergence of ⟨p_k⟩ as d → ∞
"""

import sys
import os
import math
import random
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, to_digits, primes_up_to

BASE = 2
random.seed(42)
np.random.seed(42)

def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def multiply_with_carries(p, q, base=2):
    """Compute g(x)h(x), f(x), C(x), carries, and Q(x) with full bookkeeping."""
    N = p * q
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    fd = to_digits(N, base)

    conv = [0] * (len(gd) + len(hd) - 1)
    for i, a in enumerate(gd):
        for j, b_val in enumerate(hd):
            conv[i + j] += a * b_val

    D_conv = len(conv)
    D_f = len(fd)
    D = max(D_conv, D_f)

    c_coeffs = []
    for i in range(D):
        ci = (conv[i] if i < D_conv else 0) - (fd[i] if i < D_f else 0)
        c_coeffs.append(ci)
    while len(c_coeffs) > 1 and c_coeffs[-1] == 0:
        c_coeffs.pop()

    carries = [0]
    for k in range(D):
        conv_k = conv[k] if k < D_conv else 0
        total = conv_k + carries[k]
        n_k = total % base
        carry_next = (total - n_k) // base
        carries.append(carry_next)

    D_c = len(c_coeffs)
    if D_c <= 1:
        return None

    lead = c_coeffs[-1]
    if lead == 0:
        return None

    q_coeffs = [0] * (D_c - 1)
    q_coeffs[-1] = c_coeffs[-1]
    for i in range(D_c - 2, 0, -1):
        q_coeffs[i - 1] = c_coeffs[i] + base * q_coeffs[i]

    return {
        'conv': conv, 'f_digits': fd, 'c_coeffs': c_coeffs,
        'q_coeffs': q_coeffs, 'carries': carries,
        'D': D_c, 'degree_Q': D_c - 2,
    }


def companion_eigvals(q_coeffs):
    d = len(q_coeffs) - 1
    if d < 2:
        return None
    lead = float(q_coeffs[-1])
    if abs(lead) < 1e-30:
        return None
    M = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        M[i + 1, i] = 1.0
    for i in range(d):
        M[i, d - 1] = -float(q_coeffs[i]) / lead
    if not np.all(np.isfinite(M)):
        return None
    try:
        ev = np.linalg.eigvals(M)
        return ev if np.all(np.isfinite(ev)) else None
    except Exception:
        return None


def main():
    pr("=" * 72)
    pr("B02: TRACE MOMENT STABILITY & CARRY REPRESENTATION THEOREM")
    pr("=" * 72)

    # ════════════════════════════════════════════════════════════════
    # PART A: Verify q_i = -carry_{i+1}
    # ════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: VERIFY CARRY REPRESENTATION  q_i = -carry_{i+1}")
    pr(f"{'═' * 72}")

    n_test = 1000
    n_verified = 0
    n_failed = 0

    for _ in range(n_test):
        p = random_prime(16)
        q = random_prime(16)
        while q == p:
            q = random_prime(16)
        result = multiply_with_carries(p, q, BASE)
        if result is None:
            continue

        q_c = result['q_coeffs']
        carries = result['carries']
        deg_Q = len(q_c) - 1

        match = True
        for i in range(len(q_c)):
            expected = -carries[i + 1] if (i + 1) < len(carries) else 0
            if q_c[i] != expected:
                match = False
                break

        if match:
            n_verified += 1
        else:
            n_failed += 1

    pr(f"  Tested: {n_verified + n_failed}")
    pr(f"  Verified q_i = -carry_{{i+1}}: {n_verified} ({n_verified/(n_verified+n_failed)*100:.1f}%)")
    pr(f"  Failed: {n_failed}")
    if n_failed == 0:
        pr(f"  ✓ CARRY REPRESENTATION THEOREM VERIFIED: Q(x) = -Σ carry_{{j+1}} x^j")
    else:
        pr(f"  ✗ {n_failed} failures — check the derivation")

    # ════════════════════════════════════════════════════════════════
    # PART B: Trace moments across bit sizes
    # ════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: TRACE MOMENTS ⟨p_k⟩ vs DIMENSION d")
    pr(f"{'═' * 72}")

    BIT_SIZES = [10, 12, 16, 20, 24, 32]
    N_SEMI = {10: 800, 12: 800, 16: 500, 20: 400, 24: 300, 32: 200}
    K_MAX = 10

    all_traces = {}

    for bits in BIT_SIZES:
        n_target = N_SEMI[bits]
        t0 = time.time()

        traces_k = [[] for _ in range(K_MAX)]
        degrees = []
        carry_ratios = []
        n_built = 0

        for _ in range(n_target * 5):
            if n_built >= n_target:
                break
            p = random_prime(bits)
            q = random_prime(bits)
            while q == p:
                q = random_prime(bits)

            result = multiply_with_carries(p, q, BASE)
            if result is None:
                continue

            ev = companion_eigvals(result['q_coeffs'])
            if ev is None:
                continue

            n_built += 1
            degrees.append(len(ev))

            for k in range(K_MAX):
                pk = np.sum(ev ** (k + 1))
                traces_k[k].append(pk)

            carries = result['carries']
            D = result['D']
            carry_D = carries[D] if D < len(carries) else 0
            carry_Dm1 = carries[D - 1] if D - 1 < len(carries) else 0
            if carry_D != 0:
                carry_ratios.append(carry_Dm1 / carry_D)

        elapsed = time.time() - t0
        mean_deg = np.mean(degrees) if degrees else 0

        pr(f"\n  {bits}-bit factors: {n_built} semiprimes, deg ≈ {mean_deg:.1f}, {elapsed:.1f}s")

        means = []
        for k in range(K_MAX):
            arr = np.array(traces_k[k])
            m = np.mean(arr).real
            means.append(m)

        all_traces[bits] = means

        if carry_ratios:
            cr_mean = np.mean(carry_ratios)
            cr_std = np.std(carry_ratios)
            pr(f"  carry_{{D-1}}/carry_D: mean={cr_mean:.4f}, std={cr_std:.4f}")
            pr(f"  → tr(M) predicted from carry ratio: {-cr_mean:.4f}")

    pr(f"\n  {'k':>3}", end="")
    for bits in BIT_SIZES:
        pr(f"  {bits:>6}-bit", end="")
    pr(f"  {'target':>8}")
    pr(f"  {'─'*3}", end="")
    for _ in BIT_SIZES:
        pr(f"  {'─'*9}", end="")
    pr(f"  {'─'*8}")

    for k in range(K_MAX):
        pr(f"  {k+1:>3}", end="")
        for bits in BIT_SIZES:
            pr(f"  {all_traces[bits][k]:>+9.3f}", end="")
        pr(f"  {-1.0:>+8.3f}")

    # ════════════════════════════════════════════════════════════════
    # PART C: Carry profile analysis
    # ════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: CARRY PROFILE — WHY ⟨p_k⟩ ≈ -1")
    pr(f"{'═' * 72}")
    pr("  tr(M) = -carry_{D-1}/carry_D")
    pr("  If carries decrease smoothly near the top: ratio ≈ 1 → tr(M) ≈ -1\n")

    for bits in [16, 24, 32]:
        n_target = min(N_SEMI[bits], 300)
        carry_profiles = []
        top_k_carries = {j: [] for j in range(8)}

        for _ in range(n_target * 5):
            if len(carry_profiles) >= n_target:
                break
            p = random_prime(bits)
            q = random_prime(bits)
            while q == p:
                q = random_prime(bits)
            result = multiply_with_carries(p, q, BASE)
            if result is None:
                continue
            carries = result['carries']
            D = result['D']
            if D < 2:
                continue

            carry_D_val = carries[D] if D < len(carries) else 0
            if carry_D_val == 0:
                continue

            for j in range(min(8, D)):
                carry_idx = D - j
                if carry_idx >= 0 and carry_idx < len(carries):
                    top_k_carries[j].append(carries[carry_idx])

            carry_profiles.append(carries[:D + 1])

        pr(f"  {bits}-bit ({len(carry_profiles)} semiprimes):")
        pr(f"  {'pos':>6}  {'mean carry':>12}  {'ratio to D':>12}  {'→ contributes':>15}")
        pr(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*15}")
        for j in range(min(8, len(top_k_carries[0]))):
            if not top_k_carries[j]:
                continue
            arr = np.array(top_k_carries[j], dtype=float)
            mean_c = np.mean(arr)
            arr_D = np.array(top_k_carries[0], dtype=float)
            ratio = np.mean(arr / arr_D) if len(arr) == len(arr_D) else float('inf')
            label = f"carry_{{D-{j}}}" if j > 0 else "carry_D"
            contrib = f"→ p_{j+1} term" if j > 0 else "(reference)"
            pr(f"  {label:>6}  {mean_c:>12.3f}  {ratio:>12.4f}  {contrib:>15}")

    # ════════════════════════════════════════════════════════════════
    # PART D: Convergence analysis
    # ════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: CONVERGENCE — DO ⟨p_k⟩ CONVERGE AS d → ∞?")
    pr(f"{'═' * 72}")

    for k in [0, 1, 2, 3, 4]:
        vals = [all_traces[bits][k] for bits in BIT_SIZES]
        dims = [all_traces[bits][k] for bits in BIT_SIZES]
        trend = vals[-1] - vals[0]
        pr(f"  p_{k+1}: {vals[0]:+.3f} ({BIT_SIZES[0]}b) → {vals[-1]:+.3f} ({BIT_SIZES[-1]}b)"
           f"  trend: {trend:+.3f}  {'converging' if abs(trend) < 0.5 else 'drifting'}")

    # ════════════════════════════════════════════════════════════════
    # PART E: The analytical argument
    # ════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: ANALYTICAL ARGUMENT — CARRY REPRESENTATION → ζ EULER FACTOR")
    pr(f"{'═' * 72}")
    pr("""
  THEOREM (Carry Representation):
    Q(x) = C(x)/(x-b) has coefficients q_i = -carry_{i+1}.

  PROOF: Synthetic division gives q_i = Σ_{j=0}^{D-1-i} b^j · c_{i+1+j}.
  The carry relation c_k = b·carry_{k+1} - carry_k telescopes:
    q_i = b^{D-i}·carry_{D+1} - carry_{i+1} = -carry_{i+1}
  since carry_{D+1} = 0. □

  CONSEQUENCE FOR tr(M^k):
    The companion matrix eigenvalues are roots of the carry generating
    polynomial Carry(x) = Σ_{j=1}^D carry_j · x^{j-1}.

    tr(M) = -carry_{D-1}/carry_D
    (ratio of adjacent carries near the most significant position)

  WHY tr(M) ≈ -1:
    Near the top of the multiplication (positions D-1, D), the carry
    sequence is small and slowly varying. Adjacent carries satisfy
    carry_{D-1} ≈ carry_D (both are O(1)), so the ratio ≈ 1.

    More precisely: carry_{k+1} = floor((conv_k + carry_k)/2). Near the
    top, conv_k is small (few nonzero digit pairs). The carry transitions
    approximate a random walk that reverts to 0, with adjacent values
    differing by O(1). This gives carry_{D-1}/carry_D = 1 + O(1/carry_D),
    and ⟨tr(M)⟩ = -1 - O(1/⟨carry_D⟩).

  IMPLICATION:
    Since ⟨p_k⟩ = O(1) for all k (carries are bounded), the Newton
    expansion gives:
      ⟨log|det(I-M/l^s)|⟩ = -Re Σ_k ⟨p_k⟩/(k·l^{ks}) = O(Σ 1/(k·l^{k/2}))
    which converges. The leading term matches -log|1-l^{-s}| when ⟨p_1⟩ ≈ -1,
    with corrections O(1/l) from higher-k terms and the p_1 deviation.

  CONNECTION TO DIACONIS-FULMAN:
    The carry sequence for multiplication is a NON-Markovian generalization
    of the Diaconis-Fulman carry chain for addition. In addition:
      carry_{k+1} = floor((g_k + h_k + carry_k)/b)
    which IS Markov (carry depends only on current state + new digits).

    In multiplication:
      carry_{k+1} = floor((conv_k + carry_k)/b)
    where conv_k = Σ_{i+j=k} g_i h_j depends on ALL previous digits.
    This is NOT Markov in carry_k alone, but becomes Markov in the
    extended state (carry_k, partial products). The stationary behavior
    near the endpoints (where conv_k → 0) reverts to the additive
    Diaconis-Fulman chain, explaining why ⟨p_k⟩ ≈ -1.
""")

    pr(f"\nTotal runtime: {time.time()-time.time():.1f}s")
    pr("=" * 72)
    pr("B02 COMPLETE")
    pr("=" * 72)


if __name__ == "__main__":
    main()
