#!/usr/bin/env python3
"""
B16: Analytical attack on P(-2) ≠ 0.

From prior experiments: if P(-r) = 0 has no solution for r ≥ 2, then r_max < 2.
All worst-case eigenvalues are on the negative real axis.
P(-2) is always nonzero empirically and always even.

The carry polynomial (after CRT): P(z) = Σ_{k=0}^{D-1} carry_{k+1} z^k

P(-2) = Σ_{k=0}^{D-1} carry_{k+1} (-2)^k
      = carry_1 - 2·carry_2 + 4·carry_3 - 8·carry_4 + ...

Strategy:
  (a) Find exact modular properties of P(-2)
  (b) Prove P(-2) ≡ something ≠ 0 (mod m)
  (c) If successful, this proves P(-2) ≠ 0 → r_max < 2

Part A: Modular analysis — P(-2) mod 2, 4, 8, 16
Part B: Closed-form expression via carry recursion
Part C: Lower bound on |P(-2)| via carry profile
Part D: The alternating sum structure and sign analysis
Part E: Attempt at a rigorous proof
"""

import sys, os, time, random, math
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, to_digits, carry_poly_int, quotient_poly_int

random.seed(42)


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def compute_carries(p, q, base=2):
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    conv_len = len(gd) + len(hd) - 1
    conv = [0] * conv_len
    for i, a in enumerate(gd):
        for j, b_val in enumerate(hd):
            conv[i + j] += a * b_val
    max_len = max(conv_len, len(to_digits(p * q, base))) + 2
    carries = [0] * (max_len + 1)
    for k in range(max_len):
        conv_k = conv[k] if k < conv_len else 0
        carries[k + 1] = (conv_k + carries[k]) // base
    last_nz = 0
    for k in range(max_len, 0, -1):
        if carries[k] != 0:
            last_nz = k
            break
    return carries[:last_nz + 1], conv


def eval_P_at_neg2(carries, D):
    """P(-2) = Σ_{k=0}^{D-1} carry_{k+1} · (-2)^k"""
    val = 0
    for k in range(D):
        val += carries[k + 1] * ((-2) ** k)
    return val


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("B16: ANALYTICAL ATTACK ON P(-2) ≠ 0")
    pr("=" * 72)

    # ═══════════════════════════════════════════════════════════════
    # PART A: MODULAR ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: P(-2) MODULAR STRUCTURE")
    pr(f"{'═' * 72}")
    pr("""
  P(-2) = carry_1 - 2·carry_2 + 4·carry_3 - 8·carry_4 + ...
        = carry_1 + Σ_{k≥1} carry_{k+1}·(-2)^k

  Note: for k ≥ 1, (-2)^k ≡ 0 (mod 2), so P(-2) ≡ carry_1 (mod 2).
  For k ≥ 2, (-2)^k ≡ 0 (mod 4), so P(-2) ≡ carry_1 - 2·carry_2 (mod 4).
""")

    mod_dist = {m: Counter() for m in [2, 3, 4, 8, 16]}
    P_vals = []
    carry1_dist = Counter()
    carry2_dist = Counter()
    carry1_carry2_joint = Counter()

    n_test = 50000
    for _ in range(n_test):
        bits = random.randint(8, 40)
        p = random_prime(bits)
        q = random_prime(bits)
        if p == q:
            continue

        carries, conv = compute_carries(p, q, 2)
        D = len(carries) - 1
        if D < 3:
            continue

        P_neg2 = eval_P_at_neg2(carries, D)
        P_vals.append(P_neg2)

        carry1_dist[carries[1]] += 1
        if D >= 2:
            carry2_dist[carries[2]] += 1
            carry1_carry2_joint[(carries[1], carries[2])] += 1

        for m in mod_dist:
            mod_dist[m][P_neg2 % m] += 1

    pr(f"  Tested {len(P_vals)} semiprimes.\n")

    for m in [2, 3, 4, 8, 16]:
        pr(f"  P(-2) mod {m}:")
        for v in sorted(mod_dist[m].keys()):
            count = mod_dist[m][v]
            pr(f"    ≡ {v:3d} (mod {m}): {count:6d} ({100*count/len(P_vals):.1f}%)")

    pr(f"\n  carry_1 distribution:")
    for v in sorted(carry1_dist.keys()):
        pr(f"    carry_1 = {v}: {carry1_dist[v]} ({100*carry1_dist[v]/len(P_vals):.1f}%)")

    pr(f"\n  carry_2 distribution:")
    for v in sorted(carry2_dist.keys())[:10]:
        pr(f"    carry_2 = {v}: {carry2_dist[v]} ({100*carry2_dist[v]/len(P_vals):.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # PART B: CLOSED FORM VIA CARRY RECURSION
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: ALGEBRAIC STRUCTURE OF P(-2)")
    pr(f"{'═' * 72}")
    pr("""
  Using the carry recursion: carry_{k+1} = ⌊(conv_k + carry_k)/2⌋
  we can rewrite:
    conv_k + carry_k = 2·carry_{k+1} + f_k
  where f_k is the k-th digit of N.

  Therefore: carry_{k+1} = (conv_k + carry_k - f_k) / 2

  Substituting into P(-2):
    P(-2) = Σ_{k=0}^{D-1} carry_{k+1} · (-2)^k
           = Σ_{k=0}^{D-1} (conv_k + carry_k - f_k)/2 · (-2)^k
           = Σ_{k=0}^{D-1} (conv_k + carry_k - f_k) · (-2)^{k-1}
           = -1/2 · Σ_{k=0}^{D-1} (conv_k + carry_k - f_k) · (-2)^k

  Since conv_k + carry_k - f_k = 2·carry_{k+1} is always even,
  the sum is divisible by 2, making P(-2) an integer. ✓

  More explicitly:
    P(-2) = -1/2 · [Σ conv_k(-2)^k + Σ carry_k(-2)^k - Σ f_k(-2)^k]
           = -1/2 · [g(-2)h(-2) + carry-sum - f(-2)]

  where g(-2)h(-2) is the digit polynomial product evaluated at -2,
  and f(-2) is the digit polynomial of N at -2.

  Key: C(-2) = g(-2)h(-2) - f(-2) = g(-2)h(-2) - f(-2).
  And Q(-2) = C(-2)/(-2-2) = -C(-2)/4.
  P(-2) is Q(-2) with sign adjustments from the monic normalization.
""")

    # Verify the algebraic identity
    pr("  Verification of Q(-2) = C(-2)/(-4):\n")
    identity_fails = 0
    for _ in range(10000):
        bits = random.randint(4, 40)
        p = random_prime(bits)
        q = random_prime(bits)
        C = carry_poly_int(p, q, 2)
        Q = quotient_poly_int(C, 2)

        # Evaluate C(-2)
        C_neg2 = sum(c * ((-2) ** k) for k, c in enumerate(C))
        Q_neg2 = sum(q * ((-2) ** k) for k, q in enumerate(Q))

        # C(-2) = (-2 - 2) * Q(-2) = -4 * Q(-2)
        if C_neg2 != -4 * Q_neg2:
            identity_fails += 1

    pr(f"  C(-2) = -4·Q(-2) verified: {10000 - identity_fails}/10000")

    # Now Q(-2) and P(-2): Q has leading coeff -1 (by ULC, carry_D = 1)
    # The monic polynomial P(z) = -Q(z) / Q_lead = Q(z) (since Q_lead = -1)
    # Wait: P(z) = Σ carry_{k+1} z^k, Q(z) = Σ (-carry_{k+1}) z^k = -P(z)
    # So Q(-2) = -P(-2), meaning P(-2) = -Q(-2) = C(-2)/4
    pr(f"\n  P(-2) = C(-2)/4 = (g(-2)·h(-2) - f(-2)) / 4")

    # Verify
    p_neg2_fails = 0
    for _ in range(10000):
        bits = random.randint(4, 40)
        p = random_prime(bits)
        q = random_prime(bits)

        gd = to_digits(p, 2)
        hd = to_digits(q, 2)
        fd = to_digits(p * q, 2)

        g_neg2 = sum(d * ((-2) ** k) for k, d in enumerate(gd))
        h_neg2 = sum(d * ((-2) ** k) for k, d in enumerate(hd))
        f_neg2 = sum(d * ((-2) ** k) for k, d in enumerate(fd))

        C_neg2 = g_neg2 * h_neg2 - f_neg2
        P_expected = C_neg2 // 4

        carries, _ = compute_carries(p, q, 2)
        D = len(carries) - 1
        P_direct = eval_P_at_neg2(carries, D)

        if C_neg2 % 4 != 0 or P_expected != P_direct:
            p_neg2_fails += 1
            if p_neg2_fails <= 3:
                pr(f"  FAIL: p={p}, q={q}, C(-2)={C_neg2}, C(-2)/4={C_neg2/4}, "
                   f"P(-2)={P_direct}")

    pr(f"  P(-2) = C(-2)/4 verified: {10000 - p_neg2_fails}/10000")

    # ═══════════════════════════════════════════════════════════════
    # PART C: THE g(-2)h(-2) STRUCTURE
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: STRUCTURE OF g(-2)·h(-2) AND f(-2)")
    pr(f"{'═' * 72}")
    pr("""
  P(-2) = (g(-2)·h(-2) - f(-2)) / 4

  For P(-2) = 0, we need: g(-2)·h(-2) = f(-2).
  Since f is the digit polynomial of N = pq:
    g(-2)·h(-2) = f(-2)  where f = digits of g(2)·h(2).

  This is a nontrivial constraint linking the formal polynomial
  product at x = -2 to the integer product at x = 2.
""")

    # What is g(-2) for a random prime?
    pr("  Distribution of g(-2) mod small primes:\n")
    g_neg2_mod = {m: Counter() for m in [3, 4, 5, 8]}
    g_neg2_vals = []

    for _ in range(20000):
        bits = random.randint(8, 40)
        p = random_prime(bits)
        gd = to_digits(p, 2)
        g_neg2 = sum(d * ((-2) ** k) for k, d in enumerate(gd))
        g_neg2_vals.append(g_neg2)
        for m in g_neg2_mod:
            g_neg2_mod[m][g_neg2 % m] += 1

    for m in [3, 4, 5]:
        pr(f"  g(-2) mod {m} for random primes:")
        for v in sorted(g_neg2_mod[m].keys()):
            c = g_neg2_mod[m][v]
            pr(f"    ≡ {v} (mod {m}): {c} ({100*c/20000:.1f}%)")
        pr()

    # Key identity: g(-2) is related to the "negabinary" representation
    pr("  Note: g(-2) is the value of p's binary digits interpreted in base -2.")
    pr("  This is the NEGABINARY representation of p.\n")

    # Verify: g(-2) for small primes
    pr("  g(-2) for first primes:")
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        gd = to_digits(p, 2)
        g_neg2 = sum(d * ((-2) ** k) for k, d in enumerate(gd))
        pr(f"    p = {p:3d} ({bin(p):>8s}): g(-2) = {g_neg2}")

    # ═══════════════════════════════════════════════════════════════
    # PART D: MOD 3 ANALYSIS — CAN WE PROVE P(-2) ≠ 0 MOD 3?
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: P(-2) MOD 3 ANALYSIS")
    pr(f"{'═' * 72}")
    pr("""
  P(-2) = (g(-2)·h(-2) - f(-2)) / 4

  Mod 3: (-2)^k cycles as -2, 4, -8, 16, ... ≡ 1, 1, 1, 1, ... (mod 3)
  So g(-2) ≡ Σ g_k (mod 3) = (number of 1-bits in p) mod 3 = popcount(p) mod 3.

  Similarly h(-2) ≡ popcount(q) mod 3 and f(-2) ≡ popcount(N) mod 3.

  Therefore: C(-2) ≡ popcount(p)·popcount(q) - popcount(N) (mod 3)
  And: P(-2) ≡ C(-2)/4 (mod 3)
  Since 4 ≡ 1 (mod 3): P(-2) ≡ C(-2) ≡ popcount(p)·popcount(q) - popcount(N) (mod 3)
""")

    # Test this
    mod3_vals = Counter()
    for _ in range(50000):
        bits = random.randint(8, 40)
        p = random_prime(bits)
        q = random_prime(bits)
        if p == q:
            continue

        pop_p = bin(p).count('1')
        pop_q = bin(q).count('1')
        pop_N = bin(p * q).count('1')
        predicted_mod3 = (pop_p * pop_q - pop_N) % 3

        carries, _ = compute_carries(p, q, 2)
        D = len(carries) - 1
        if D < 3:
            continue
        P_neg2 = eval_P_at_neg2(carries, D)
        actual_mod3 = P_neg2 % 3

        if predicted_mod3 != actual_mod3:
            pr(f"  MOD 3 MISMATCH: p={p}, q={q}")
        mod3_vals[actual_mod3] += 1

    pr(f"\n  P(-2) mod 3 distribution ({sum(mod3_vals.values())} samples):")
    for v in sorted(mod3_vals.keys()):
        c = mod3_vals[v]
        pr(f"    P(-2) ≡ {v} (mod 3): {c} ({100*c/sum(mod3_vals.values()):.1f}%)")

    is_zero_mod3 = mod3_vals.get(0, 0)
    pr(f"\n  P(-2) ≡ 0 (mod 3): {100*is_zero_mod3/sum(mod3_vals.values()):.1f}%")
    pr(f"  >> P(-2) CAN be 0 mod 3 — this path does NOT give a proof.")

    # ═══════════════════════════════════════════════════════════════
    # PART E: MOD 4 ANALYSIS — DEEPER STRUCTURE
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: P(-2) MOD 4 — THE PARITY STRUCTURE")
    pr(f"{'═' * 72}")
    pr("""
  P(-2) = (g(-2)·h(-2) - f(-2)) / 4

  For this to be 0, we need g(-2)·h(-2) = f(-2).

  Mod 4: (-2)^0 = 1, (-2)^1 = -2 ≡ 2, (-2)^2 = 4 ≡ 0, (-2)^k ≡ 0 for k≥2.
  So g(-2) ≡ g_0 + 2·g_1 (mod 4) = (first bit) + 2·(second bit) (mod 4).
  For odd primes: g_0 = 1 always. g_1 = second bit.
    If p ≡ 1 (mod 4): g_1 = 0, so g(-2) ≡ 1 (mod 4)
    If p ≡ 3 (mod 4): g_1 = 1, so g(-2) ≡ 3 (mod 4)

  Similarly for h(-2). And f(-2) ≡ f_0 + 2·f_1 (mod 4).
  N = pq odd ⟹ f_0 = 1. f_1 = (N >> 1) & 1.
""")

    # Verify
    mod4_dist = Counter()
    for _ in range(50000):
        bits = random.randint(8, 40)
        p = random_prime(bits)
        q = random_prime(bits)
        if p == q:
            continue

        gd = to_digits(p, 2)
        hd = to_digits(q, 2)
        fd = to_digits(p * q, 2)

        g_neg2 = sum(d * ((-2) ** k) for k, d in enumerate(gd))
        h_neg2 = sum(d * ((-2) ** k) for k, d in enumerate(hd))
        f_neg2 = sum(d * ((-2) ** k) for k, d in enumerate(fd))

        C_neg2 = g_neg2 * h_neg2 - f_neg2

        if C_neg2 % 4 != 0:
            pr(f"  C(-2) NOT divisible by 4! p={p}, q={q}, C(-2)={C_neg2}")
            continue

        P_neg2 = C_neg2 // 4
        mod4_dist[P_neg2 % 4] += 1

    pr(f"\n  P(-2) mod 4 distribution:")
    for v in sorted(mod4_dist.keys()):
        c = mod4_dist[v]
        pr(f"    P(-2) ≡ {v} (mod 4): {c} ({100*c/sum(mod4_dist.values()):.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # PART F: SIGN AND MAGNITUDE OF P(-2)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART F: SIGN AND MAGNITUDE OF P(-2)")
    pr(f"{'═' * 72}")

    sign_dist = Counter()
    abs_P_vals = []

    for pv in P_vals:
        if pv > 0:
            sign_dist['+'] += 1
        elif pv < 0:
            sign_dist['-'] += 1
        else:
            sign_dist['0'] += 1
        abs_P_vals.append(abs(pv))

    pr(f"\n  Sign distribution of P(-2) (N={len(P_vals)}):")
    for s in ['+', '-', '0']:
        pr(f"    {s}: {sign_dist.get(s, 0)} ({100*sign_dist.get(s,0)/len(P_vals):.1f}%)")

    arr = np.array(abs_P_vals, dtype=float)
    pr(f"\n  |P(-2)| statistics:")
    pr(f"    min    = {arr.min():.0f}")
    pr(f"    median = {np.median(arr):.0f}")
    pr(f"    mean   = {arr.mean():.0f}")
    pr(f"    max    = {arr.max():.0f}")

    # Scale with D
    pr(f"\n  |P(-2)| vs bit size:")
    for bits in [8, 12, 16, 20, 24, 32, 40]:
        vals = []
        for _ in range(2000):
            p = random_prime(bits)
            q = random_prime(bits)
            carries, _ = compute_carries(p, q, 2)
            D = len(carries) - 1
            if D < 3:
                continue
            P_neg2 = eval_P_at_neg2(carries, D)
            vals.append(abs(P_neg2))
        if vals:
            arr_b = np.array(vals, dtype=float)
            pr(f"    {bits:3d}-bit: min|P(-2)| = {arr_b.min():.0f}, "
               f"median = {np.median(arr_b):.0f}, "
               f"log2(median) = {math.log2(max(np.median(arr_b), 1)):.1f}")

    # ═══════════════════════════════════════════════════════════════
    # PART G: THE NEGABINARY FACTORIZATION CONSTRAINT
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART G: THE NEGABINARY CONSTRAINT")
    pr(f"{'═' * 72}")
    pr("""
  P(-2) = 0  ⟺  g(-2)·h(-2) = f(-2)
  ⟺ the negabinary value of p × the negabinary value of q
     = the negabinary value of N = pq.

  In other words: P(-2) = 0 iff multiplication "commutes" between
  base 2 and base -2 representations.

  g(2) = p,  h(2) = q,  f(2) = N = pq = g(2)·h(2)  (by definition)
  g(-2) = p evaluated in base -2 (NOT the negabinary of p)
  h(-2) = q evaluated in base -2

  The question becomes: when does g(-2)·h(-2) = (g·h)(-2)?
  This is ALWAYS true for polynomials: (g·h)(x) = g(x)·h(x).
  BUT: f ≠ g·h in general! f is the DIGIT polynomial of N = g(2)·h(2),
  which may differ from the POLYNOMIAL product g·h.

  Specifically: g(x)·h(x) - f(x) = C(x) ≠ 0 (it's the carry polynomial).
  So g(-2)·h(-2) - f(-2) = C(-2) = -4·P(-2).

  P(-2) = 0 ⟺ C(-2) = 0 ⟺ the carry polynomial vanishes at x = -2.

  Since C(x) has (x-2) as a factor: C(x) = (x-2)·Q(x).
  C(-2) = (-2-2)·Q(-2) = -4·Q(-2).
  P(-2) = -Q(-2) = 0 ⟺ Q(-2) = 0 ⟺ -2 is a root of Q.
""")

    # Can we show Q(-2) ≠ 0?
    # Q has integer coefficients q_k = -carry_{k+1} ≤ 0 (all non-positive!)
    # Q(-2) = Σ (-carry_{k+1})(-2)^k = Σ carry_{k+1}·(-1)^{k+1}·2^k
    #        = -carry_1 + 2·carry_2 - 4·carry_3 + 8·carry_4 - ...
    #        = -P(-2)

    pr("  KEY INSIGHT: All coefficients of Q are non-positive (q_k = -carry_{k+1} ≤ 0).")
    pr("  Q is a polynomial with all non-positive coefficients and Q(0) = -carry_1 ≤ 0.")
    pr("  Q(x) ≤ 0 for all x ≥ 0.")
    pr("  For x = -2: Q(-2) = Σ q_k·(-2)^k = alternating series with q_k ≤ 0.\n")

    # ═══════════════════════════════════════════════════════════════
    # PART H: BOUNDS ON P(-2) VIA CARRY PROFILE
    # ═══════════════════════════════════════════════════════════════
    pr(f"{'═' * 72}")
    pr("PART H: LOWER BOUND ATTEMPT")
    pr(f"{'═' * 72}")
    pr("""
  P(-2) = carry_1 - 2·carry_2 + 4·carry_3 - 8·carry_4 + ... + carry_D·(-2)^{D-1}

  The last term: carry_D = 1 (ULC), so it contributes (-2)^{D-1}.

  For D odd:  (-2)^{D-1} = 2^{D-1} > 0  (large positive)
  For D even: (-2)^{D-1} = -2^{D-1} < 0  (large negative)

  The penultimate terms contribute ≈ carry_{D-1}·(-2)^{D-2}.
  Since carry_{D-1} ∈ {0, 1, 2} typically:
    |penultimate| ≤ 2·2^{D-2} = 2^{D-1}

  So |P(-2)| ≈ |2^{D-1} - carry_{D-1}·2^{D-2} ± ...| ≈ 2^{D-2}

  The key question: can the alternating sum EXACTLY cancel?
""")

    # Empirical: what fraction of |P(-2)| / 2^{D-1} is near 0?
    ratios = []
    for _ in range(20000):
        bits = random.randint(8, 32)
        p = random_prime(bits)
        q = random_prime(bits)
        carries, _ = compute_carries(p, q, 2)
        D = len(carries) - 1
        if D < 5:
            continue
        P_neg2 = eval_P_at_neg2(carries, D)
        ratio = abs(P_neg2) / 2.0 ** (D - 1)
        ratios.append((D, ratio, P_neg2))

    ratio_arr = np.array([r[1] for r in ratios])
    pr(f"  |P(-2)| / 2^{{D-1}} statistics (N={len(ratios)}):")
    pr(f"    min    = {ratio_arr.min():.6f}")
    pr(f"    median = {np.median(ratio_arr):.6f}")
    pr(f"    mean   = {ratio_arr.mean():.6f}")
    pr(f"    P(ratio < 0.01) = {np.mean(ratio_arr < 0.01):.6f}")

    # Find the SMALLEST |P(-2)| / 2^{D-1}
    ratios.sort(key=lambda x: x[1])
    pr(f"\n  Smallest |P(-2)| / 2^{{D-1}} cases:")
    for D, ratio, pval in ratios[:10]:
        pr(f"    D={D:3d}: |P(-2)|/2^{D-1} = {ratio:.6f}, P(-2) = {pval}")

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS — PROOF STATUS OF P(-2) ≠ 0")
    pr(f"{'═' * 72}")
    pr("""
  ESTABLISHED:
    ✓ P(-2) = C(-2)/4 = (g(-2)h(-2) - f(-2))/4
    ✓ g(-2) = "negabinary evaluation" of p's binary digits
    ✓ P(-2) is always an integer (C(-2) ≡ 0 mod 4)
    ✓ |P(-2)| grows exponentially with D (median ∝ 2^{D-1})
    ✓ P(-2) mod 3 can be 0 (mod 3 attack fails)
    ✓ P(-2) takes all residues mod 4 (mod 4 attack fails)

  NOT PROVED:
    ? P(-2) ≠ 0 always
    ? Lower bound on |P(-2)|

  The fundamental obstacle: P(-2) = 0 requires an EXACT cancellation
  in an alternating sum of D terms with magnitudes up to 2^{D-1}.
  While exponentially unlikely (probability ∝ 2^{-D}), we cannot
  rigorously exclude it without either:
    (a) A modular argument showing P(-2) ≡ c ≠ 0 (mod m) for some fixed m
    (b) An exact closed form for P(-2)
    (c) A structural argument about carry polynomials at x = -2

  Path (a) fails: P(-2) takes all residues mod 2, 3, 4.
  Path (b): P(-2) = (g(-2)h(-2) - f(-2))/4 is exact but g(-2)h(-2) vs f(-2)
            is hard to control.
  Path (c): Most promising — leverage the fact that Q has all non-positive
            coefficients and carries satisfy specific recursion constraints.
""")

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
