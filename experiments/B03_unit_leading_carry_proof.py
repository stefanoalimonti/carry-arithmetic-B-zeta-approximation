#!/usr/bin/env python3
"""
B03: THEOREM — carry_D = 1 always (rigorous proof + exhaustive verification).

THEOREM: For base-2 multiplication of any two positive integers p, q,
let C(x) = g(x)h(x) - f(x) where g, h, f are digit polynomials of p, q, N=pq.
Let D = deg(C). Then carry_D = 1.

PROOF (5 steps):

1. For k > D: c_k = conv_k - f_k = 0, hence conv_k = f_k.
   The carry relation gives c_k = 2·carry_{k+1} - carry_k = 0,
   so carry_k = 2·carry_{k+1} for all k > D.

2. For k > D: the digit equation f_k = (conv_k + carry_k) mod 2
   becomes f_k = (f_k + carry_k) mod 2, hence carry_k ≡ 0 (mod 2).
   All carries above D are EVEN.

3. From step 1: carry_{k+1} = carry_k / 2 (exact integer division,
   since carry_k is even). So carry_{D+1+j} = carry_{D+1} / 2^j.

4. If carry_{D+1} > 0: since carry_{D+1} is even, write carry_{D+1} = 2^m · r
   with r odd, m ≥ 1. Then carry_{D+1+m} = r (odd).
   But carry_{D+1+m} must be even (step 2, since D+1+m > D). Contradiction.
   Therefore carry_{D+1} = 0.

5. From carry_{D+1} = 0:
   - c_D = 2·carry_{D+1} - carry_D = -carry_D ≠ 0, so carry_D ≥ 1.
   - carry_{D+1} = (conv_D + carry_D) // 2 = 0, so conv_D + carry_D ≤ 1.
   - Since carry_D ≥ 1 and conv_D ≥ 0: carry_D = 1, conv_D = 0.  □

COROLLARY: The leading coefficient of Q = C/(x-2) is always -1.
The companion matrix M is the companion of a MONIC polynomial (after
dividing by -1), with all Vieta coefficients being integers.

This experiment verifies the theorem exhaustively for small cases and
statistically for large cases.
"""

import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import to_digits

BASE = 2
random.seed(42)

def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def compute_carry_at_D(p, q, base=2):
    """Compute carry_D for the multiplication p * q in given base."""
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
        total = conv_k + carries[k]
        carries[k + 1] = total // base

    c_coeffs = []
    for k in range(D_max):
        ci = (conv[k] if k < len(conv) else 0) - (fd[k] if k < len(fd) else 0)
        c_coeffs.append(ci)
    while len(c_coeffs) > 1 and c_coeffs[-1] == 0:
        c_coeffs.pop()

    D_c = len(c_coeffs)
    if D_c < 2:
        return None

    D = D_c - 1
    carry_D = carries[D]

    carry_Dplus1 = carries[D + 1] if D + 1 < len(carries) else 0
    conv_D = conv[D] if D < len(conv) else 0

    carries_above_D_even = all(
        carries[k] % 2 == 0
        for k in range(D + 1, min(D_max + 1, len(carries)))
        if carries[k] > 0
    )

    return {
        'carry_D': carry_D,
        'carry_Dplus1': carry_Dplus1,
        'conv_D': conv_D,
        'c_D': c_coeffs[D],
        'D': D,
        'D_c': D_c,
        'carries_above_even': carries_above_D_even,
    }


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("B03: THEOREM — carry_D = 1 ALWAYS")
    pr("=" * 72)

    # ════════════════════════════════════════════════════════════════
    # PART A: Exhaustive verification for small primes
    # ════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: EXHAUSTIVE VERIFICATION (all prime pairs up to 1024)")
    pr(f"{'═' * 72}")

    primes = [p for p in range(3, 1024) if is_prime(p)]
    total = 0
    violations = 0
    prop_checks = {'carry_D_is_1': 0, 'carry_Dplus1_is_0': 0,
                   'conv_D_is_0': 0, 'c_D_is_minus1': 0,
                   'carries_above_even': 0}

    for i, p in enumerate(primes):
        for q in primes[i:]:
            result = compute_carry_at_D(p, q)
            if result is None:
                continue
            total += 1
            if result['carry_D'] == 1:
                prop_checks['carry_D_is_1'] += 1
            else:
                violations += 1
            if result['carry_Dplus1'] == 0:
                prop_checks['carry_Dplus1_is_0'] += 1
            if result['conv_D'] == 0:
                prop_checks['conv_D_is_0'] += 1
            if result['c_D'] == -1:
                prop_checks['c_D_is_minus1'] += 1
            if result['carries_above_even']:
                prop_checks['carries_above_even'] += 1

    pr(f"\n  Total prime pairs tested: {total}")
    pr(f"  Violations (carry_D ≠ 1): {violations}")
    pr(f"\n  Property verification:")
    for name, count in prop_checks.items():
        status = "✓" if count == total else f"FAIL ({total - count} violations)"
        pr(f"    {name:30s}: {count}/{total}  {status}")

    # ════════════════════════════════════════════════════════════════
    # PART B: Also test NON-prime pairs (theorem applies to all integers)
    # ════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: NON-PRIME PAIRS (theorem applies to ALL positive integers)")
    pr(f"{'═' * 72}")

    total_np = 0
    violations_np = 0
    for a in range(2, 200):
        for b in range(a, 200):
            result = compute_carry_at_D(a, b)
            if result is None:
                continue
            total_np += 1
            if result['carry_D'] != 1:
                violations_np += 1
                if violations_np <= 5:
                    pr(f"  VIOLATION: {a} × {b} = {a*b}, "
                       f"carry_D = {result['carry_D']}, "
                       f"D = {result['D']}, c_D = {result['c_D']}")

    pr(f"\n  Total integer pairs [2,200] tested: {total_np}")
    pr(f"  Violations: {violations_np}")

    # ════════════════════════════════════════════════════════════════
    # PART C: Statistical verification at large bit sizes
    # ════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: STATISTICAL VERIFICATION AT LARGE BIT SIZES")
    pr(f"{'═' * 72}")

    from carry_utils import random_prime
    BIT_SIZES = [16, 32, 48, 64]
    N_SAMPLES = {16: 5000, 32: 3000, 48: 2000, 64: 1000}

    for bits in BIT_SIZES:
        n_samples = N_SAMPLES[bits]
        count = 0
        violations_b = 0
        for _ in range(n_samples * 3):
            if count >= n_samples:
                break
            p = random_prime(bits)
            q = random_prime(bits)
            result = compute_carry_at_D(p, q)
            if result is None:
                continue
            count += 1
            if result['carry_D'] != 1:
                violations_b += 1

        pr(f"  {bits:>3}-bit: {count:>5} samples, "
           f"violations = {violations_b}  "
           f"{'✓' if violations_b == 0 else 'FAIL'}")

    # ════════════════════════════════════════════════════════════════
    # PART D: The formal proof (printed)
    # ════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: FORMAL PROOF")
    pr(f"{'═' * 72}")
    pr("""
  THEOREM (Unit Leading Carry): For base-2 multiplication of positive
  integers p, q with N = pq, let C(x) = g(x)h(x) - f(x) and D = deg(C).
  Then carry_D = 1 and conv_D = 0.

  PROOF:
  Define the carry sequence: carry_0 = 0,
  carry_{k+1} = (conv_k + carry_k) // 2, where conv is the digit
  convolution and // denotes integer floor division.

  The fundamental relation: c_k = conv_k - f_k = 2·carry_{k+1} - carry_k.

  Step 1 (Carry-coefficient coupling above D):
    For k > D: c_k = 0, so carry_k = 2·carry_{k+1}.

  Step 2 (Parity constraint):
    For k > D: the digit equation f_k = (conv_k + carry_k) mod 2
    with conv_k = f_k gives carry_k ≡ 0 (mod 2).

  Step 3 (Exact halving):
    Combining Steps 1-2: carry_{k+1} = carry_k/2 (exact integer
    division). So carry_{D+1+j} = carry_{D+1}/2^j for all j ≥ 0.

  Step 4 (Vanishing above D):
    If carry_{D+1} > 0: write carry_{D+1} = 2^m · r with r odd, m ≥ 1.
    Then carry_{D+1+m} = r is odd, contradicting Step 2.
    Therefore carry_{D+1} = 0.

  Step 5 (Unit carry at D):
    From Step 4: c_D = -carry_D ≠ 0, so carry_D ≥ 1.
    Also: carry_{D+1} = (conv_D + carry_D)//2 = 0, so conv_D + carry_D ≤ 1.
    With carry_D ≥ 1 and conv_D ≥ 0: carry_D = 1, conv_D = 0.  □

  REMARK: The proof works for ANY base b ≥ 2 with modification:
  Step 4 generalizes to carry_{D+1} = 0 (carries above D divisible by
  arbitrarily high powers of b). But Step 5 gives carry_D ≤ b-1,
  not necessarily 1, for b > 2. The UNIT CARRY is specific to binary.

  COROLLARIES:
  (a) The leading coefficient of Q = C/(x-2) is exactly -1.
  (b) Q is monic (up to sign): Q(x) = -x^n + carry_{D-1}·x^{n-1} - ...
  (c) All Vieta symmetric functions σ_k = carry_{D-k-1} are INTEGERS.
  (d) tr(M) = -carry_{D-1}, an integer in {1, 2, ...}.
  (e) tr(M^k) = O(1) for fixed k (carries near endpoint are O(1)).
""")

    # ════════════════════════════════════════════════════════════════
    # PART E: Verify the proof steps individually
    # ════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: VERIFY EACH PROOF STEP")
    pr(f"{'═' * 72}")

    primes_16 = [random_prime(16) for _ in range(200)]
    step_checks = {
        'step1_carry_eq_2next': 0,
        'step2_carries_above_even': 0,
        'step4_carry_Dplus1_zero': 0,
        'step5_carry_D_one': 0,
        'step5_conv_D_zero': 0,
        'corollary_c_D_minus1': 0,
    }
    n_checked = 0

    for i in range(0, len(primes_16) - 1, 2):
        p, q = primes_16[i], primes_16[i + 1]
        N = p * q
        gd = to_digits(p, BASE)
        hd = to_digits(q, BASE)
        fd = to_digits(N, BASE)

        conv = [0] * (len(gd) + len(hd) - 1)
        for ii, a in enumerate(gd):
            for jj, b_val in enumerate(hd):
                conv[ii + jj] += a * b_val

        D_max = max(len(conv), len(fd))
        carries = [0] * (D_max + 2)
        for k in range(D_max):
            conv_k = conv[k] if k < len(conv) else 0
            carries[k + 1] = (conv_k + carries[k]) // BASE

        c_coeffs = []
        for k in range(D_max):
            c_coeffs.append(
                (conv[k] if k < len(conv) else 0) -
                (fd[k] if k < len(fd) else 0))
        while len(c_coeffs) > 1 and c_coeffs[-1] == 0:
            c_coeffs.pop()

        D = len(c_coeffs) - 1
        if D < 1:
            continue
        n_checked += 1

        s1 = all(
            carries[k] == 2 * carries[k + 1]
            for k in range(D + 1, min(D_max, len(carries) - 1))
        )
        if s1:
            step_checks['step1_carry_eq_2next'] += 1

        s2 = all(
            carries[k] % 2 == 0
            for k in range(D + 1, min(D_max + 1, len(carries)))
        )
        if s2:
            step_checks['step2_carries_above_even'] += 1

        if carries[D + 1] == 0:
            step_checks['step4_carry_Dplus1_zero'] += 1

        if carries[D] == 1:
            step_checks['step5_carry_D_one'] += 1

        conv_D = conv[D] if D < len(conv) else 0
        if conv_D == 0:
            step_checks['step5_conv_D_zero'] += 1

        if c_coeffs[D] == -1:
            step_checks['corollary_c_D_minus1'] += 1

    pr(f"\n  Checked {n_checked} semiprimes (16-bit factors):")
    for step, count in step_checks.items():
        status = "✓" if count == n_checked else f"FAIL ({n_checked - count})"
        pr(f"    {step:35s}: {count}/{n_checked}  {status}")

    pr(f"\nTotal runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)
    pr("B03 COMPLETE")


if __name__ == "__main__":
    main()
