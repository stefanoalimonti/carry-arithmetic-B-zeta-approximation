#!/usr/bin/env python3
"""
B15: Rigorous proof of the Carry Representation Theorem and
         the Unit Leading Carry Theorem.

═══════════════════════════════════════════════════════════════════════
THEOREM 1 (Carry Representation Theorem, CRT).
  Let p, q be positive integers with N = pq. In base b, define:
    - g(x) = Σ g_i x^i  (digit polynomial of p)
    - h(x) = Σ h_j x^j  (digit polynomial of q)
    - f(x) = Σ f_k x^k  (digit polynomial of N)
    - C(x) = g(x)h(x) - f(x)  (carry polynomial)
    - Q(x) = C(x)/(x - b)  (quotient polynomial, well-defined since C(b)=0)

  Let carry_0 = 0, carry_{k+1} = floor((conv_k + carry_k) / b), where
  conv_k = Σ_{i+j=k} g_i h_j is the convolution.

  Then: Q(x) = Σ_{k=0}^{D-1} q_k x^k  with  q_k = -carry_{k+1}.

═══════════════════════════════════════════════════════════════════════
THEOREM 2 (Unit Leading Carry, ULC, base 2).
  For b = 2 and p, q ≥ 2:
    The last nonzero carry in the sequence carry_0, carry_1, ... is always 1.
    Equivalently, Q(x) has leading coefficient -1.

═══════════════════════════════════════════════════════════════════════

Part A: Proof of C(b) = 0 (division lemma)
Part B: Proof of CRT by induction
Part C: Proof of ULC for base 2
Part D: Computational verification (thousands of examples)
Part E: Extension to general base b
"""

import sys, os, time, random, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import (random_prime, carry_poly_int, quotient_poly_int,
                          to_digits, is_prime)

random.seed(42)


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def compute_carries(p, q, base=2):
    """Compute the full carry sequence for p × q in base b."""
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    fd = to_digits(p * q, base)

    conv_len = len(gd) + len(hd) - 1
    conv = [0] * conv_len
    for i, a in enumerate(gd):
        for j, b_val in enumerate(hd):
            conv[i + j] += a * b_val

    max_len = max(conv_len, len(fd)) + 2
    carries = [0] * (max_len + 1)
    for k in range(max_len):
        conv_k = conv[k] if k < conv_len else 0
        carries[k + 1] = (conv_k + carries[k]) // base

    last_nonzero = 0
    for k in range(max_len, 0, -1):
        if carries[k] != 0:
            last_nonzero = k
            break

    return carries[:last_nonzero + 2], conv


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("B15: RIGOROUS PROOF OF CRT AND ULC")
    pr("=" * 72)

    # ═══════════════════════════════════════════════════════════════
    # PART A: PROOF OF C(b) = 0
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART A: PROOF THAT C(b) = 0 (DIVISION LEMMA)")
    pr(f"{'═' * 72}")
    pr("""
  Lemma. For any positive integer n with base-b digits d_0, ..., d_{D-1}:
    n = Σ_{k=0}^{D-1} d_k · b^k

  The digit polynomial g(x) = Σ d_k x^k satisfies g(b) = n.

  Therefore:
    C(b) = g(b)·h(b) - f(b) = p·q - N = 0.

  Since C(b) = 0, the polynomial (x - b) divides C(x), so
  Q(x) = C(x)/(x-b) is a polynomial with integer coefficients.  □

  Verification: testing C(b) = 0 for 10,000 random pairs:
""")

    n_test = 10000
    failures = 0
    for _ in range(n_test):
        bits = random.randint(4, 48)
        p = random_prime(bits)
        q = random_prime(bits)
        for base in [2, 3, 5, 10]:
            C = carry_poly_int(p, q, base)
            # Evaluate C at x = base
            val = 0
            bpow = 1
            for c in C:
                val += c * bpow
                bpow *= base
            if val != 0:
                failures += 1
                pr(f"  FAILURE: p={p}, q={q}, base={base}, C(b)={val}")

    pr(f"  Tested {n_test} pairs × 4 bases = {n_test*4} evaluations.")
    pr(f"  Failures: {failures} {'(all pass ✓)' if failures == 0 else '(!!! FAILURES !!!)'}")

    # ═══════════════════════════════════════════════════════════════
    # PART B: PROOF OF CRT
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART B: PROOF OF THE CARRY REPRESENTATION THEOREM")
    pr(f"{'═' * 72}")
    pr("""
  Theorem. Q(x) = C(x)/(x-b) has coefficients q_k = -carry_{k+1}.

  Proof. Since C(x) = (x-b)·Q(x), comparing coefficients:

    C(x) = Σ c_k x^k  where c_k = b·carry_{k+1} - carry_k
           (from the carry recurrence: conv_k + carry_k = b·carry_{k+1} + f_k
            ⟹ conv_k - f_k = b·carry_{k+1} - carry_k
            ⟹ c_k = b·carry_{k+1} - carry_k)

    (x-b)·Q(x) = x·Q(x) - b·Q(x)
    If Q(x) = Σ q_k x^k, then:
      [x^0]:  c_0 = -b·q_0
      [x^k]:  c_k = q_{k-1} - b·q_k     (1 ≤ k ≤ D-1)
      [x^D]:  c_D = q_{D-1}

    Base case (k=0):
      c_0 = -b·q_0
      c_0 = b·carry_1 - carry_0 = b·carry_1    (since carry_0 = 0)
      ⟹ -b·q_0 = b·carry_1
      ⟹ q_0 = -carry_1  ✓

    Inductive step: Assume q_{k-1} = -carry_k. Then:
      c_k = q_{k-1} - b·q_k
      b·carry_{k+1} - carry_k = (-carry_k) - b·q_k
      b·carry_{k+1} = -b·q_k
      q_k = -carry_{k+1}  ✓

  By induction, q_k = -carry_{k+1} for all k = 0, ..., D-1.  □
""")

    # Computational verification
    pr("  Verification: testing CRT for 10,000 random pairs:\n")

    crt_failures = 0
    for _ in range(n_test):
        bits = random.randint(4, 40)
        p = random_prime(bits)
        q = random_prime(bits)
        for base in [2, 3, 5]:
            C = carry_poly_int(p, q, base)
            Q = quotient_poly_int(C, base)
            carries, conv = compute_carries(p, q, base)

            for k in range(len(Q)):
                expected = -carries[k + 1] if k + 1 < len(carries) else 0
                if Q[k] != expected:
                    crt_failures += 1
                    if crt_failures <= 3:
                        pr(f"  FAILURE: p={p}, q={q}, base={base}, k={k}: "
                           f"Q[k]={Q[k]}, -carry_{{k+1}}={expected}")

    pr(f"  Tested {n_test} pairs × 3 bases, all coefficients checked.")
    pr(f"  Failures: {crt_failures} {'(all pass ✓)' if crt_failures == 0 else '(!!! FAILURES !!!)'}")

    # ═══════════════════════════════════════════════════════════════
    # PART C: PROOF OF ULC (BASE 2)
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART C: PROOF OF THE UNIT LEADING CARRY THEOREM (BASE 2)")
    pr(f"{'═' * 72}")
    pr("""
  Theorem. For b = 2 and p, q ≥ 2, the last nonzero carry equals 1.

  Proof. Let D be the highest index with carry_D > 0. We show carry_D = 1.

  Case 1: D is in the "draining region" (beyond the convolution range).
    For k ≥ d_p + d_q - 1, conv_k = 0 and carry_{k+1} = ⌊carry_k / 2⌋.
    Starting from any positive integer c, the sequence
      c, ⌊c/2⌋, ⌊c/4⌋, ...
    eventually reaches 1, then 0. The last nonzero term is always 1.

  Case 2: D is in the convolution range (D < d_p + d_q - 1).
    Then carry_{D+1} = 0, meaning:
      ⌊(conv_D + carry_D) / 2⌋ = 0
      ⟹ conv_D + carry_D < 2
      ⟹ conv_D + carry_D ∈ {0, 1}

    Since carry_D ≥ 1 (it's the last nonzero carry) and conv_D ≥ 0:
      carry_D = 1 and conv_D = 0.  ✓

  In both cases, carry_D = 1.

  Note: Case 2 occurs when conv_D = 0 at position D (the convolution
  has a zero there, and the incoming carry is exactly 1 which gets
  "absorbed" as a digit without producing further carry).

  Case 1 is the common case: the carries propagate past the convolution
  range and drain to 1 via repeated halving.  □
""")

    # Verification
    pr("  Verification: testing ULC for 20,000 random pairs:")

    ulc_fails_b2 = 0
    ulc_total = 0
    case1_count = 0
    case2_count = 0

    for _ in range(20000):
        bits = random.randint(3, 48)
        p = random_prime(bits)
        q = random_prime(bits)

        carries, conv = compute_carries(p, q, 2)

        last_nz = 0
        for k in range(len(carries) - 1, 0, -1):
            if carries[k] != 0:
                last_nz = k
                break

        if last_nz == 0:
            continue
        ulc_total += 1

        if carries[last_nz] != 1:
            ulc_fails_b2 += 1
            if ulc_fails_b2 <= 3:
                pr(f"  FAILURE: p={p}, q={q}, carry[{last_nz}]={carries[last_nz]}")
        else:
            gd = to_digits(p, 2)
            hd = to_digits(q, 2)
            conv_range = len(gd) + len(hd) - 1
            if last_nz >= conv_range:
                case1_count += 1
            else:
                case2_count += 1

    pr(f"  Tested: {ulc_total} pairs (base 2)")
    pr(f"  Failures: {ulc_fails_b2} {'(all pass ✓)' if ulc_fails_b2 == 0 else '!!!'}")
    pr(f"  Case 1 (draining): {case1_count} ({100*case1_count/max(ulc_total,1):.1f}%)")
    pr(f"  Case 2 (in-conv):  {case2_count} ({100*case2_count/max(ulc_total,1):.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # PART D: c_k = b·carry_{k+1} - carry_k VERIFICATION
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART D: CARRY DECOMPOSITION c_k = b·carry_{k+1} - carry_k")
    pr(f"{'═' * 72}")
    pr("  This is the key algebraic identity in the CRT proof.\n")

    decomp_fails = 0
    for _ in range(10000):
        bits = random.randint(4, 40)
        p = random_prime(bits)
        q = random_prime(bits)
        for base in [2, 3, 5, 7, 10]:
            C = carry_poly_int(p, q, base)
            carries, conv = compute_carries(p, q, base)

            for k in range(len(C)):
                c_k = C[k]
                carry_k = carries[k] if k < len(carries) else 0
                carry_k1 = carries[k + 1] if k + 1 < len(carries) else 0
                expected = base * carry_k1 - carry_k
                if c_k != expected:
                    decomp_fails += 1
                    if decomp_fails <= 3:
                        pr(f"  FAILURE: p={p}, q={q}, b={base}, k={k}: "
                           f"c_k={c_k}, b*c_{{k+1}}-c_k={expected}")

    pr(f"  Tested 10,000 pairs × 5 bases, all positions checked.")
    pr(f"  Failures: {decomp_fails} {'(all pass ✓)' if decomp_fails == 0 else '!!!'}")

    # ═══════════════════════════════════════════════════════════════
    # PART E: ULC FOR GENERAL BASE
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("PART E: ULC EXTENSION TO GENERAL BASE b")
    pr(f"{'═' * 72}")
    pr("""
  For general base b ≥ 2, the last nonzero carry equals:
    carry_D = leading digit of carry_K in base b

  where K is the position just past the convolution range.

  For b = 2: leading digit is always 1 (binary MSB = 1).
  For b > 2: leading digit ∈ {1, ..., b-1}.

  Specifically, in the draining region:
    carry, ⌊carry/b⌋, ⌊carry/b²⌋, ... → d_{m-1}, d_{m-2}, ..., d_0
  where d_i are base-b digits. The last nonzero is d_{m-1} (the MSB).
""")

    for base in [2, 3, 5, 10]:
        last_carry_dist = {}
        n_tested = 0
        for _ in range(5000):
            bits = random.randint(3, 32)
            p = random_prime(bits)
            q = random_prime(bits)
            carries, _ = compute_carries(p, q, base)

            last_nz = 0
            for k in range(len(carries) - 1, 0, -1):
                if carries[k] != 0:
                    last_nz = k
                    break
            if last_nz == 0:
                continue
            n_tested += 1
            v = carries[last_nz]
            last_carry_dist[v] = last_carry_dist.get(v, 0) + 1

        pr(f"\n  Base {base}: last nonzero carry distribution (N={n_tested}):")
        for v in sorted(last_carry_dist.keys()):
            pr(f"    carry_D = {v}: {last_carry_dist[v]} "
               f"({100*last_carry_dist[v]/n_tested:.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("SUMMARY: PROOF STATUS")
    pr(f"{'═' * 72}")
    pr("""
  ✓ PROVED: C(b) = 0 (division lemma)
       g(b)h(b) = pq = N = f(b), so C(b) = g(b)h(b) - f(b) = 0.

  ✓ PROVED: c_k = b·carry_{k+1} - carry_k
       Direct from the carry recurrence: conv_k + carry_k = b·carry_{k+1} + f_k.

  ✓ PROVED: CRT (q_k = -carry_{k+1})
       By induction using C(x) = (x-b)Q(x) and the carry decomposition.
       Base case: q_0 = -carry_1 from c_0 = -b·q_0 = b·carry_1.
       Step: q_k = -carry_{k+1} from c_k = q_{k-1} - b·q_k = b·carry_{k+1} - carry_k.

  ✓ PROVED: ULC for base 2 (carry_D = 1)
       Case 1 (draining): repeated ⌊·/2⌋ always reaches 1 before 0.
       Case 2 (in-conv): carry_{D+1}=0 forces conv_D+carry_D < 2, so carry_D = 1.

  ✓ PROVED: ULC for general base b (carry_D ∈ {1, ..., b-1})
       Same argument: draining gives leading base-b digit, in-conv forces
       carry_D < b with conv_D = 0. For b = 2, this gives carry_D = 1.
""")

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
