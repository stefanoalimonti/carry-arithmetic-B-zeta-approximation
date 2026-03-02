#!/usr/bin/env python3
"""Find the P(-2) = 0 case from B17 and verify whether it's a real counterexample."""

import sys, os, random, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, to_digits, carry_poly_int, quotient_poly_int, is_prime

random.seed(42)  # Same seed as B17


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
    val = 0
    for k in range(D):
        val += carries[k + 1] * ((-2) ** k)
    return val


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


# Replay exact same random sequence as B17 Part F
pr("Searching for P(-2) = 0 case (replaying B17 seed)...\n")

n_test = 50000
count = 0
for trial in range(n_test):
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
    count += 1

    if P_neg2 == 0:
        pr(f"FOUND P(-2) = 0!")
        pr(f"  Trial #{trial}, count #{count}")
        pr(f"  p = {p} ({p.bit_length()}-bit, prime={is_prime(p)})")
        pr(f"  q = {q} ({q.bit_length()}-bit, prime={is_prime(q)})")
        pr(f"  N = {p*q}")
        pr(f"  D = {D}")
        pr(f"  carries = {carries}")
        pr(f"  conv = {conv}")
        pr(f"  p binary: {bin(p)}")
        pr(f"  q binary: {bin(q)}")

        # Build companion matrix and check eigenvalues
        C = carry_poly_int(p, q, 2)
        Q = quotient_poly_int(C, 2)
        pr(f"  Q = {Q}")
        pr(f"  Q(-2) = {sum(qk * ((-2)**k) for k, qk in enumerate(Q))}")

        if len(Q) >= 3:
            lead = float(Q[-1])
            n = len(Q) - 1
            M = np.zeros((n, n))
            for i in range(n - 1):
                M[i + 1, i] = 1.0
            for i in range(n):
                M[i, n - 1] = -float(Q[i]) / lead
            ev = np.linalg.eigvals(M)
            r_max = max(abs(e) for e in ev)
            pr(f"  r_max = {r_max:.10f}")
            pr(f"  eigenvalues: {sorted(ev, key=lambda x: -abs(x))[:5]}")

            # Check if -2 is an eigenvalue
            min_dist_to_neg2 = min(abs(e - (-2)) for e in ev)
            pr(f"  min|λ - (-2)| = {min_dist_to_neg2:.10e}")

        pr()

# Also search exhaustively for small primes
pr("\nExhaustive search: all pairs of primes p,q with 3 ≤ p,q ≤ 1000...")
found = 0
for p in range(3, 1001, 2):
    if not is_prime(p):
        continue
    for q in range(3, 1001, 2):
        if not is_prime(q) or p == q:
            continue
        carries, conv = compute_carries(p, q, 2)
        D = len(carries) - 1
        if D < 3:
            continue
        P_neg2 = eval_P_at_neg2(carries, D)
        if P_neg2 == 0:
            found += 1
            pr(f"  P(-2)=0: p={p}, q={q}, N={p*q}, D={D}")
            pr(f"    carries={carries}")

            C = carry_poly_int(p, q, 2)
            Q = quotient_poly_int(C, 2)
            if len(Q) >= 3:
                lead = float(Q[-1])
                n = len(Q) - 1
                M = np.zeros((n, n))
                for i in range(n - 1):
                    M[i + 1, i] = 1.0
                for i in range(n):
                    M[i, n - 1] = -float(Q[i]) / lead
                ev = np.linalg.eigvals(M)
                r_max = max(abs(e) for e in ev)
                min_dist = min(abs(e - (-2)) for e in ev)
                pr(f"    r_max = {r_max:.10f}, min|λ-(-2)| = {min_dist:.6e}")

pr(f"\nExhaustive search found {found} cases with P(-2)=0 among primes ≤ 1000.")
