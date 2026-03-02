#!/usr/bin/env python3
"""
Experimental verification of the (b-1)/b Carry Anti-Correlation Law.

Given a semiprime N = p * q in base b, the carry polynomial Q(x) = C(x)/(x-b)
(where C(x) = g(x)*h(x) - f(x), with g, h, f the digit polynomials of p, q, N)
satisfies:

    ρ(l) = P(factor residue in roots(Q) mod l) / P(random in roots(Q) mod l)
         = (b-1)/b + ζ(2)/(6l) + o(1/l)

The asymptotic limit is exactly (b-1)/b. The per-prime correction ζ(2)/(6l)
= π²/(36l) was identified by high-precision Rust measurements (200K trials,
c = 0.283 ± 0.008, consistent with π²/36 ≈ 0.274 at 1.1σ).

This script provides EXPLORATORY verification (suitable up to ~80 bits).
For definitive high-precision results, use carry_verify_rust/.

Three independent experiments:
  1. Precision measurement at fixed base and bit size
  2. Multi-base verification (prime and composite bases)
  3. Universality check (independence from factor structure)
"""

import math
import random
import sys
import time
from collections import defaultdict

random.seed(42)

# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------

def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True


def random_prime(bits):
    while True:
        p = random.getrandbits(bits)
        p |= (1 << (bits - 1)) | 1
        if is_prime(p):
            return p


def int_to_poly(n, base):
    """Little-endian digit polynomial: n = sum(coeffs[i] * base^i)."""
    if n == 0: return [0]
    coeffs = []
    while n > 0:
        coeffs.append(n % base)
        n //= base
    return coeffs


def poly_mul(f, g):
    if not f or not g: return [0]
    result = [0] * (len(f) + len(g) - 1)
    for i, a in enumerate(f):
        for j, b in enumerate(g):
            result[i + j] += a * b
    return result


def compute_carry_quotient(N, p, q, base):
    """Compute Q(x) = C(x)/(x - base) via synthetic division.

    C(x) = g(x)*h(x) - f(x), where g, h, f are digit polynomials of p, q, N.
    Since N = p*q, evaluating at x = base gives C(base) = 0, so (x - base) | C(x).
    """
    f = int_to_poly(N, base)
    g = int_to_poly(p, base)
    h = int_to_poly(q, base)
    gh = poly_mul(g, h)
    m = max(len(gh), len(f))
    C = [(gh[i] if i < len(gh) else 0) - (f[i] if i < len(f) else 0) for i in range(m)]
    if len(C) < 2:
        return None
    Q = [0] * (len(C) - 1)
    Q[-1] = C[-1]
    for i in range(len(C) - 2, 0, -1):
        Q[i - 1] = C[i] + base * Q[i]
    return Q


def poly_eval_mod(poly, x, mod):
    val = 0
    xp = 1
    for c in poly:
        val = (val + c * xp) % mod
        xp = (xp * x) % mod
    return val


def poly_roots_mod(poly, l):
    return frozenset(x for x in range(l) if poly_eval_mod(poly, x, l) == 0)


def measure_ratio(Q, p, q, test_primes):
    """Anti-correlation ratio: observed factor-root hits / expected by random."""
    total_hits = 0
    total_expected = 0.0
    for l in test_primes:
        roots = poly_roots_mod(Q, l)
        if not roots:
            continue
        pm, qm = p % l, q % l
        hits = int(pm in roots) + (int(qm in roots) if qm != pm else 0)
        n_distinct = 1 if pm == qm else 2
        total_hits += hits
        total_expected += len(roots) / l * n_distinct
    return total_hits / total_expected if total_expected > 0 else None


def prime_factorization(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


# ---------------------------------------------------------------------------
# Experiment 1: Precision measurement + random polynomial control
# ---------------------------------------------------------------------------

def experiment_precision(bits=35, trials=5000, base=2):
    """Measure the anti-correlation ratio at high precision and compare
    against random polynomials of the same degree distribution."""
    test_primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]

    print("=" * 75)
    print(f" EXPERIMENT 1: Precision measurement (base {base}, {bits}-bit, {trials} trials)")
    print("=" * 75)

    ratios = []
    degrees = []
    t0 = time.time()

    for trial in range(trials):
        half = bits // 2
        p = random_prime(half)
        q = random_prime(bits - half)
        Q = compute_carry_quotient(p * q, p, q, base)
        if Q is None:
            continue
        r = measure_ratio(Q, p, q, test_primes)
        if r is not None:
            ratios.append(r)
            degrees.append(len(Q))
        if (trial + 1) % 1000 == 0:
            avg = sum(ratios) / len(ratios)
            se = (sum((x - avg) ** 2 for x in ratios) / len(ratios)) ** 0.5 / len(ratios) ** 0.5
            print(f"  [{trial+1}/{trials}] ratio = {avg:.6f} +/- {se:.6f}")

    elapsed = time.time() - t0
    avg = sum(ratios) / len(ratios)
    se = (sum((x - avg) ** 2 for x in ratios) / len(ratios)) ** 0.5 / len(ratios) ** 0.5
    avg_deg = sum(degrees) / len(degrees)

    print(f"\n  Real ratio:  {avg:.6f} +/- {se:.6f}  (n={len(ratios)}, {elapsed:.1f}s)")
    print(f"  Avg degree:  {avg_deg:.1f}")
    theory = (base - 1) / base
    sigma = (avg - theory) / se if se > 0 else 0
    print(f"  (b-1)/b:     {theory:.6f}  (distance: {sigma:+.1f} sigma)")

    # Random polynomial control
    print(f"\n  Random polynomial control ({min(trials, 3000)} trials)...")
    rand_ratios = []
    for _ in range(min(trials, 3000)):
        p = random_prime(bits // 2)
        q = random_prime(bits - bits // 2)
        deg = int(avg_deg)
        C = [0] * (deg + 2)
        for i in range(1, deg + 2):
            C[i] = random.randint(-1, 1)
        C[0] = -sum(C[i] * (base ** i) for i in range(1, deg + 2))
        Q_rand = [0] * (deg + 1)
        Q_rand[-1] = C[-1]
        for i in range(len(C) - 2, 0, -1):
            Q_rand[i - 1] = C[i] + base * Q_rand[i]
        r = measure_ratio(Q_rand, p, q, test_primes)
        if r is not None:
            rand_ratios.append(r)

    avg_rand = sum(rand_ratios) / len(rand_ratios)
    se_rand = (sum((x - avg_rand) ** 2 for x in rand_ratios) / len(rand_ratios)) ** 0.5 / len(rand_ratios) ** 0.5
    z = abs(avg - avg_rand) / (se ** 2 + se_rand ** 2) ** 0.5 if (se ** 2 + se_rand ** 2) > 0 else 0

    print(f"  Random ratio: {avg_rand:.6f} +/- {se_rand:.6f}")
    print(f"  Z-score (real vs random): {z:.1f}")
    if z > 3:
        print(f"  -> SIGNIFICANT: carry structure causes anti-correlation beyond coefficient statistics")
    else:
        print(f"  -> Not significant: coefficient distribution alone explains the ratio")

    # Comparison with mathematical constants
    print(f"\n  Closest constants:")
    constants = [('1/2', 0.5), ('pi/6', math.pi / 6), ('ln(2)', math.log(2)),
                 ('(b-1)/b', theory), ('6/pi^2', 6 / math.pi ** 2)]
    for name, val in sorted(constants, key=lambda x: abs(x[1] - avg)):
        s = abs(val - avg) / se if se > 0 else float('inf')
        print(f"    {name:>10} = {val:.6f}  ({s:.1f} sigma)")

    print()
    return avg, se


# ---------------------------------------------------------------------------
# Experiment 2: Multi-base verification
# ---------------------------------------------------------------------------

def experiment_multi_base(bits=35, trials=10000):
    """Verify (b-1)/b for prime bases 2..23 and composite bases."""
    test_primes_pool = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]

    print("=" * 75)
    print(f" EXPERIMENT 2: Multi-base verification ({bits}-bit, {trials} trials/base)")
    print("=" * 75)

    def run_base(base, n_trials, semiprime_bits=None):
        if semiprime_bits is None:
            semiprime_bits = bits
        tp = [l for l in test_primes_pool if l > base and l % base != 0]
        ratios = []
        half = semiprime_bits // 2
        for _ in range(n_trials):
            p = random_prime(half)
            q = random_prime(semiprime_bits - half)
            Q = compute_carry_quotient(p * q, p, q, base)
            if Q is None:
                continue
            r = measure_ratio(Q, p, q, tp)
            if r is not None:
                ratios.append(r)
        if not ratios:
            return None, None
        avg = sum(ratios) / len(ratios)
        se = (sum((x - avg) ** 2 for x in ratios) / len(ratios)) ** 0.5 / len(ratios) ** 0.5
        return avg, se

    # Prime bases
    print(f"\n  Prime bases:")
    print(f"  {'base':>4}  {'(b-1)/b':>8}  {'measured':>8}  {'SE':>8}  {'residual':>9}  {'sigma':>6}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*6}")
    prime_bases = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    t0 = time.time()
    for b in prime_bases:
        avg, se = run_base(b, trials)
        if avg is None:
            continue
        th = (b - 1) / b
        resid = avg - th
        sig = resid / se if se > 0 else 0
        print(f"  {b:>4}  {th:>8.5f}  {avg:>8.5f}  {se:>8.5f}  {resid:>+9.5f}  {sig:>+6.1f}")
    print(f"  ({time.time() - t0:.1f}s)")

    # Composite bases
    print(f"\n  Composite bases:")
    print(f"  {'base':>4}  {'factors':>10}  {'(b-1)/b':>8}  {'Euler':>8}  {'measured':>8}  {'winner':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    composite_bases = [4, 6, 8, 9, 10, 12, 15, 16, 25]
    t0 = time.time()
    bm1_wins = 0
    for b in composite_bases:
        avg, se = run_base(b, trials)
        if avg is None:
            continue
        pf = prime_factorization(b)
        euler = 1.0
        for p_factor in pf:
            euler *= (1 - 1 / p_factor)
        th = (b - 1) / b
        fstr = "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(pf.items()))
        winner = "(b-1)/b" if abs(avg - th) < abs(avg - euler) else "Euler"
        if winner == "(b-1)/b":
            bm1_wins += 1
        print(f"  {b:>4}  {fstr:>10}  {th:>8.5f}  {euler:>8.5f}  {avg:>8.5f}  {winner:>10}")
    print(f"  ({time.time() - t0:.1f}s)")
    print(f"\n  (b-1)/b wins {bm1_wins}/{len(composite_bases)} against Euler product")

    # Scaling test
    print(f"\n  Base-2 scaling (does residual -> 0 as N -> inf?):")
    print(f"  {'bits':>5}  {'measured':>8}  {'SE':>8}  {'residual':>9}")
    for bsz in [20, 25, 30, 35, 40, 50, 60, 80]:
        tr = trials if bsz <= 40 else trials // 2
        avg, se = run_base(2, tr, semiprime_bits=bsz)
        if avg is not None:
            print(f"  {bsz:>5}  {avg:>8.5f}  {se:>8.5f}  {avg - 0.5:>+9.5f}")

    print()


# ---------------------------------------------------------------------------
# Experiment 3: Universality (independence from factor structure)
# ---------------------------------------------------------------------------

def experiment_universality(bits=30, trials=500, base=2):
    """Check that the ratio is independent of factor class, balance, and Hamming weight."""
    test_primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    print("=" * 75)
    print(f" EXPERIMENT 3: Universality check ({bits}-bit, {trials} trials, base {base})")
    print("=" * 75)

    data = []
    t0 = time.time()
    for _ in range(trials):
        p = random_prime(bits // 2)
        q = random_prime(bits - bits // 2)
        Q = compute_carry_quotient(p * q, p, q, base)
        if Q is None:
            continue
        r = measure_ratio(Q, p, q, test_primes)
        if r is None:
            continue
        data.append({
            'ratio': r,
            'p_mod4': p % 4, 'q_mod4': q % 4,
            'balance': max(p, q) / min(p, q),
            'hw': bin(p).count('1') + bin(q).count('1'),
            'N_mod8': (p * q) % 8,
        })

    elapsed = time.time() - t0
    ratios = [d['ratio'] for d in data]
    avg = sum(ratios) / len(ratios)
    std = (sum((x - avg) ** 2 for x in ratios) / len(ratios)) ** 0.5
    cv = std / avg if avg > 0 else float('inf')

    print(f"\n  Global: mean={avg:.4f}, std={std:.4f}, CV={cv:.4f} (n={len(data)}, {elapsed:.1f}s)")

    # Factor congruence class
    print(f"\n  By factor class (p mod 4, q mod 4):")
    groups = defaultdict(list)
    for d in data:
        key = (min(d['p_mod4'], d['q_mod4']), max(d['p_mod4'], d['q_mod4']))
        groups[key].append(d['ratio'])
    for key in sorted(groups):
        v = groups[key]
        g_avg = sum(v) / len(v)
        print(f"    ({key[0]},{key[1]}): {g_avg:.4f}  (n={len(v)})")

    # Balance quartiles
    print(f"\n  By factor balance (p/q ratio quartiles):")
    sorted_data = sorted(data, key=lambda x: x['balance'])
    chunk = len(sorted_data) // 4
    for i, label in enumerate(["Balanced", "Moderate", "Unbalanced", "Very unbal."]):
        subset = sorted_data[i * chunk:(i + 1) * chunk if i < 3 else len(sorted_data)]
        g_avg = sum(d['ratio'] for d in subset) / len(subset)
        print(f"    {label:>14}: {g_avg:.4f}  (n={len(subset)})")

    # N mod 8
    print(f"\n  By N mod 8:")
    m8 = defaultdict(list)
    for d in data:
        m8[d['N_mod8']].append(d['ratio'])
    for key in sorted(m8):
        v = m8[key]
        if len(v) >= 5:
            print(f"    N = {key} mod 8: {sum(v)/len(v):.4f}  (n={len(v)})")

    # Verdict
    max_dev = max(abs(sum(v) / len(v) - avg) for v in groups.values())
    print(f"\n  Max group deviation from mean: {max_dev:.4f}")
    if max_dev < std * 0.5:
        print(f"  -> UNIVERSAL: ratio is constant across all factor classes")
        print(f"  -> Anti-correlation does NOT leak factor information")
    else:
        print(f"  -> CAUTION: some group deviation detected (investigate further)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    experiment_precision(bits=bits, trials=trials, base=2)
    experiment_multi_base(bits=bits, trials=min(trials, 10000))
    experiment_universality(bits=min(bits, 30), trials=min(trials, 500), base=2)
