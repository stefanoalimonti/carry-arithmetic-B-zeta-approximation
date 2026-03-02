"""
Shared utilities for carry barrier experiments.

Provides exact-integer arithmetic for carry/quotient polynomials,
modular evaluation, multiplicative order computation, Dirichlet
character tables, and prime generation via Miller-Rabin.
"""

import random
import math


# ─── Primality and prime generation ───

def is_prime(n, k=20):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def random_prime(bits):
    while True:
        n = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(n):
            return n


def primes_up_to(limit):
    """Sieve of Eratosthenes returning list of primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]


# ─── Digit and polynomial utilities ───

def to_digits(n, base=2):
    """Little-endian digit decomposition: n = sum(d[i] * base^i)."""
    if n == 0: return [0]
    d = []
    while n > 0:
        d.append(int(n % base))
        n //= base
    return d


def carry_poly_int(p, q, base=2):
    """C(x) = g(x)*h(x) - f(x) as list of Python ints (constant-first).
    Uses exact integer arithmetic."""
    n = p * q
    gd = to_digits(p, base)
    hd = to_digits(q, base)
    fd = to_digits(n, base)
    gh = [0] * (len(gd) + len(hd) - 1)
    for i, a in enumerate(gd):
        for j, b in enumerate(hd):
            gh[i + j] += a * b
    mx = max(len(gh), len(fd))
    c = []
    for i in range(mx):
        gi = gh[i] if i < len(gh) else 0
        fi = fd[i] if i < len(fd) else 0
        c.append(gi - fi)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    return c


def quotient_poly_int(c_int, base=2):
    """Q(x) = C(x)/(x - base) via synthetic division.
    Exact integer arithmetic. Returns list of Python ints (constant-first)."""
    n = len(c_int)
    if n <= 1:
        return [0]
    q = [0] * (n - 1)
    q[-1] = c_int[-1]
    for i in range(n - 2, 0, -1):
        q[i - 1] = c_int[i] + base * q[i]
    return q


# ─── Modular arithmetic ───

def eval_poly_mod(coeffs_int, x, mod):
    """Evaluate polynomial (constant-first) at x mod mod. Exact int arithmetic."""
    val = 0
    xp = 1
    for c in coeffs_int:
        val = (val + c * xp) % mod
        xp = (xp * x) % mod
    return val


def poly_roots_mod(poly, l):
    """Return frozenset of all roots of poly in Z/lZ."""
    return frozenset(x for x in range(l) if eval_poly_mod(poly, x, l) == 0)


def multiplicative_order(a, n):
    """Order of a in (Z/nZ)*. Returns 0 if gcd(a,n) > 1."""
    if math.gcd(a, n) > 1:
        return 0
    order = 1
    cur = a % n
    while cur != 1:
        cur = (cur * a) % n
        order += 1
        if order > n:
            return 0
    return order


def euler_totient(n):
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def primitive_root(l):
    """Find a primitive root (generator) of (Z/lZ)* for prime l."""
    if l == 2:
        return 1
    phi = l - 1
    # Factor phi
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


def discrete_log(x, g, l):
    """Compute log_g(x) mod (l-1) for prime l via baby-step giant-step."""
    if x % l == 0:
        return None
    x = x % l
    n = l - 1
    m = int(math.isqrt(n)) + 1
    # Baby step: g^j for j = 0..m-1
    table = {}
    power = 1
    for j in range(m):
        table[power] = j
        power = (power * g) % l
    # Giant step: x * (g^{-m})^i
    factor = pow(g, n - m, l)  # g^{-m} mod l
    gamma = x
    for i in range(m):
        if gamma in table:
            ans = (i * m + table[gamma]) % n
            return ans
        gamma = (gamma * factor) % l
    return None


def build_character_table(l):
    """Build Dirichlet character table for prime l.
    Returns (g, log_table, phi) where:
      - g is a primitive root mod l
      - log_table[x] = log_g(x) for x in 1..l-1
      - phi = l-1 = order of the group
    Character chi_j(x) = exp(2*pi*i * j * log_table[x] / phi)
    """
    g = primitive_root(l)
    phi = l - 1
    log_table = {}
    power = 1
    for k in range(phi):
        log_table[power] = k
        power = (power * g) % l
    return g, log_table, phi


def legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p) for odd prime p."""
    a = a % p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def measure_ratio(Q, p_val, q_val, test_primes):
    """Anti-correlation ratio for one semiprime: observed / expected.
    Matches the original carry_law_verify.py definition."""
    total_hits = 0
    total_expected = 0.0
    for l in test_primes:
        roots = poly_roots_mod(Q, l)
        if not roots:
            continue
        pm = p_val % l
        qm = q_val % l
        hits = int(pm in roots) + (int(qm in roots) if qm != pm else 0)
        n_distinct = 1 if pm == qm else 2
        total_hits += hits
        total_expected += len(roots) / l * n_distinct
    return total_hits / total_expected if total_expected > 0 else None
