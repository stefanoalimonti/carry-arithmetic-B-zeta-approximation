#!/usr/bin/env python3
"""
B28: Multi-Base c₁ — Test c₁ = ln(b)/b² in bases 2,3,5,7,10

If c₁(base 2) = ln(2)/4 = ln(2)/2², then the natural conjecture is:
  c₁(base b) = ln(b)/b²

This would mean the trace anomaly encodes the entropy of the carry
chain per unit volume: ln(b) = base entropy, 1/b² = Markov scaling.

Quick test: compute ⟨c_{top-1}⟩ - 1 in each base at several bit sizes.
"""

import sys, os, time, random, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from carry_utils import random_prime, carry_poly_int, quotient_poly_int

random.seed(42)
np.random.seed(42)


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def get_ctop1(p, q, base=2):
    C = carry_poly_int(p, q, base)
    Q = quotient_poly_int(C, base)
    if len(Q) < 3:
        return None
    c_top = -Q[-1]
    c_top1 = -Q[-2]
    return c_top, c_top1


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("P4-03: MULTI-BASE c₁ — IS c₁ = ln(b)/b² ?")
    pr("=" * 72)
    pr(f"""
  Predictions:
    base 2:  c₁ = ln(2)/4    = {math.log(2)/4:.8f}
    base 3:  c₁ = ln(3)/9    = {math.log(3)/9:.8f}
    base 5:  c₁ = ln(5)/25   = {math.log(5)/25:.8f}
    base 7:  c₁ = ln(7)/49   = {math.log(7)/49:.8f}
    base 10: c₁ = ln(10)/100 = {math.log(10)/100:.8f}
""")

    n_samp = 40000

    for base in [2, 3, 5, 7, 10]:
        pr(f"\n{'═' * 72}")
        pr(f"  BASE {base}: predicted c₁ = ln({base})/{base}² = {math.log(base)/base**2:.8f}")
        pr(f"{'═' * 72}\n")

        predicted = math.log(base) / base**2

        for bits in [14, 18, 22, 28, 36, 44]:
            ctop_vals = []
            ctop1_vals = []
            ulc_ok = 0
            total = 0

            for _ in range(n_samp):
                p = random_prime(bits)
                q = random_prime(bits)
                if p == q:
                    continue
                result = get_ctop1(p, q, base)
                if result is None:
                    continue
                ct, ct1 = result
                total += 1
                ctop_vals.append(ct)
                if ct > 0:
                    ctop1_vals.append(ct1)
                    if ct == 1 and base == 2:
                        ulc_ok += 1
                    elif base > 2 and ct <= base - 1:
                        ulc_ok += 1

            if not ctop1_vals:
                continue

            arr = np.array(ctop1_vals, dtype=float)
            ct_arr = np.array(ctop_vals, dtype=float)

            c1_raw = arr.mean() - ct_arr[ct_arr > 0].mean()
            se = arr.std() / math.sqrt(len(arr))

            c1_measured = np.mean(arr / ct_arr[ct_arr > 0].mean()) - 1 if ct_arr[ct_arr > 0].mean() > 0 else 0

            tr_mean = -np.mean(arr / ct_arr[ct_arr > 0])
            c1_trace = -tr_mean - 1

            delta = c1_trace - predicted
            nsigma = abs(delta) / se if se > 0 else 999

            pr(f"    bits={bits:2d}: ⟨c_top⟩={ct_arr.mean():.4f}, "
               f"⟨c_top-1⟩={arr.mean():.4f}, "
               f"c₁(trace)={c1_trace:+.6f}, "
               f"predicted={predicted:.6f}, "
               f"Δ={delta:+.6f} ({nsigma:.1f}σ)")

    # ═══════════════════════════════════════════════════════════════
    # ALTERNATIVE: c₁ = (b-1)·ln(b) / (2·b²) ?
    # ═══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("ALTERNATIVE FORMULAS FOR c₁(base b)")
    pr(f"{'═' * 72}\n")

    formulas = {
        'ln(b)/b²':       lambda b: math.log(b) / b**2,
        '(b-1)ln(b)/(2b²)': lambda b: (b-1)*math.log(b) / (2*b**2),
        'ln(b)/(b²+b)':   lambda b: math.log(b) / (b**2 + b),
        'ln(b)/(2b(b-1))': lambda b: math.log(b) / (2*b*(b-1)) if b > 1 else 0,
        '1/(2b)':          lambda b: 1.0 / (2*b),
        '(b-1)/(2b²)':    lambda b: (b-1) / (2*b**2),
    }

    pr(f"  {'Formula':<22s}", end='')
    for b in [2, 3, 5, 7, 10]:
        pr(f"  {'b='+str(b):>10s}", end='')
    pr()
    pr(f"  {'─'*22}", end='')
    for _ in [2, 3, 5, 7, 10]:
        pr(f"  {'─'*10}", end='')
    pr()

    for name, f in formulas.items():
        pr(f"  {name:<22s}", end='')
        for b in [2, 3, 5, 7, 10]:
            pr(f"  {f(b):10.6f}", end='')
        pr()

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
