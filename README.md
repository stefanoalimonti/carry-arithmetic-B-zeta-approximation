# carry-arithmetic-B-zeta-approximation

**Carry Polynomials and the Euler Product: An Approximation Framework**

*Author: Stefano Alimonti* · [ORCID 0009-0009-1183-1698](https://orcid.org/0009-0009-1183-1698)

---

## Abstract

Positional integer multiplication in base $b$ generates companion matrices $M_l$ — one per prime $l$ — whose ensemble-averaged spectral determinant approximates individual Euler factors of the Riemann zeta function:

$$\langle|\det(I - M_l/l^s)|\rangle_N = \frac{R(l,s)}{|1 - l^{-s}|}$$

where $R(l,s) = 1 + O(l^{-\sigma})$ is a genuine correction factor ($R \neq 1$). Two structural results are proven rigorously: the **Carry Representation Theorem** (CRT: $q_i = -c_{i+1}$) and the **Unit Leading Carry** theorem (ULC: $c_{\text{top}} = 1$ in base 2). The correction factor admits the universal expansion $\ln R = c_1/l^s + c_2/l^{2s} + \cdots$, where $c_1 = \pi/18$ (conjectured, 4.3 digits from direct enumeration). This paper provides the central approximation framework connecting all companion papers.

## Repository Structure

```
├── paper/
│   └── carry_zeta_approximation.md
├── experiments/
│   ├── B01_bm1_over_b_anticorrelation_law.py
│   ├── ...
│   └── B28_multibase_c1.py
├── src/
│   └── carry_utils.py
├── CITATION.cff
├── LICENSE
└── README.md
```

## Key Results

| Result | Status |
|--------|--------|
| CRT: $q_i = -c_{i+1}$ | **Proven** (Theorem 1) |
| ULC: $c_{\text{top}} = 1$ (base 2) | **Proven** (Theorem 2) |
| Per-factor error $O(1/l^2)$ | **Empirical** ($l^2(R-1) \approx 1/6$) |
| Trace anomaly $c_1 = \pi/18$ (base 2) | **Conjectured** (4.3 digits from enumeration at $K = 21$) |
| Angular Uniqueness: $\pi$ only for base 2 | **Proved** (algebraic; [G]) |
| Jensen hypothesis ($R = 1$) | **FALSE** |

## Companion Papers

| Label | Repository | Title |
|-------|-----------|-------|
| [A] | [`carry-arithmetic-A-spectral-theory`](https://github.com/stefanoalimonti/carry-arithmetic-A-spectral-theory) | Spectral Theory of Carries in Positional Multiplication |
| [C] | [`carry-arithmetic-C-matrix-statistics`](https://github.com/stefanoalimonti/carry-arithmetic-C-matrix-statistics) | Eigenvalue Statistics of Carry Companion Matrices: Markov-Driven GOE↔GUE Transition |
| [D] | [`carry-arithmetic-D-factorization-limits`](https://github.com/stefanoalimonti/carry-arithmetic-D-factorization-limits) | The Carry-Zero Entropy Bound |
| [E] | [`carry-arithmetic-E-trace-anomaly`](https://github.com/stefanoalimonti/carry-arithmetic-E-trace-anomaly) | The Trace Anomaly of Binary Multiplication |
| [F] | [`carry-arithmetic-F-covariance-structure`](https://github.com/stefanoalimonti/carry-arithmetic-F-covariance-structure) | Exact Covariance Structure of Binary Carry Chains |
| [G] | [`carry-arithmetic-G-angular-uniqueness`](https://github.com/stefanoalimonti/carry-arithmetic-G-angular-uniqueness) | The Angular Uniqueness of Base 2 |
| [H] | [`carry-arithmetic-H-euler-control`](https://github.com/stefanoalimonti/carry-arithmetic-H-euler-control) | Carry Polynomials and the Partial Euler Product: A Control Experiment |

## Requirements

Python 3.8+, NumPy, SciPy, SymPy, mpmath.

### Citation

```bibtex
@article{alimonti2026euler_product,
  author  = {Alimonti, Stefano},
  title   = {Carry Polynomials and the Euler Product: An Approximation Framework},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/stefanoalimonti/carry-arithmetic-B-zeta-approximation}
}
```

## License

Paper: CC BY 4.0. Code: MIT License.
