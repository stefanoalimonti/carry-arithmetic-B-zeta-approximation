# Experiments — Paper B

| Script | Description | Referenced in |
|--------|-------------|---------------|
| `B01_bm1_over_b_anticorrelation_law.py` | Anti-correlation law $(b-1)/b$ verification | §5 |
| `B02_carry_representation_theorem.py` | CRT verification (185K semiprimes) | §2, Theorem 1 |
| `B03_unit_leading_carry_proof.py` | ULC proof verification (20K tests) | §3, Theorem 2 |
| `B04_eigenvalue_bound.py` | Eneström–Kakeya bound $r_{\max} \leq 3$ | §6.1 |
| `B05_perfactor_identity_decomposition.py` | Per-factor identity decomposition | §4 |
| `B06_carry_phantom_fix.py` | Phantom zero analysis (carry vs Euler product) | §7.2 |
| `B07_renormalization_factor_R.py` | $R(l,s)$ measurement and characterization | §4.2 |
| `B08_rmax_analytical_proof.py` | $r_{\max} \leq 2$ analytical proof | §6.1 |
| `B09_functional_equation_symbolic.py` | Functional equation symbolic test | §7.2 |
| `B10_trace_anomaly_unitarity.py` | Renormalized determinant unitarity | §4.2, §4.3 |
| `B11_rmax_tighter_bound.py` | $r_{\max}$ tighter bound | §6.1 |
| `B12_constant_c_precision.py` | $c_1$ precision measurement | §4.2 |
| `B13_phantom_zero_multiplier.py` | Phantom zero multiplier $M(L,T)$ | §7.2 |
| `B14_c2_analytical_decomposition.py` | $\alpha_k$ algebraic decomposition ($c_2$) | §4.2, §5 |
| `B15_CRT_ULC_proof.py` | CRT + ULC combined verification | §2, §3 |
| `B16_P_neg2_analytical_proof.py` | $P(-2) \neq 0$ proof ($r_{\max} < 2$) | §6.1 |
| `B17_P_neg2_counterexample_search.py` | $P(-2) = 0$ counterexample search | §6.1 |
| `B18_rmax_boundary_analysis.py` | $r_{\max}$ boundary analysis | §6.1 |
| `B19_alpha2_markov_chain.py` | $\alpha_2$ via carry Markov chain | §5 |
| `B20_exact_perfactor_convergence.py` | Per-factor convergence as $D \to \infty$ | §4.1, §7.2 |
| `B21_markov_correction_R_factor.py` | Persistent $O(1/l^2)$ correction in $R$ | §4.2 |
| `B22_R_characterization.py` | $R(l,s)$ analytical characterization | §4.2, §4.3 |
| `B23_dirichlet_decomposition.py` | Dirichlet character decomposition of $R$ | §4.3 |
| `B24_identify_c1_precision.py` | $c_1$ universality and constant identification | §4.2 |
| `B25_asymptotic_c1_limit.py` | $c_1(D \to \infty)$ asymptotic limit | §4.2 |
| `B26_highprec_c1_measurement.py` | High-statistics Monte Carlo $c_1$ | §4.2, §7.3 |
| `B27_jensen_gap_test.py` | Jensen gap test ($R = 1$ hypothesis: **FALSE**) | §4.3 |
| `B28_multibase_c1.py` | Multi-base $c_1(b)$ analysis | §4.2 |
| `B29_transfer_operator.py` | Transfer operator eigenvalue $\lambda_2 = 1/b$ | §4.2, §7.3 |
| `B30_convergence_model.py` | Convergence model $c_1(K) = \pi/18 + P(K)(1/2)^K$ | §4.2, §7.3 |

## Shared Utilities

`carry_utils.py` is in `../src/`.

## Requirements

Python 3.8+, NumPy, SciPy, SymPy, mpmath.
