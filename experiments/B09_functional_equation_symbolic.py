import sympy as sp
from sympy import pprint, Eq, simplify, collect, factor

def test_functional_equation_symbolic():
    print("=" * 60)
    print(" B09: Symbolic Derivation of Functional Equation")
    print("=" * 60)
    
    # 1. Define symbols
    s, l = sp.symbols('s l')
    d = 4 # Use a fixed small degree for symbolic tractability
    
    print(f"\n1. Defining a symmetric (reciprocal) polynomial of degree {d}")
    # Symmetric coefficients: c0 = c4, c1 = c3
    c0, c1, c2 = sp.symbols('c0 c1 c2')
    # For a reciprocal polynomial, roots have z <-> 1/z symmetry
    # Q(z) = c0*z^4 + c1*z^3 + c2*z^2 + c1*z + c0
    z = sp.symbols('z')
    Q = c0*z**4 + c1*z**3 + c2*z**2 + c1*z + c0
    print("Q(z) = ")
    pprint(Q)
    
    print("\nCheck reciprocal symmetry: z^d * Q(1/z) == Q(z)")
    Q_inv = simplify(z**d * Q.subs(z, 1/z))
    print(f"z^{d} * Q(1/z) = ")
    pprint(Q_inv)
    assert simplify(Q - Q_inv) == 0, "Symmetry failed"
    
    print("\n2. Formulating the Spectral Determinant Delta(x) = det(I - M*x)")
    # For a matrix M with characteristic polynomial Q(lambda) = 0
    # det(I - M*x) = x^d * det(1/x * I - M) = x^d * Q(1/x) / c0 (if c0 is the leading coeff)
    # Let's verify this relation explicitly for a companion matrix
    x = sp.symbols('x')
    # Normalized Q(z) so leading coeff is 1
    Q_norm = Q / c0
    
    # Companion matrix for Q_norm = z^4 + (c1/c0)z^3 + (c2/c0)z^2 + (c1/c0)z + 1
    M = sp.Matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, -c1/c0, -c2/c0, -c1/c0]
    ])
    
    I = sp.eye(4)
    Delta_x = (I - M * x).det()
    print("Delta(x) = det(I - M*x) = ")
    Delta_x = sp.collect(sp.expand(Delta_x), x)
    pprint(Delta_x)
    
    # Observe that Delta_x is exactly Q(x)/c0 for a reciprocal polynomial!
    print("\nNotice that Delta(x) is identical to Q(x)/c0:")
    pprint(simplify(Delta_x - Q.subs(z, x)/c0))
    
    print("\n3. Evaluating on the critical strip: x = l^(-s)")
    # Let x = l^(-s)
    Delta_s = Delta_x.subs(x, l**(-s))
    print("Delta(s) = ")
    pprint(Delta_s)
    
    print("\n4. Evaluating the reflection: 1-s")
    # Let x = l^(-(1-s)) = l^(s-1)
    Delta_1_minus_s = Delta_x.subs(x, l**(s-1))
    print("Delta(1-s) = ")
    pprint(Delta_1_minus_s)
    
    print("\n5. Searching for the Functional Equation connecting Delta(s) and Delta(1-s)")
    # We want to find a factor F(s, l) such that Delta(s) = F(s, l) * Delta(1-s)
    # Let's look at the ratio
    ratio = sp.simplify(Delta_s / Delta_1_minus_s)
    print("Ratio Delta(s) / Delta(1-s) = ")
    pprint(ratio)
    
    # Let's force the symmetry algebraically
    # Delta(s) = 1 + c1/c0 * l^(-s) + c2/c0 * l^(-2s) + c1/c0 * l^(-3s) + l^(-4s)
    # Delta(1-s) = 1 + c1/c0 * l^(s-1) + c2/c0 * l^(2s-2) + c1/c0 * l^(3s-3) + l^(4s-4)
    
    print("\nLet's extract l^(-d*s) from Delta(s)")
    # Pull out l^(-d*s/2) to see if we get a symmetric real function on the critical line
    # On the critical line s = 1/2 + i*t, l^(-s) = l^(-1/2) * l^(-i*t)
    
    t = sp.symbols('t', real=True)
    s_crit = 1/2 + sp.I * t
    
    Delta_crit = Delta_x.subs(x, l**(-s_crit))
    
    # Multiply by l^(d*s/2) = l^(2*s) = l^(1 + 2*i*t)
    # Actually, the standard symmetric form is chi(s) = Delta(s) * l^(d*s/2)
    sym_factor = l**(d * s / 2)
    Symmetric_Delta = sp.simplify(Delta_s * sym_factor)
    
    print("\nSymmetric function Xi(s) = l^(d*s/2) * Delta(s):")
    sp.pprint(Symmetric_Delta)
    
    print("Let's check if Xi(s) == Xi(-s) or similar.")
    # For a reciprocal polynomial, Delta(x) = x^d Delta(1/x)
    # So Delta(l^-s) = l^(-d*s) * Delta(l^s)
    
    rel1 = simplify(Delta_s - l**(-d*s) * Delta_x.subs(x, l**s))
    print(f"\nFundamental Algebraic Relation derived:")
    print(f"Delta(s) - l^(-{d}s) * Delta(-s) = {rel1}")
    
    print("\n6. Bridging to Riemann's s <-> 1-s")
    print("The derived relation is Delta(s) = l^(-d*s) * Delta(-s).")
    print("But Riemann's functional equation is about s <-> 1-s.")
    print("Where does the '1' come from?")
    
    # In Riemann Zeta, the Euler product is Prod (1 - p^-s)^-1
    # The individual factors DO NOT satisfy s <-> 1-s.
    # (1 - p^-s) != (1 - p^(s-1)) * something simple.
    # The s <-> 1-s symmetry emerges ONLY AFTER the infinite product is completed with the Gamma factor.
    
    print("\nLet's analyze the roots of Q(z).")
    print("If roots are on |z| = 1, then z = e^(i*theta).")
    print("x = l^-s. The spectral determinant is zero when 1 - lambda * l^-s = 0 => l^s = lambda.")
    print("s * ln(l) = i*theta + 2*pi*i*k")
    print("s = i * (theta + 2*pi*k) / ln(l)")
    print("This means the roots of Delta(s) have Re(s) = 0! NOT Re(s) = 1/2.")
    
    print("\n7. The critical Re(s) = 1/2 Shift")
    print("To shift the roots from Re(s) = 0 to Re(s) = 1/2, we need:")
    print("l^(s - 1/2) = lambda  =>  l^s = lambda * sqrt(l)")
    print("This means the matrix eigenvalues must be |lambda| = sqrt(l), NOT |lambda| = 1.")
    
    print("\nCONCLUSION:")
    print("If the carry companion matrix M has eigenvalues exactly on |lambda| = sqrt(l),")
    print("then the roots of det(I - M/l^s) will lie EXACTLY on Re(s) = 1/2.")
    print("Let's check the eigenvalues of our actual empirical carry matrices!")

if __name__ == "__main__":
    test_functional_equation_symbolic()
