# Polynomial Solvers

Several minimal geometry solvers in calibration-rs reduce to polynomial equations. This chapter documents the polynomial root-finding utilities used by P3P, 7-point fundamental, and 5-point essential solvers.

## Quadratic

$$ax^2 + bx + c = 0$$

Solved via the standard discriminant formula: $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$. Returns 0, 1, or 2 real roots depending on the discriminant sign.

## Cubic

$$ax^3 + bx^2 + cx + d = 0$$

Solved using Cardano's formula:

1. Convert to depressed cubic $t^3 + pt + q = 0$ via substitution $x = t - b/(3a)$
2. Compute discriminant $\Delta = -4p^3 - 27q^2$
3. For $\Delta > 0$ (three real roots): use trigonometric form
4. For $\Delta \leq 0$ (one real root): use Cardano's formula with cube roots

Returns 1 or 3 real roots.

## Quartic

$$ax^4 + bx^3 + cx^2 + dx + e = 0$$

Solved via the companion matrix approach:

1. Normalize to monic form: divide by $a$
2. Construct the $4 \times 4$ companion matrix
3. Compute eigenvalues using Schur decomposition (via nalgebra)
4. Extract real eigenvalues: those with $|\text{Im}(\lambda)| < \epsilon$

Returns 0 to 4 real roots.

## Usage in Minimal Solvers

| Solver | Polynomial | Degree | Max roots |
|--------|-----------|--------|-----------|
| P3P (Kneip) | Distance ratio | Quartic | 4 |
| 7-point fundamental | $\det(F_1 + t F_2) = 0$ | Cubic | 3 |
| 5-point essential | Action matrix eigenvalues | Degree 10 | 10 |

The 5-point solver uses eigendecomposition of a $10 \times 10$ matrix rather than a polynomial solver directly, but the underlying mathematics involves degree-10 polynomial constraints.

## Numerical Considerations

- All roots are deduplicated (roots closer than $\epsilon$ are merged)
- Real root extraction uses a threshold on the imaginary part ($|\text{Im}| < 10^{-8}$)
- The companion matrix approach is more numerically stable than analytical quartic formulas for extreme coefficient ratios
