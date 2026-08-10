# Urankar Part V — complete method chain, audited equation by equation

Source: L. Urankar, "Vector potential and magnetic field of current-carrying
circular finite arc segment in analytical form — Part V: polygon cross
section," IEEE Trans. Magn. 26(3), 1171–1180 (1990).  Read from 400-dpi page
renders of the ITER library PDF; every equation below was transcribed from the
printed page, not from OCR.  Notation kept as printed except where stated.
Page references are to the journal pagination 1171–1180.

The goal of this audit: the FULL TURN (axisymmetric ring, arc length 2π) flux
and field of a polygon-section conductor, in closed form, at production aspect
ratios (section/major radius ~ 1/100).  The full turn is the paper's case
`3π/2 < α_i ≤ 2π` (eq 26 region), which reduces every elliptic integral to its
COMPLETE value at α = π/2, so the whole chain below is specialised at the end
to α = π/2.

---

## 1. Geometry and basic quantities (pp. 1171–1173)

Field (target) point `(r, φ=0, z)`; source point `(r', φ', z')` on the section.
Cross-section in the r–z plane, an n-sided polygon, edge ν running from
`(r'_ν1, z'_ν1)` to `(r'_ν2, z'_ν2)`.

Eq (3a/4a) normalisation: `A_j = J_φ/(4π) Â_j`, `B_l = μ0 J_φ/(4π) Ĥ_l`,
with `J_φ` the (constant) azimuthal current density.

Eq (3b): `Â_j(r̄) = ∫_{φ1}^{φ2} dφ f(φ) [−sin φ; cos φ]` (rows j = r, φ).

Eq (4b): `Ĥ_l(r̄) = ∫_{φ1}^{φ2} dφ {−cos φ f_z(φ); −sin φ f_z(φ);
cos φ f_r(φ) − (1/r) sin φ f_φ(φ)}` (rows l = r, φ, z; subscripts = partial
derivatives of f).

Eq (5): `f(φ) = ∬ r' dr' dz' / D(φ)`,
Eq (5a): `γ = z' − z`, `φ = φ' − φ_field`, `D²(φ) = γ² + r'² + r² − 2 r r' cos φ`.

Stokes ⇒ contour sum, eq (6): `f(φ) = Σ_ν f_ν(φ)`.

Eq (7)/(7a): parametrise the edge; `Δr = r'_ν2 − r'_ν1`, `Δz = z'_ν2 − z'_ν1`.
**For Δz = 0 (horizontal edge), f_ν(φ) vanishes** (stated below eq 7a).

Eq (8)/(8a): with `u = z' − z` as edge parameter,

    u_ν1 = z'_ν1 − z ,   u_ν2 = z'_ν2 − z ,
    b1 = Δr/Δz  (edge slope),   r1 = r'_ν1 − b1 u_ν1  (edge intercept),
    D²(u) = u² + r² sin²φ + (r1 + b1 u − r cos φ)² ,
    β1(u) = (r1 + b1 u − r cos φ)/(u² + r² sin²φ)^{1/2} ,
    f_ν(φ) = −∫_{u_ν1}^{u_ν2} du [ D(u) + r cos φ arsinh β1(u) ].

## 2. The edge antiderivative (eq 9, 9a; p. 1173)

Integrating (8) (suppressing limits, writing γ for u):

    f_ν(φ) = −{ γ r cos φ arsinh β1(φ)
              + 1/(2 a0³) [B²(φ) + 2 a0² r cos φ (r1 − r cos φ)] arsinh β2(φ)
              − (r²/2) sin 2φ arctan β3(φ)
              + Γ(φ) D(φ)/(2 a0²) }                                   (9)

(The printed "Γ(φ)/2a0² D(φ)" has D as a NUMERATOR factor — established
typesetting trap, already regression-pinned in `nova/biot/polygon.py`.)

Eq (9a) symbol definitions:

    a0² = 1 + b1² ,            r' = r1 + b1 γ ,
    Γ(φ)  = γ + b1 (r' − r cos φ) ,
    G²(φ) = γ² + r² sin²φ ,
    B²(φ) = (r1 − r cos φ)² + a0² r² sin²φ ,
    β1(φ) = (r' − r cos φ)/G(φ) ,
    β2(φ) = Γ(φ)/B(φ) ,
    β3(φ) = [γ (r' − r cos φ) − b1 G²(φ)] / (r sin φ D(φ)) .

Eq (10a/11a): potential and field are sums over edges of `[·]` evaluated
between the two u-limits ν1 → ν2.

Eq (10b): `Â_jν = −∫ dφ [−sinφ; cosφ] {Γ D/(2a0²) + γ r cosφ arsinh β1
+ 1/(2a0³)[B² + 2a0² r cosφ (r1 − r cosφ)] arsinh β2 − r²/2 sin2φ arctan β3}`.

Eq (11b), the FIELD edge integrand (verbatim, rows l = r, φ, z):

    Ĥ_lν = −∫_{φ1}^{φ2} dφ {
      cos φ [ D(φ)/a0² + r cos φ arsinh β1(φ)
              − b1/a0³ (r1 + b1² r cos φ) arsinh β2(φ) ]          (l = r)
      sin φ [ … same bracket … ]                                   (l = φ)
      γ arsinh β1(φ) + 1/a0³ [b1² r1 − (2a0² − 1) r cos φ] arsinh β2(φ)
              − r sin φ arctan β3(φ) − b1/a0² D(φ)                 (l = z)
    }

**ERRATUM, verified (see §7 for the derivation).** The z row's rational term is
printed as `− b1²/a0² D(φ)`; it is `− b1/a0² D(φ)`, LINEAR in the edge slope.
The transcription above is CORRECTED.  The two forms agree only at `b1 = 0` and
`b1 = 1`, so a rectangular section and a 45-degree edge both hide the error —
which is why it survives casual checking.  With the printed form the assembled
`B_Z` is wrong by 286% over a trapezium, 7.6% over a thin plate and 1.1% over a
hexagon; with the linear form the closed form agrees with an independent boundary
quadrature to 1.3e-12 on a trapezium whose edge slopes are −5.0, 0.2, 4.375 and
−0.417 (none of them 0 or 1).

For `b1 = 0` the paper notes (p. 1173) the expressions of Part III
(rectangular section) are regained immediately.

**Structural zero of B_φ (full turn):** the φ row is the r-row bracket
weighted by `sin φ` instead of `cos φ`; the bracket is even about φ = π while
`sin φ` is odd, so the full-turn integral vanishes identically.  Axisymmetry
is reproduced by the algebra, not assumed.

## 3. Angle transformation and case bookkeeping (pp. 1173–1174, 1177–1178)

Eq (A): `φ = π − 2α` (system-invariant angle transformation).  Hence

    cos φ = −cos 2α = 2 sin²α − 1 ,   sin φ = sin 2α ,   sin 2φ = −sin 4α ,
    x ≡ sin²α :  cos φ = 2x − 1 ,  sin²φ = 4x(1−x) .

Eq (12): `Â_jν = Σ_{i=1,2} (−1)^{i+1} ∫_0^{α_i} dα [sin 2α; cos 2α] ·
[Γ(α)/a0² · a (1 − k² sin²α)^{1/2} − 2γr cos 2α arsinh β1(α)
+ 1/a0³ {B²(α) − 2a0² r cos 2α (r1 + r cos 2α)} arsinh β2(α)
+ r² sin 4α arctan β3(α)]`, with

    k² = 4 r r'/a² ,   a² = γ² + (r + r')² ,   α_i = ½(π + φ − φ_i) .

Eq (13): `Â_jν(r̄) = −Σ_{i=1,2} (−1)^{i+1} (δ_jr + δ_jφ sgn α_i)
[Â_jν(α)]_{α=|α_i|}`.

Eq (17), field analogue: `Ĥ_lν(r̄) = −Σ_{i=1,2} (−1)^{i+1}
(δ_lφ + δ_lm sgn α_i) [Ĥ_lν(α)]_{α=|α_i|}`, (l = r, φ, z; m = r, z).

Cases (p. 1177–1178): the closed expressions hold for `0 < α ≤ π/2`.
Case B (`π/2 < |α_i| ≤ 3π/2`): `θ_i = π − |α_i|`,
eq (25): `Â_φ(|α_i|) = 2Â_φ(π/2) − sgn θ_i Â_φ(|θ_i|)`;
eq (26): `Ĥ_m(|α_i|) = 2Ĥ_m(π/2) − sgn θ_i Ĥ_m(|θ_i|)` (m = r, z).
Case C (`3π/2 < α_i ≤ 2π`): `θ_i = 2π − α_i`,
`Â_φ(|α_i|) = 4Â_φ(π/2) − Â_φ(θ_i)` (and likewise Ĥ_m).

**Full-turn specialisation** (this project's case).  Take `φ2 = φ1 + 2π`,
target azimuth anywhere between: `ψ = φ − φ1 ∈ (0, 2π)`,
`α_1 = (π + ψ)/2 ∈ (π/2, 3π/2)` (case B), `α_2 = α_1 − π ∈ (−π/2, π/2)`
(case A), and `θ_1 = π − α_1 = −α_2`.  Then from (13):

    Â_φ = −[ Â_φ(|α_1|) − sgn α_2 Â_φ(|α_2|) ]
        = −[ 2Â_φ(π/2) − sgn θ_1 Â_φ(|θ_1|) − sgn α_2 Â_φ(|α_2|) ]
        = −2 Â_φ(π/2)                                   (the θ/α_2 terms cancel)

and identically `Ĥ_r = −2Ĥ_r(π/2)`, `Ĥ_z = −2Ĥ_z(π/2)`.  Everything is
evaluated at α = π/2 exactly: sn = 1, cn = 0, dn = k', every elliptic
integral COMPLETE.  `cn K = 0` annihilates all odd-cn terms — precisely the
r-component of Â (axisymmetry).

## 4. Vector potential closed form (eqs 14, 14a–14e; p. 1174)

Integrating (12) twice partially, for positive definite α (rows j = r, φ):

    Â_jν(α) = C_jν(α)|_0^α + δ_jφ r [γ 𝒥1(α) + b1²/a0³ r1 𝒥2(α)]
             + I_jν^{(1)}(α) + I_jν^{(2)}(α) + I_jν^{(3)}(α)          (14)

Eq (14e): `𝒥_p(α) = ∫_0^α dα arsinh β_p(α)` (p = 1, 2) — evaluated
numerically ("evade an analytical treatment as yet").

Eq (14a), integration constant (rows [j=r; j=φ]):

    C_jν(α) = a/(2a0²) Γ(α)(1 − k² sin²α)^{1/2} [cos 2α; −sin 2α]
            − γr/4 arsinh β1(α) [3 + cos 4α; −sin 4α]
            − 1/(6a0³) arsinh β2(α) ×
              [ cos 2α (A0 r² cos²2α − 3(r1² + a0²r² − b1² r r1 cos 2α)) + 3b1² r r1 ;
                sin 2α (A0 r² sin²2α + 3(r1² − a0²r² − b1² r r1 cos 2α)) − 3b1² r² sin 2α ]
            − r²/3 arctan β3(α) [sin³ 2α; −cos³ 2α]                   (14a)

with `c² = γ² + r²` and `A0 = 3a0² − 1` (p. 1174, after 14d).

Eq (14b):

    I_jν^{(1)}(α) = 2 b1 r/a0² ∫_0^α dα sin 2α [ a(1 − k² sin²α)^{1/2}
        [cos 2α; −sin 2α] + r/(6 a0² a) (1 − k² sin²α)^{−1/2}
        [2A0 r cos³2α − 3b1² r1 sin²2α ; 2A0 r sin³2α − (3/2) b1² r1 sin 4α] ]

Eq (14c):

    I_jν^{(2)}(α) = −γ r²/(3a) ∫_0^α dα G^{−2}(α)(1 − k² sin²α)^{−1/2}
        [ 3 sin²2α (c² + r r' cos 2α) + 2r cos 2α (r' + r cos 2α)
          [cos 4α + 2; cos 4α − 2] ] [sin 2α; cos 2α]

Eq (14d):

    I_jν^{(3)}(α) = −r²/(3a) ∫_0^α dα Γ(α) B^{−2}(α) (1 − k² sin²α)^{−1/2}
        [ 2r(r + r1 cos 2α) [sin 2α (cos 4α + 2); cos 2α (cos 4α − 2)]
        + 1/a0⁴ (r1 − b1² r cos 2α)
          [sin 2α (2A0 r cos³2α − 3b1² r1 sin²2α);
           sin²2α (2A0 r sin²2α − 3b1² r1 cos 2α)] ]

(Note `r'` in (14c) and `Γ(α)` carry the edge-limit γ; `r' = r1 + b1 γ`.)

## 5. Jacobi reduction and coefficients (eqs 15a–16c; pp. 1174–1175)

Substituting `sin α = sn u` (modulus k implicit, amplitude `am u = |α|`)
converts (14b–14d) to integrals over Jacobi functions, eqs (15a–15c) —
integrands listed for the audit trail:

    I^{(1)} = 4b1 r/a0² ∫_0^u du [ sn cn (u_{1r} − u_{2r} sn² + 2u_{3r} sn⁴ − x0 sn⁶) ;
                                    −sn² (u_{1φ} − u_{2φ} sn² + 2u_{3φ} sn⁴ − x0 sn⁶) ]
    I^{(2)} = 2γ/(3a) ∫_0^u du [ sn cn (v_{1r} − 2r(b+7r) sn² + 16r² sn⁴) ;
                                  v_{1φ} − v_{2φ} sn² + 2r(b+11r) sn⁴ − 16r² sn⁶ ]
            + γ/(6ar) Σ_{p=1,2} (−1)^p P_j(n_p) ∫_0^u du/(1 − n_p² sn²) [n_p² sn cn; 1]
    I^{(3)} = 2/(3a0⁴b1³a) ∫_0^u du [ sn cn (w_{1r} − 2r w_{2r} sn² + 4r w_{3r} sn⁴ − y0 sn⁶) ;
                                       w_{1φ} − w_{2φ} sn² + 2r w_{3φ} sn⁴ + w_{4φ} sn⁶ − y0 sn⁸ ]
            + 1/(12a0⁵b1⁵) Σ_{p=1,2} (−1)^p Q_j(m_p) ∫_0^u du/(1 − m_p² sn²) [m_p² sn cn; 1]

Nomenclature (p. 1175, before 16a):

    b = r + r' ,  f = r + r1 ,  γ0 = γ + b1 b ,  g = b1² r − r1 ,
    h = b1² r + r1 ,  k'² = 1 − k² ,  d² = b1² r² + r1² .

Eq (16a), coefficients in I^{(1)}; `x0 = 8A0 r²/(3a0²a)`, `u0 = A0 r + b1² r1`:

    u_{1j} = { a + x0/8            ;  2a + b1² r r1/(a0² a) }
    u_{2j} = { a(2 + k²) + 2r u0/(a0² a)  ;  x0 + 3u_{1φ} − 2a(1 + k'²) }
    u_{3j} = { a k² + r(2u0 − b1² r1)/(a0² a)  ;  x0 + u_{1φ} − a(1 + k'²) }

Eq (16b), coefficients in I^{(2)}; `n_p² = 2r/[r + (−1)^p c]`:

    v_{1j} = { c² + r(b + 5r)  ;  b/(2r) (c² − 2r²) }
    v_{2φ} = 6r² − r'(r' − 4r/k²)
    P_j(n_p) = [r' − (−1)^p c] { (−1)^p (c² + 5r²)  ;  n_p² c (3r² − c²)/(2r) }

Eq (16c), coefficients in I^{(3)}; `y0 = 16A0 b1⁴ r³`,
`m_p² = 2b1² r/[g + (−1)^p a0 d]`:

    w_{1j} = { b1(γ0 x2 + 2b1 r x1) − (2g/b1²) w_{2r} + f² w̃_{3r} ;
               b1(γ0 y2 + b1 r y1) − (g/b1²) r w_{2φ} + (f²/2) b1² r w_{3φ} }
    w_{2j} = { b1(2A0 b1 f² + γ0 x3 + b1 x2) − 2g w̃_{3r} ;
               b1(γ0 y3 + 2b1 r y2) − (2g/b1²) w_{3φ} − f² w̃_{4φ} }
    w_{3r} = b1² r w̃_{3r} ,        w̃_{3r} = x3 − 2A0(a0² r' + 3g) ,
    w_{3φ} = b1(γ0 y4 + b1 y3 − 2A0 b1 f²) + 2g w̃_{4φ} ,
    w_{4φ} = 4b1² r² w̃_{4φ} ,      w̃_{4φ} = 2A0(a0² r' − g) − y4 ,
    Q_j(m_p) = 1/(a d r²) { −2r [Z1 − (−1)^p a0 d Z2] ;  m_p² [Z3 + (−1)^p a0 d Z4] }

with

    x1 = a0⁴ r (3r1 + 4f) − A0 b1² r² − 3g u0
    x2 = 4a0⁴ r (2r1 + f) − 3g(2u0 − b1² r1) − 6b1² r u0
    x3 = 4a0⁴ r1 − 2A0 g − 3b1²(2u0 − b1² r1)
    y1 = a0⁴ r (2f + r) − 3b1² r1 g
    y2 = 3a0⁴ r (2f + r1) − (3/2) b1² r1 (5g + 2r1) − 4A0 r g
    y3 = 4a0⁴ r (f + 3r1) − 3b1² r1 (4g + 3r1) − 8A0 r (2g + r1)
    y4 = 4a0⁴ r1 − 3b1⁴ r1 − 2A0 (5g + 4r1)
    Z1 = b1² [f² w_{1r} + 2b1³ γ0 r³ (3a0⁴ f − A0 g)] − g Z2
    Z2 = 2b1³ r [γ0 x1 + b1 r² (3a0⁴ f − A0 g)] − 2g w_{1r} + f² w_{2r}
    Z3 = 2b1² r f [a0⁴ b1³ r³ γ0 − f w_{1φ}] + g Z4
    Z4 = 2b1³ r² (γ0 y1 − a0⁴ b1 f r²) − 8 r g w_{1φ} + f² w_{2φ}

## 6. Final potential expressions, case A / α = π/2 (eqs 21a–21c; p. 1177)

With `x = k sn u`, `El2m = ∫_0^u sn^{2m} du`, `I_m(x)` per eq (24a), and
`π(α, η², k)` the incomplete third-kind integral (Legendre, modulus k,
parameter η²):

    I_jν^{(1)}(α) = 4b1 r/a0² { k^{−2}[u_{1r} I1(x) − k^{−2} u_{2r} I3(x)
                                 + 2k^{−4} u_{3r} I5(x) − k^{−6} x0 I7(x)] ;
                                −u_{1φ} El2 + u_{2φ} El4 − 2u_{3φ} El6 + x0 El8 }   (21a)

    I_jν^{(2)}(α) = 2γ/(3a) [ k^{−2}[v_{1r} I1(x) − 2r(b+7r) k^{−2} I3(x) + 16r² k^{−4} I5(x)] ;
                              v_{1φ} El0 − v_{2φ} El2 + 2r(b+11r) El4 − 16r² El6 ]
                  + γ/(6ar) Σ_{p=1,2} (−1)^p P_j(n_p) { I(n_p²) ; π(α, n_p², k) }   (21b)

    I_jν^{(3)}(α) = 2/(3a0⁴b1³a) [ k^{−2}[w_{1r} I1 − 2r k^{−2} w_{2r} I3
                                     + 4r k^{−4} w_{3r} I5 − y0 k^{−6} I7] ;
                                   w_{1φ} El0 − w_{2φ} El2 + 2r w_{3φ} El4
                                     + w_{4φ} El6 − y0 El8 ]
                  + 1/(12a0⁵b1⁵) Σ_{p=1,2} (−1)^p Q_j(m_p) { I(m_p²) ; π(α, m_p², k) }  (21c)

Eq (23): `I(η²) = η² ∫_0^u du sn cn/(1 − η² sn²u)`.
Eq (24a): `(2m+1) I_{2m+1}(x) = −x^{2m} dn u + 2m I_{2m−1}(x)`, x = k sn u
(as an antiderivative; all integration constants are `[·]|_0^α`, per the
Appendix-B note).  Complete limit: `I1 = 1 − k'`,
`(2m+1) I_{2m+1} = −k^{2m} k' + 2m I_{2m−1}`.
Eq (24b): `(2m+1) k² El2p = sn^{2m+1} cn dn + 2m(1+k²) El2m − (2m−1) El2q`
(p = m+1, q = m−1); `El0 = K(α,k)`, `k² El2 = K(α,k) − E(α,k)`.

At α = π/2 every `El2m` is the complete moment (`nova.biot.elliptic
.sn_moments`), every `π(π/2, η², k) = Π(η² | k²)`
(`complete_pi`), and `I(η²)` over the quarter period is elementary
(`pole_moment`).

**Full-turn φ-row constant:** at both α = 0 and α = π/2 every sin-2α/sin-4α
factor in (14a) vanishes; only the arctan row survives:

    C_φν|_0^{π/2} = −(π r²/6) [ sgn(γ (r1 − r)) + sgn(γ (r1 + r)) ]

using `β3 → sgn(γ(r1 − r))·∞` as φ → 0 and `→ sgn(γ(r1 + r))·∞` as φ → π.

## 7. Field closed form (eqs 18–20b, 22a–22c; pp. 1175–1177)

**ERRATUM in eq 11b's z row (carried into every field expression below).** The
printed rational term `− b1²/a0² D` is `− b1/a0² D`.  Derivation: apply eq 4b's
own prescription for the z row, `cos φ ∂g/∂r − (sin φ/r) ∂g/∂φ`, to the flux
antiderivative `g` of eq 9.  The `B²` denominator cancels through
`r² sin²φ + Y²/a0² = B²/a0²`, and the remainder collapses on
`Γ = a0² γ + b1 Y` together with `Γ² + B² = a0² D²`, leaving the slope to the
FIRST power.  Checked against a numerical derivative of `g` at five edge slopes
(b1 = −5, 0, 0.1, 1, 4): the linear form matches to 5e-10, the finite-difference
floor, at every slope; the printed quadratic form matches only at b1 = 0 and
b1 = 1, where `b1² = b1`.

Anything derived from this row inherits the correction — the `D_lν` boundary term
of eq 19a, the `J_l^{(p)}` families, and (for a finite arc) the azimuthal row that
the full turn never forms.

Eq (18) (rows l = r, φ, z):

    Ĥ_lν(α) = D_lν(α)|_0^α + (r δ_lr + 2γ δ_lz) 𝒥1(α)
             − b1²/a0³ (b1 r δ_lr − 2 r1 δ_lz) 𝒥2(α)
             + J_lν^{(1)}(α) + J_lν^{(2)}(α) + J_lν^{(3)}(α)

Eq (19a):

    D_lν(α) = r/4 arsinh β1(α) [sin 4α; cos 4α; 0]
            + δ_lz r cos 2α arctan β3(α)
            + 1/a0³ arsinh β2(α) [ b1(r1 sin 2α − b1² r/4 sin 4α) ;
                                    b1(r1 cos 2α − b1² r/4 cos 4α) ;
                                    (a0² + b1²) r sin 2α ]

Full-turn boundary: `D_rν|_0^{π/2} = 0` (every r-row term carries sin 2α or
sin 4α);
`D_zν|_0^{π/2} = −(π r/2) [ sgn(γ(r1 − r)) + sgn(γ(r1 + r)) ]`.

Eqs (19b–19d) integrands (rows l = r, φ, z):

    J^{(1)} = 2a/a0² ∫_0^u du { −1 + (2+k²) sn² − 2k² sn⁴ ;
                                 2 sn cn (1 − k² sn²) ;  −b1(1 − k² sn²) }
    J^{(2)} = 1/a ∫_0^u du [ b c²/r − 2(a² − r'²) sn² + 4 r r' sn⁴ ;
                             −2 sn cn ((c² + r r') − 2 r r' sn²) ;
                             2γ(−b + 2r sn²) ]
            − 1/(4ar) Σ_p (−1)^p R_l(n_p) ∫_0^u du (1 − n_p² sn²)^{−1/2·[sic: −1]}
              [1; n_p² sn cn; 1]/(1 − n_p² sn²)
    J^{(3)} = 1/(a0²a) ∫_0^u du [ t_{1r} + 2b1 t_{2r} sn² − 4b1² r r' sn⁴ ;
                                  2b1 sn cn (t_{1φ} − 2b1 r r' sn²) ;
                                  (2/b1)(t_{1z} − 2b1 r (a0² γ + b1 r') sn²) ]
            + 1/(4a0³b1²) Σ_p (−1)^p S_l(m_p) ∫_0^u du [1; m_p² sn cn; 1]/(1 − m_p² sn²)

(The printed (19c/19d) exponent on `(1 − η² sn²u)` reads `−1/2` on the row
brackets; the integrated forms (22b/22c) and the third-kind definition (23)
fix the intended meaning: rows r and z are `1/(1 − η² sn²)` integrals →
π(α, η², k), row φ is the `sn cn/(1 − η² sn²)` integral → I(η²).)

Eq (20a):

    t_{1r} = r'(r' + r1) − b1²(a² − r' b) ,
    t_{1φ} = t_{2r} − b1 r r' ,
    t_{1z} = d² + r' h − a0²(r1 b − r r') ,
    t_{2r} = b1 a² − r'(γ + b1 r')

Eq (20b):

    R_l(n_p) = [r' − (−1)^p c] { n_p² γ² c/r ;  (−1)^p (c² + γ²) ;  −2 n_p² γ c }
    S_l(m_p) = 1/(a r d) { m_p² [L1 + (−1)^p a0 d L2] ;
                           b1² t0 [L3 + (−1)^p a0 d r'] ;
                           (2/b1) m_p² [L4 − (−1)^p a0 d L5] }

with `n_p² = 2r/(r ∓ c)`, `m_p² = 2b1²r/(g ∓ a0 d)` (upper sign p = 1) and

    t0 = 2f² + r(g − 3r1)
    L1 = −b1² f² t_{1r} + g L2
    L2 = 2g t_{1r} − (b1/r)(r² + r1² + 2rg) t_{2r} + 2b1² r r'(g − r1)
    L3 = r'(2g + a0² r') − b1² a²
    L4 = d² [g h + a0² r' r1 (a0² − 2)]
    L5 = r' g h + r1 d² (a0² − 2)

Eqs (22a–22c), integrated (case A; at α = π/2 all complete):

    J_lν^{(1)} = 2a/a0² { −El0 + (2+k²) El2 − 2k² El4 ;
                          2k^{−2} [I1(x) − I3(x)] ;
                          −b1 [El0 − k² El2] }
    J_lν^{(2)} = 1/a [ b c²/r El0 − 2(a² − r'²) El2 + 4 r r' El4 ;
                       −2k^{−2} [(c² + r r') I1(x) − a²/2 I3(x)] ;
                       2γ (−b El0 + 2r El2) ]
               − 1/(4ar) Σ_p (−1)^p R_l(n_p) { π(α,n_p²,k); I(n_p²); π(α,n_p²,k) }
    J_lν^{(3)} = 1/(a0²a) [ t_{1r} El0 + 2b1 t_{2r} El2 − 4b1² r r' El4 ;
                            2b1 k^{−2} [t_{1φ} I1(x) − b1 a²/2 I3(x)] ;
                            (2/b1)[t_{1z} El0 − 2b1 r (a0² γ + b1 r') El2] ]
               + 1/(4a0³b1²) Σ_p (−1)^p S_l(m_p) { π(α,m_p²,k); I(m_p²); π(α,m_p²,k) }

## 8. Exact complements and factorisations (derived, for conditioning)

These are algebraic identities of the printed symbols (not in the paper);
they carry the small quantities exactly and are what a floating-point
implementation must use at slender aspect:

    G²(α) = γ² (1 − n1² x)(1 − n2² x) ,   x = sin²α ,
      n1² = 2r/(r − c) < 0 ,  n2² = 2r/(r + c) → 1⁻ ,
      1 − n2² = γ²/(r + c)² ,   1 − n1² = (r + c)²/γ² ≥ 1  [times sign bookkeeping]
      (exactly:  1 − n1² = (c + r)/(c − r) = (c + r)²/γ²)

    B²(α) = f² (1 − m1² x)(1 − m2² x) ,
      m1² = −2r(a0 d + g)/f² < 0 ,   m2² = 2r(a0 d − g)/f² → 1⁻ ,
      (a0 d − g)(a0 d + g) = b1² f²   [d² a0² − g² = b1² (r + r1)²]
      1 − m2² = (a0 r − d)²/f² = (r − r1)²(r + r1)²/((a0 r + d)² f²)
              = [(r − r1)(r + r1) / ((a0 r + d) f)]²    →  use (r−r1)/(a0 r + d) · f/f
      1 − m1² = (a0 r + d)²/f²

    k'² = 1 − k² = (γ² + (r − r')²)/a²   with  r − r' formed as
      (r − ra) + b1 (za − z) − b1 γ  (no large-number cancellation).

    Confluence structure at slender aspect ε = section/R0:
      k'² ~ ε², 1 − n2² ~ ε², 1 − m2² ~ ε², and n2² − m2² ~ ε² differences.
      The printed split NEVER divides by n2² − m2²: eq (14c) carries only the
      G² pole pair, eq (14d) only the B² pole pair.  A reduction that expands
      the arctan-β3 term over the joint {n1,n2,m1,m2} pole set (as the
      re-derivation in `polygonanalytic.py` does) acquires partial-fraction
      weights 1/(n2² − m2²) that are absent from the paper's formulation.

## 9. Full-turn assembly and normalisation

Per edge ν and limit γ ∈ {u_ν2, u_ν1}, with the audit definition
`W(γ) = 4 ∫_0^{π/2} dα (−cos 2α) g(γ, α)` (the same integral the shipped
quadrature performs, in the transformed angle):

    W(γ) = 2 Â_φν(π/2; γ)     [eq 14 chain]      — verified numerically
    W_l(γ) = 4 ∫_0^{π/2} dα h_l(γ, α) = 2 Ĥ_lν(π/2; γ)   (l = r, z)

Flux:  ψ(r, z) = norm · r · Σ_ν weight_ν · (−[W(u_ν2) − W(u_ν1)]) / 2,
`norm` from `nova.biot.polygon.pack_section` (folds orientation, μ0, area,
2πR and half-turn doubling).  Field per (4a):
`B_l = sign · μ0/(4π·area) · Σ_ν (−2)[Ĥ_lν(π/2)]_{ν1}^{ν2}` with the same
orientation sign — pinned against `polygon_greens` in the tests.

## 10. Errata (Appendix B, p. 1179–1180)

The printed corrections apply to Parts II–IV only; Part V's own equations are
not corrected.  Two general notes DO apply to Part V's reading:
1. "All integration constants should be interpreted as
   `[mathematical expressions given]|_0^α`" — i.e. every antiderivative-style
   recursion (24a/24b) and constant (14a/19a) is a two-point evaluation.
2. Part V cites the same reduction conventions as [1]–[4]; the companion
   papers are not present in the references directory, but nothing in the
   full-turn chain requires them: the only external pieces are the standard
   recursions (24a/24b) and the third-kind integrals (23), all self-contained
   above.

## 11. Degenerate branches

* Horizontal edge (Δz = 0): contributes nothing (eq 7a); handled by edge
  weights in `pack_section`.
* Vertical edge (b1 = 0): eq (14b) integrand carries an overall b1 → I^{(1)} = 0.
  Eq (14c) is b1-free.  Eq (14d)'s INTEGRAND is regular at b1 = 0 (B² becomes
  linear in x: the m2 pole survives with m² = 4 r r1/f², the m1 pole
  disappears), but the PACKAGED coefficients (16c, 21c) divide by b1³/b1⁵ and
  by m_p² ~ b1², so the printed packaging is 0/0 at b1 = 0.  The paper's own
  remedy (p. 1173): "For b1 = 0 we immediately regain the corresponding
  expressions of Part III."  Implementation: a dedicated b1 = 0 reduction of
  the same (14d) integrand (single pole, degree-4 numerator), same for the
  field rows — algebraically the Part III full-turn expressions.
* Target level with an edge end (γ = 0): k², n_p², m_p² all remain regular
  (c → r keeps 1 − n2² = γ²/(r+c)² → 0: the G pole reaches the range end).
  γ = 0 makes W odd-in-γ terms vanish; the C boundary sgn(γ·…) dead-band is
  the same one the rectangle kernel carries.
* Target on axis (r = 0): k² = 0, all moments trivial; the φ-row brackets
  carry overall r factors.  (The flux itself carries r·W → 0; the FIELD
  z-component stays finite through Ĥ_z.)

## 12. Term-by-term audit protocol (what the tests pin)

Every closed-form object above is pinned against a converged quadrature of
the specific printed integral it claims to equal, per edge, per limit, at
fat (aspect ~1/3) AND slender (~1/30, ~1/100) geometry:

  * 𝒥1, 𝒥2         vs eq (14e) quadrature,
  * I^{(1),(2),(3)}_φ(π/2) vs eqs (14b–14d) quadrature (φ rows),
  * C_φν|_0^{π/2}    vs eq (14a) endpoint algebra (analytic),
  * Â_φν(π/2)        vs ½·alpha_quadrature (eq 10b, whole edge integrand),
  * J^{(1),(2),(3)}_{r,z}(π/2) vs eqs (19b–19d) quadrature,
  * Ĥ_{r,z}ν(π/2)    vs ½·alpha_field_quadrature (eq 11b),
  * B_φ structural zero vs eq (11b) φ row quadrature,
  * assembled ψ, B_R, B_Z vs the boundary-quadrature oracle over whole
    sections (the CANDIDATES gate in tests/test_biotpolygonanalytic.py).
