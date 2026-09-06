# third-light: implementation design

Multiphysics simulator for solid-state Tesla coils (SSTC, DRSSTC, QCW-DRSSTC).
Python front end, CUDA back end (CuPy + Numba CUDA), NumPy/Numba CPU fallback.

## 1. Scope

Inputs: coil geometry, top load, primary, bridge topology, semiconductor
datasheet parameters, driver settings (phase lead, interrupter/MIDI program,
QCW ramp), bus supply. Outputs per run: resonant frequencies and modes,
coupling, primary/secondary/top-load waveforms, streamer length and load,
IGBT loss and junction temperature, capacitor and wire heating, spark audio.
Batch mode evaluates thousands of design variants in parallel on one GPU for
sweeps and optimisation.

Fidelity target: match the existing quasi-static tool chain (JavaTC, tssp,
acmi) on inductance, capacitance and resonant frequency to about 1 %, match
SPICE on the lumped circuit to numerical precision, and match published spark
length versus power data within the spread of that data.

Out of scope: full-wave 3D Maxwell solves (the coil is electrically small,
height/wavelength < 0.1), spark-gap coils except as a validation case, and
detailed plasma chemistry.

## 2. Physics domains

| Domain | Model | Primary references |
|---|---|---|
| Magnetostatics | Filament/ring mutual inductance by complete elliptic integrals (Maxwell), assembled into dense L matrix over secondary sections, primary turns and top-load ring | acmi [4], Knight [10] |
| Electrostatics | Axisymmetric method of moments: ring charges on secondary, top load, breakout point, ground-plane images; potential coefficient matrix P, Maxwell capacitance C = P^-1. The winding former is an equivalent bound surface charge on its own boundary, radiating in the same vacuum Green's function | tssp [3], Medhurst [9], Voitkāns [7] |
| Conductor loss | Skin and proximity AC resistance per section from the exact cylinder diffusion solution (Butterworth [27]), the proximity term scaled to Medhurst's measured phi [9]; tank capacitor ESR; form dielectric loss | [27], [11], [9] |
| Secondary dynamics | N-section coupled L/C ladder from the matrices above; modal reduction to M eigenmodes for time domain; full ladder for voltage-profile studies | tssp [3], Voitkāns [7], Denicolai [1] |
| Primary circuit + bridge | Half/full bridge as piecewise-linear switch states; Vce(sat) = V0 + r·I; body/anti-parallel diode; dead time; DC bus C with rectifier ripple | Denicolai [1], de Queiroz [5], [6] |
| Driver/control | Primary current transformer feedback, phase-lead (UD2.x style), comparator + gate delay, interrupter (pulse width, PRF, MIDI note → PRF), QCW bus ramp or phase-shift modulation | Ward [12], Loneoceans [13], Kaizer [14], Burnett [15] |
| Streamer load | Fritz lumped model: R_s ≈ 220 kΩ in series with ~1–2 pF per foot of streamer, length ℓ(t) evolves with top voltage and per-bang energy; QCW channel persistence | Fritz [8], Freau law [16] |
| Breakout/corona | Surface field from the MoM solve; onset via Peek-type critical field with surface factor | Peek [17] |
| Streamer geometry (phase 6) | Dielectric breakdown model, growth probability ∝ φ^η, fast Laplacian growth on GPU; segment charges feed back into C matrix | NPW [18], Kim et al. [19] |
| Semiconductor thermal | E_on(I,Tj), E_off(I,Tj), E_rr polynomial fits from datasheet; conduction ∫Vce·I dt; Foster or Cauer rungs from junction to case to sink to ambient, exact-exponential between bursts | [20], [21] |
| Acoustics | Thermoacoustic simple source: the far-field pressure is the time derivative of the channel dissipation, delayed and 1/r | Teraguchi [22] |
| Coil pairs | Two coils solved on their own axisymmetric problems and coupled at their top electrodes alone: each electrode a point charge at its charge-weighted centroid, one mutual potential coefficient with its ground-plane image, and the two-node Maxwell capacitance the coupled modes come out of | Maxwell coefficients as §3.1, tssp [3], Voitkāns [7] |

## 3. Numerical core

### 3.1 Geometry → matrices (once per design)

1. Discretise secondary into N sections (N = 50–400), each a ring at radius
   a_k, height z_k, carrying n_k turns. Primary: one ring per turn (flat spiral,
   helical, conical). Top load: toroid or sphere surface as rings; breakout
   point as a small sphere on the same node. A conductor enters the self terms
   only through the self geometric mean distance of its cross-section, so a
   rectangular primary strap is the round wire of radius exp(-3/2)(w + t)
   (Rosa [28]), exact as t/w -> 0 and 0.25 % low at worst; which side is w and which
   is t is only which one lies along the winding axis.
2. L matrix: L_ij = n_i n_j M_ring(a_i, a_j, |z_i − z_j|) using
   M = μ0 √(a_i a_j) [(2/κ − κ) K(κ) − (2/κ) E(κ)]; diagonal via Lyle/Rayleigh
   self-inductance of a finite-width ring. K, E via AGM in a Numba CUDA
   kernel; O(N²) elements per design, batched over designs on a 2D grid.
3. P matrix: ring-charge potential coefficients (elliptic integral K), image
   rings for the ground plane, optional grounded strike ring. A dielectric former
   enters as bound band charges b on its boundary, satisfying
   sigma_b = 2 eps0 lam E_n with lam = (eps_r - 1)/(eps_r + 1) and E_n the
   principal-value normal field of every charge; eliminating b leaves
   P_eff = P_cc + P_cb (I - G A F_bb)^-1 G A F_bc, so nothing downstream of P
   changes. The self term of the normal-field operator is fixed by Gauss's law
   rather than dropped: a unit charge on a closed surface sends 1/(2 eps0)
   through it, which pins the area-weighted column sums and hence the diagonal,
   and is worth an order of convergence. Point-to-band integrals use a sinh
   grading (Johnston-Elliott), without which the wire sitting one wire radius off
   the former wall makes P indefinite. C = P^-1 by Cholesky (P is SPD) in float64
   via cuSOLVER (`cupy.linalg`), batched.
   Medhurst's C_L is *not* this matrix: it is the lumped equivalent of the
   resonating coil, and at l/D = 1, D = 10 cm his 4.6 pF is below the 5.56 pF of
   the inscribed sphere, so no static capacitance can equal it. The static
   uniform-potential extraction is checked against the sphere and toroid instead;
   Medhurst is a cross-check on f_res after step 5.
4. R vector: per-section AC resistance at f_res from [11]; iterated once after
   the first eigen-solve because R depends on f.
5. Eigen-solve of the ladder (generalised problem with L and C) →
   f_res, mode shapes v_m(z), effective inductance/capacitance per mode, k
   between primary and mode 1. Keep M = 4–16 modes for time domain.

### 3.1a Phase-1 residuals and what closed them

The air-only model ran f_res 2–9 % above Medhurst's C_L across l/D = 1..5,
one-signed. Medhurst wound every coil on a solid polystyrene rod, eps_r = 2.56
(Knight [10] §4), and the bound-charge former of §3.1 step 3 moves f_res by
−9.3, −5.3, −3.4 and −1.9 % at l/D = 1, 2, 3 and 5: the right sign and the right
size, closing the residual at l/D = 1–2 and overshooting at l/D >= 3. That the
overshoot grows with l/D is consistent with the published Medhurst table in
circulation being up to 8.8 % above Medhurst's own regression for l/D >= 2.5, so
part of that column is transcription rather than physics. The dielectric operator
itself is validated where a closed form exists — a conducting sphere inside a
concentric dielectric shell, every interface dielectric-to-vacuum — where its own
contribution to the error is below 1e-4 and the residual is the conductor
discretisation the air model already carried.

Against the thirteen bare air-cored coils tssp publishes with measurements [3],
f1 lands at 3.7 % rms and is unbiased, mean +0.05 %, against tssp's own 2.2 % rms
on the same coils; the twenty-three published overtones f3..f13 land at 1.4 % rms
and 3.8 % worst. That f1 is the weak mode and the overtones are not points at what
neither model carries: tssp images a ground plane of the coil's own height in
radius where this model images an infinite one, and neither carries the formers,
whose material tssp publishes for no coil. Higher modes hold their charge along
the winding rather than at its ends, so they see little of either. The two coils
tssp itself misses worst, sk16b55 at -3.7 % and sk20b49 at -5.0 %, this model
misses by -3.5 % and -4.7 %, which puts that residual in the measurements or in
the published geometry rather than in either model. Denicolai's Thor, measured at
80.22 mH, comes out at 79.35 mH from a filament sum over its 939 turns.

Winding AC resistance is exact for skin effect and for proximity in a uniform
transverse field, but that formulation omits the neighbouring turns'
eddy-current reaction and holds the field factor at its infinite-solenoid value
u = pi^2. Its high-frequency excess over the straight wire is then pi^2 (d/s)^2/2,
which is +0.6 %, +16 % and +74 % above Medhurst's measured phi at d/s = 0.2, 0.5
and 1.0, and carries no l/D dependence at all, over which his phi spans 23 % at
d/s = 0.8. Extending the theory would be circular — his d/s <= 0.3 cells are
themselves Butterworth's — so the proximity term is instead scaled by the ratio
of his measured excess to the model's, interpolated over his Table VIII in d/s
and in l/D compactified as x/(1+x). The scaling is a constant per coil, exact in
the asymptotic regime he tabulated and immaterial below it, where the proximity
term is O(q^4); it absorbs both the eddy reaction and the finite-length field
factor. R_ac then reproduces Table VIII to 7e-5 over every measured cell, and the
close-wound infinite solenoid falls from Butterworth's 5.94 without the reaction
field to Medhurst's 3.41 with it.

Mode 1 converges to 0.1 % by 200 sections. Higher modes converge more slowly —
the 4th is still 6 % short of the uniform-line 1:3:5:7 ratio at 400 sections —
which sets the section count needed for the M = 4..16 modal reduction of §3.3.

### 3.2 Time domain: piecewise-LTI exponential integrator

State x = [i_p, i_1..i_M, v_Cp, v_1..v_M], with v_bus last when the bus is a
reservoir rather than stiff, and thermal states to come in phase 4. Input
u = [drop, i_load, v_supply]. The modal equivalents l_m, c_m are referred to the
top node, so the modes are C-orthogonal and L is an arrowhead matrix — primary
self, modal self, and the mutuals k_m √(L_p l_m) — while a load current at the
top node forces every mode identically:

    L di/dt = e_p (σ g v_bus + drop) − R_σ i − v
    dv_Cp/dt = i_p / C_p,   dv_m/dt = (i_m − i_load) / c_m
    C_bus dv_bus/dt = (v_supply − v_bus) / R_s − σ g i_p

with g the bridge's swing per unit bus voltage, 1 full and 1/2 half. A stiff bus
drops the last row and carries σ g v_supply as an input column instead. Each
bridge configuration σ ∈ {+V, −V, freewheel, open} and each diode
conduction state gives a constant (A_σ, B_σ); the conducting device's
differential resistance sits in R_σ and its constant drop in u, and σ appears in
both A and B, so the stack is five: an IGBT or a diode conducting at either
polarity, and the blocked bridge. Instead of tabulating propagators
at a fixed step and its binary subdivisions, each A_σ is diagonalised once per
design:

    A_σ = V_σ Λ_σ V_σ^-1
    Φ_σ(t) = V_σ e^{Λ_σ t} V_σ^-1
    Γ_σ(t) = V_σ Λ_σ^-1 (e^{Λ_σ t} − I) V_σ^-1 B_σ   (rows with Λ ≈ 0 use t)
    x_{n+1} = Φ_σ(h) x_n + Γ_σ(h) u_n

The propagator is then available at *arbitrary* t for one complex diagonal
scaling and two n×n matvecs. Switching events (zero-crossings of the phase-led
feedback signal plus gate delay, dead time, interrupter edges, diode turn-on
and turn-off) are located by interpolating the relevant linear functional of x
within the step and propagating by the exact sub-step, so event timing is
exact rather than quantised to h/2^J; a step containing an event costs two
propagations instead of one plus J. No re-factorisation is ever needed inside a
run. h = T_res/256 (≈ 13 ns at 300 kHz).

Storage per design is one complex n×n pair (V_σ, V_σ^-1) and one complex n
eigenvalue vector per switch state, against (J+1) real n×n propagators for the
tabulated scheme: about 5× less at J = 8, and independent of the event-timing
accuracy demanded.

RLC bridge states are diagonalisable in practice but A_σ can be defective or
near-defective. The decomposition is accepted only when cond(V_σ) is below a
threshold; otherwise that state falls back to a scaling-and-squaring Padé
propagator tabulated at h and h/2^j, j = 1..32, composed by the semigroup rule on
the binary expansion of the requested sub-step, rounded to nearest so a dyadic
step is exact. Both paths are checked against `scipy.linalg.expm`. The two are
not equally conditioned: in raw [i, v] coordinates 1/c_m spans eleven decades and
the composed Padé path sheds three digits to cancellation where the diagonalised
path holds 1e-12, so a state that falls back wants the energy-normalised scaling
`secondary.py` already applies to the modes.

A crossing search evaluates one functional at many sub-steps of an interval
whose state and held input do not move, so the state's coordinates in the
eigenbasis and the functional's row against it are constant across the search:
y = V^-1 x and w = c^T V are formed once and every trial after that is a
diagonal scaling and a dot, O(n) against the O(n^2) of propagating the state and
then taking the functional of it. On the CPU that is worth about 1 %, because the
loop is interpreter bound rather than matvec bound: a 5.7x larger state costs
only 21 % more time, so the flops the reformulation removes were never what the
run was spending. It is the whole inner cost of the batched kernel of §3.3
instead, where there is no interpreter.

The bracket a functional standing on its own zero needs is noise limited at the
bottom. Its value at a small s is reconstructed from terms of the size of the
state, so the cancellation that leaves it near zero floors its relative accuracy,
and below the s at which the true value falls under that floor a search can enter
spuriously and report a crossing there rather than none at all. That costs
nothing in the case the bracket exists for, where the state part of the
functional is exactly zero because a commutation pinned i_p to it.

Both ends of the sub-step are exact: Φ(0) = I and Γ(0) = 0 are returned as such
rather than reconstructed, because the event locator brackets a functional
against the value it started from and a switching instant pins that value to
zero. A functional standing on that zero and leaving it in the direction the new
conduction state admits is not crossing, though: the crossing due is the one it
comes back to, so the locator brackets past the departure. Reporting the instant
it is standing on returns a zero-length step, and the run then loops on it.

Which devices conduct is a complementarity problem, not a schedule. Both bridge
polarities present the same network to the tank, so only three linear circuits
exist: IGBT conducting, diode conducting, and the blocked bridge, where nothing
is forward biased and i_p is pinned to zero with the tank charge frozen. The
blocked state drops the primary row and column of L, leaving the modes ringing on
their own l_m. Leaving it is a two-candidate admissibility test: a polarity is
taken when the loop equation it implies gives a di_p/dt of that same sign, so the
diode dead zone at |v_loop| < n_dev V0 and the reverse conduction driven by the
modes' induced EMF both fall out rather than being cased on.

Nonlinear branches (saturating Vce, corona) enter as current injections
evaluated from x_n with their own small explicit ODE, and the input column
u[1] carries any load the caller supplies. The streamer is not one of them.
Its R_s C_s is 0.7 µs at a metre but 0.3 ns at the length where the branch first
matters, so a held injection would be unstable exactly where a channel starts,
and the damping it applies to the modes would depend on h — which is the very
quantity a spark-length fit reads. It is one more row of the state space
instead, and C_s, which moves on the bang timescale rather than the RF one, is
quantised geometrically: the propagators are rebuilt when it has drifted 2 %, a
few hundred times in a bang against 10^4 steps, and quartering the quantisation
moves the settled length by 4e-4. Charge is what carries across a level change
while the channel grows, and voltage while it cools and takes the charge of the
recombined part away with it; both directions release energy, so a change can
never create any.

Slow inputs (QCW bus ramp, MIDI PRF schedule) are zero-order-held per step.

Justification: the circuit is linear except at switch instants and the top
node, so a matrix-exponential scheme is both cheaper and more accurate than
generic stiff ODE integration; it also removes the trapezoidal ringing SPICE
shows on hard switching. Diagonalising rather than tabulating keeps that
exactness at the event instants, where the tabulated scheme is only accurate
to h/2^J.

### 3.3 GPU execution model

Two paths, same algorithm:

* Single design, full ladder (n ≈ 2N+8 ≈ 800): CuPy dense matvec per step;
  memory-bound, ~1 ms of coil time per second of GPU time at 300 kHz.
* Batch of B designs, modal model (n ≤ 32): one thread per design, with the
  design indexing the last axis of every packed array, so the threads of a warp
  read consecutive addresses of the same matrix element and every load
  coalesces. The per-design eigenbasis for S = 5 switch states is 2·S·n²
  complex64 = 80 KB — far over any shared-memory budget, so it stays in global
  memory and streams through L1/L2, with only the state vector and the S·n
  eigenvalues resident per design. B = 10^4 designs is then 0.8 GB of basis,
  which sets the practical batch size; wider sweeps are chunked. The
  tabulated-propagator alternative would need several times that and would not
  fit. B = 10^4 designs × 10 ms QCW burst ≈ 10 s on a mid-range GPU.

  An earlier draft put one warp on each design and one lane on each state row,
  reducing the matvec by warp shuffle, in order to coalesce those basis reads.
  Two things retired it. The event locator of §3.2 works in eigen coordinates,
  so the basis is touched twice per interval rather than once per bisection
  trial, and the traffic the warp layout existed to smooth is no longer the
  inner cost. And a design-major layout coalesces just as well with one thread
  per design, wastes no lanes on a state shorter than a warp, and needs no
  shuffle — which is what lets the CPU and CUDA paths compile from one source,
  so the CPU path validates the exact algorithm the GPU runs. With no GPU in CI
  that verifiability decides it.

  Complex data is carried as separate real and imaginary float arrays rather
  than a complex dtype. The same source then runs in float64 or float32 for the
  precision gate below, and asks neither compiler for complex support.

Thermal states use their own exact-exponential update between bursts, because
their time constants are 10^3–10^6 times the electrical ones. Per-burst energies
enter it as the mean power of each of a burst's sub-intervals rather than as one
impulse: §3.6 refines that, since a burst comparable with the fastest rung is not
resolved by a single one and the subdivision costs a propagation per window.

Elliptic integrals, potential coefficients, event stepping and DBM growth are
written as scalar and flat-array functions compiled by `numba.njit` for the CPU
and `numba.cuda.jit` for the GPU from one source; dense linear algebra
(Cholesky, eigen-solve, matvec) goes through an array-namespace handle `xp`
bound to NumPy/SciPy or CuPy. The two mechanisms are deliberately separate:
Numba's CPU and CUDA targets do not share a namespace, so kernels are
dispatched per backend and only the library-level linear algebra is
namespace-generic.

"One source" is a build rather than a decorator, because device code cannot call
an `@njit` dispatcher: compiling a device function that calls one, or that calls
another device function by dispatcher reference, fails before typing. The kernel
bodies are therefore left undecorated and call their siblings by bare name, and a
builder rebinds each onto a namespace carrying that target's siblings compiled so
far, so a target's call tree closes over itself. The CPU build keeps its on-disk
cache through the rebinding, which matters because the batched kernel costs 38 s
to compile cold and 7.7 s warm; the rebound functions must carry the module's own
`__name__` and `__file__` for that, or the first run writes a cache the next one
cannot load. The handful of names with no common spelling are bound the same way:
`nextafter` is NumPy's on the CPU and libdevice's on the GPU, neither being
callable from the other's target.

Precision: float64 for matrix assembly, inversion and eigen-solve; float32
optional for stepping, gated by a conservation-of-energy check on a lossless test
circuit. P conditioning is benign — cond(P) grows linearly at about 3.3 per ring,
reaching only 1.4e3 at N = 400 — so Cholesky retains twelve digits. The real
failure mode is geometric: P loses positive definiteness when rings overlap
(conductor radius above the ring spacing), which surfaces as a Cholesky error
rather than silent error.

### 3.4 Breakout and streamer length dynamics

The surface field is linear in the state the integrator already carries, so no
part of the MoM is re-entered per step. Ring potentials are the modal shapes
scaled by the modal top-node voltages, ring charges are the potential solve
applied to those, and Gauss at a conductor turns charge into field; the chain
collapses to one (electrode rings x modes) matrix built once per design, against
which Peek's threshold is evaluated per ring at its own component's curvature.
The breakout point is a sphere on the end of a stalk, tied to the top node with
the top load; the stalk carries far less field than the tip and is left out.

That field needs one correction. A sphere's polar band is a disc rather than a
ring -- its radius is half its width at every section count -- so the ring model
puts 10.6 % too little charge on it, and refining the sphere only makes a smaller
cap of the same shape. An isolated sphere's field is uniform and known, so the
sphere's own solve calibrates the error, which is local: applied to a breakout
point mounted over a coil the corrected coarse pole field lands within 0.02 % of
the refined solution, against 10.6 % low without it. A toroid's bands are all
slender and need nothing.

Per bang: a channel starts when the electrode surface field reaches the Peek
threshold, and once one exists its own tip carries the field, so growth then
continues on the top voltage alone. ℓ grows at a rate proportional to the excess
of that voltage over the gradient E ℓ the channel needs to sustain itself, and
decays with a channel-cooling time constant, so the PRF dependence is what the
duty cycle leaves rather than a parameter of its own. Both regimes of (|v| − Eℓ)+
are linear, so a step is one exponential rather than a sub-stepped integration.
The load is Fritz R_s + C_s(ℓ) on the top node, carried as a state of the
piecewise-linear system rather than as an injection, for the reasons in §3.2.
Phase 6 replaces the ℓ scalar with the DBM tree and derives C_s from segment
charges.

### 3.4a Phase-3 residuals, and what the published data does and does not pin

No coiler source states a spark-length law. Freau's 1.7 in/√P [16] is a
spark-gap result, and it appears in JavaTC's documentation in that context;
Ward's design guide [12] and Kaizer's [14] carry no spark-length formula at all.
What the builders do publish is coils: input power and measured spark length
together, from which k = L/√P is 1.2 to 2.1 in/√W across the DRSSTCs that
publish both [12], [13], [14], with two systematic caveats. Ward's figures are
wattmeter readings at the outlet, on a coil he separately measured at 0.64 power
factor, while Kaizer's and Slawinski's are V·I products, so part of the spread is
watts against VA. And the small coils saturate against their own winding length:
Ward puts racing sparks at about three times it, and the coils furthest below the
band are the ones nearest that limit.

Within one coil the law is steeper than √P. Ward's DRSSTC-0.5 table [12], the one
published set that varies power at fixed geometry over a useful range, runs
0.254 m at 33 W to 0.457 m at 180 W: an exponent of 0.341, with k falling 1.74 to
1.34. This model's sweep of the example machine predicts 0.270, and its own k
falls 2.03 to 1.19 over 194 W to 2.0 kW, so every point from 194 W to 1.3 kW lies
inside the published band, and the 2.0 kW point lies just under it at a spark 2.7
times the winding length, where a real coil of that size would be racing. Nothing
in the length dynamics puts that exponent there: it comes out of the circuit,
where the channel's own loading holds the top voltage back as the bus rises.

The two constants §3.4 nominates as the fitted ones are not the ones that matter.
A DRSSTC bang is long enough for the channel to reach the length its top voltage
sustains, so the model is ceiling limited at V/E: raising the growth gain 7.5x
moves the settled length by 5 %, quadrupling the cooling time moves it by 4 %,
and a least-squares fit against the band pushes both to their bounds without
improving the residual. Both are therefore set at their physical scales — the
gain at a channel velocity of 2.8e5 m/s at 700 kV, inside the published leader
range, the cooling time at the channel's own 0.5 ms — and the constant that sets
the answer is the sustaining gradient, which is not fitted. At the 5 kV/cm of a
cold positive streamer the model runs a factor of two short of every measured
coil; at the 1.5 to 3 kV/cm coilers measure across long Tesla sparks, whose
channels are thermalised leaders rather than cold streamers, it lands in the
band. 2 kV/cm is the value used.

One double count is known and left standing: Fritz's 220 kΩ is the channel's own
resistance, so i R_s already accounts for part of the gradient along it and the
E ℓ term accounts for it again. At the operating point the two are comparable,
which is one reason the gradient that lands in the band sits below the cold
streamer value. Phase 6 removes the question by growing the channel
geometrically instead.

### 3.4b Filament electrostatics of the discharge tree

The channel is a set of straight segments, each carrying its total charge spread
uniformly along it, so its potential is the closed form
ln((R1 + R2 + L)/(R1 + R2 − L)) / (4 π ε0 L) in the distances to the two
endpoints. The symmetric form is used rather than the projection form because
nothing in it cancels away from the wire. The singular diagonal is the same
expression on the wire surface at the segment midpoint, where R1 = R2;
rationalising its denominator gives the algebraically identical
2 ln((√(L² + 4 r_w²) + L) / (2 r_w)) / (4 π ε0 L), which holds full precision at
any aspect ratio where the difference form loses ten digits at r_w/L = 1e−5, and
which tends to 2 ln(L/r_w) with no special case. Ground is the image segment of
opposite charge, exactly as it is the image ring for the rings.

The tree and the electrode share one potential coefficient matrix ordered
[rings, segments]. The ring-ring block is §3.1's own, unchanged; the
segment-segment block is the filament kernel point matched at segment midpoints;
and the ring-segment block needs nothing new, because the ring potential is
axisymmetric and depends on a three-dimensional field point only through
(hypot(x, y), z). The Green's function is reciprocal, so one triangle and one
off-diagonal block are evaluated and the rest is their transpose, which leaves
the matrix symmetric and Cholesky-solvable like the ring matrix alone.

Holding every ring and every segment at unit potential — the tree is an
equipotential for the electrostatic problem, the drop along it being the
reduction below — the total charge is the loaded system's capacitance, and the
channel's contribution is what it adds, C(rings + tree) − C(rings). That is what
replaces Fritz's 1 pF/ft, a per-unit-length constant that knows nothing about
the electrode the channel hangs off; a 1 m channel on the example toroid comes
out at twice it.

The resistance comes from the same charges rather than from a lumped 220 kΩ.
Segment k has R_k = ρ L_k / A over the channel cross section and carries the
charging current of everything at and below it, so the series resistance
dissipating that distribution's power is Σ_k R_k (q_k/Q)². The subtree charges
q_k need no path walk and no ancestor matrix: growth appends nodes, so every
parent has a lower index than its child and one reverse scatter-add over
segments accumulates every subtree sum in O(n).

### 3.5 Loss extraction

Conduction loss is already in the state space and comes back out of it by
subtraction. The primary loop resistance of a bridge state is tank.resistance
plus n_dev r of the device conducting in it, and tank.resistance is itself the
rest of the loop plus the tank capacitor's ESR, DF/(omega C) at the driven
resonance, so the device differential, the capacitor and the loop separate
analytically instead of being re-derived; the modal rows split the same way into
the winding resistance and the series equivalent tan d/(omega c_m) of the
former's dielectric conductance. The constant part of the device drop is the
input column u[0] the run already recorded, so n_dev v0 |i_p| needs nothing
either. Every weight is a function of the recorded bridge state alone, which
makes the ledger a post-pass over a Result: IGBT and diode conduction, primary
loop, tank ESR, per-mode winding, former dielectric and streamer channel, whose
total is Result.dissipation to rounding -- the quantity the burst energy balance
of §5 already validates.

Switching energy is not separable that way because it is not in there at all.
The piecewise-linear state space carries no transition dynamics, so a
commutation costs nothing in it, and adding a switching term to that balance
would open the ledger rather than close it. It is attributed afterwards, at the
instants the event stepper resolved exactly, and reported beside the total
rather than inside it. E_on, E_off and E_rr are low-order polynomials in current
with a linear junction-temperature coefficient, scaled against the datasheet's
test voltage by (V/V_test)^Kv: the app-note form of [20] and [21]. At Kv = 1,
the default, that is the closed form of a linear voltage/current transition
overlap; [20] quotes 1.3 to 1.4 for an IGBT and about 0.6 for a diode's E_rr,
so the exponent is a parameter of the fit rather than fixed.

The bridge polarity is what the attribution reads. A change of conducting kind
at one polarity is the handover between IGBTs and the diodes gated across them,
which happens at the current zero those IGBTs already stand at and costs
nothing. A change of polarity is the hard commutation: the IGBT leaving
conduction turns off the current it was carrying, and the one entering takes the
current off the opposite leg's diodes against the whole bus and recovers them.
Every device in series with the tank commutates together, two in a full bridge
and one in a half, and each blocks the bus itself, so the swing gain does not
enter. Under ZCS both fall on the current zero and cost nothing; a phase lead
turns off into current and turns on soft, and a gate delay does the reverse,
which is the asymmetry that kills the IGBTs of a lagging DRSSTC.

Junction temperature is an argument with a default, not a state of the run. The
default applies each fit at its own test temperature, extrapolating nothing; §3.6
carries the networks that produce a real one and passes it back in. It is one
temperature for every fit, the junction the outer loop tracks, so a diode
recovering at its own junction is evaluated at the IGBT's.

### 3.6 Thermal networks and junction temperature

The two forms of thermal impedance in circulation are one state space. A Cauer
ladder's rungs are physical nodes: capacitance i is node i's heat capacity and
resistance i carries heat to the next node, the last onto whatever the branch
stands on. A Foster chain's rungs are the fitted Zth a datasheet publishes: each
R parallel C carries the whole flow, so its states are temperature differences
and only their sum is a rise. Both assemble through one primitive -- a rung is an
edge between two terminals, adding g (e − f)(e − f)^T to G, Cauer between
successive nodes and Foster onto the reference -- which leaves
C dT/dt = −G T + S p as the only equation and a Foster-to-Cauer synthesis
unnecessary. A branch's terminal is one row vector doing double duty: it reads
the temperature at the top of the branch and, transposed, spreads a flow entering
there, so the injection matrix and the observation matrix are the same S and
reciprocity is structural rather than checked. Two dies over the case, sink and
ambient path of their own module is then a group of branches on a shared path,
and the coil and the tank capacitor are groups of one. An energy E into a port is
the state jump C^-1 S E, and the port temperature is ambient + S^T T.

Propagation is §3.2's machinery over intervals 10^3 to 10^6 times longer:
A = −C^-1 G is diagonalised once and evaluated at each interval a cycle needs. A
burst enters as the energy of each of its sub-intervals, held as a mean power
across the sub-interval rather than lumped at its start, which is the exact
solution for the power it stands for and the impulse in the limit. One impulse
per burst does not resolve a burst comparable with the fastest rung -- 150 µs
against a die's 0.7 ms -- and the subdivision removes that for one propagation
per window. The windows are cut at equal steps measured against the burst's own
end rather than the run's, so a boundary lands where the junction peaks instead
of the energy being averaged across it; that alignment is worth more than an
order of magnitude of window count. Their energies are §3.5's ledger evaluated on
the slices and not a second arithmetic: consecutive windows share their boundary
sample, so every interval closes in exactly one and every commutation is
attributed to exactly one, and the windows sum to the run. The default of 16 is
the smallest count whose settled peak is within 1e-4 of its own rise of the four
times finer one: it lands at 2.2e-5 where 8 misses at 1.3e-4, 4 at 1.9e-4 and one
impulse at 2.5e-3, and the whole subdivision costs 20 propagations of a 13-state
network.

A repeated interrupter cycle is an affine map T_end = Φ T_start + q, Φ the
product of its segment propagators, so the settled cycle is the fixed point
(I − Φ)^-1 q -- solved, not iterated to: the sink's own time constant is 10^4
cycles of the example machine, which is what iterating would have to run through.
The cycle mean falls out in closed form and needs no quadrature at all, since a
settled cycle returns the heat it stored, leaving G times the mean state equal to
S times the mean power: the mean is the DC state at the mean power. The peak is
what decides whether a die survives, and it is the ripple that puts it there --
the example settles at a 176 C mean junction with a 44 K swing across the
cycle, so its die peaks at 209 C on 144 W of the 264 W it dissipates.

Temperature re-enters the losses twice: each switching fit carries its own
coefficient, and the winding's resistivity sets R_dc and hence the modal Q, so
the network is rebuilt through `from_modes` off the eigen-solve the machine
already holds, for the cost of one resistance sweep. That makes an outer fixed
point around the closed-form one, one burst per pass at the last pass's junction
and winding temperature; the diode and the tank capacitor are outputs of it
rather than inputs. Its loop gain is not small -- at the example's lagging driver
a kelvin at the junction returns 0.38 of one, so plain iteration needs seven
passes to 1e-3 where the secant through the last two passes' residuals needs four
-- and a gain of one or more is thermal runaway, which has no fixed point at all
and is reported as unconverged rather than iterated over. A channel seeded from
the previous burst rides the same loop, which is §3.4's own fixed point.

What has no network is stated rather than hidden: the primary loop resistance is
busbar and the streamer channel is in the air, so neither enters one. No
published worked example of a junction-temperature calculation was reproducible
here -- [20], whose switching example §3.5 does reproduce, stops at total device
loss and publishes no Zth rungs -- so every check in §5 is analytic or
self-consistent, and none of it is fitted.

### 3.7 Spark acoustics

A spark channel is a heat release in air, and in linear acoustics that is a
volume source and nothing else. Heat at rate P into a gas of ratio gamma at
ambient p0 displaces V = (gamma - 1) E / (gamma p0) with E the integral of P --
the constant-pressure expansion of an ideal gas, (gamma - 1) / gamma of the heat
doing the work p0 dV and the rest raising the internal energy -- and a simple
source of volume V radiates p = rho0 Vddot(t - r/c) / (4 pi r). With
c^2 = gamma p0 / rho0 the two collapse to

    p(r, t) = (gamma - 1) Pdot(t - r/c) / (4 pi c^2 r)

so the radiated pressure is the time derivative of the channel dissipation §3.5
already carries, delayed by the propagation time and falling as 1/r. Ambient air
and the run's own `streamer_power` are the only inputs; there is no constant left
in it to fit.

The run is event-stepped, so the source is resampled onto the audio grid through
its own energy: each sample is the mean power over the interval it covers, taken
as the difference of the energy interpolated at the sample edges. That energy is
§3.5's own trapezoid, so the resampling creates no heat and loses none, and the
mean is a boxcar whose nulls sit at every multiple of the sample rate, which is
what keeps the carrier ripple at twice the resonant frequency -- ultrasound, and
absorbed within centimetres of the channel -- from folding into the band rather
than being decimated out of it. What survives is the burst envelope, which the
grid resolves. Pdot is then a centred difference of that, the composite stencil
(P_k+2 + 2 P_k+1 - 2 P_k-1 - P_k-2) / 8h, which deviates from the derivative by
5 h^2 P'''(t) / 12: local and second order, where a spectral derivative would
impose on a burst a periodicity it has not got.

A MIDI program is 10^4 bangs, so the placement of them is one scatter-add and not
a loop. The burst starts are the schedule's own edges that leave the gate enabled
-- the array §3.2 already synchronises on -- their offsets are rint(t * rate),
and the whole train is one `bincount` of the signature broadcast over every
offset at once. A fixed PRF then puts the spectrum exactly on the PRF comb and a
`Melody` puts each note span on its own, which is what §5 checks. Output is
normalised 16-bit mono PCM through the standard library's `wave`.

### 3.8 Two coils standing side by side

A pair of towers is not one axisymmetric problem, but at the separations one is
built at it is two: neither tower perturbs the other's own solve, so the pair is
the two single-coil solves of §3.1 plus one coupling term, taken at the top node
because that is where essentially all of a grounded quarter-wave resonator's
charge sits.

The top electrode reduces to a point charge at the charge-weighted centroid
height of its own unit-potential solve. Two such charges at separation s and
heights h_a, h_b have the mutual potential coefficient

    p12 = (1/sqrt(s^2 + (h_a - h_b)^2) - 1/sqrt(s^2 + (h_a + h_b)^2)) / 4 pi eps0

whose second term is the ground plane's image of the other electrode, screening
the coupling by s/2h where there is a plane and absent where there is not. That
is the leading order in electrode size over separation. It omits the electrodes'
own multipoles, second order in a/s against it, and it carries no inductive
coupling between the two windings at all: the near field of a grounded
quarter-wave resonator is dominantly electric at the top, and the windings'
mutual inductance is far weaker at these separations than the electrodes' mutual
capacitance.

P = [[1/c_a, p12], [p12, 1/c_b]] over the two coils' mode-1 top-referred modal
capacitances closes the network. C = P^-1 is the Maxwell capacitance matrix, its
off-diagonal the mutual capacitance -C[0,1], and with the modal inductances to
ground the pair's two resonances are diag(1/l_a, 1/l_b) v = omega^2 C v. Scaled
by diag(sqrt(l_a), sqrt(l_b)) that is a standard symmetric problem, whose
eigenvectors are orthonormal in the tower basis and so measure directly how much
of each mode sits on each tower, where the generalised solve's vectors are
normalised to unit modal energy instead.

Two numbers place a pair: the detune |f_a - f_b| over the mean of the two
isolated frequencies, and the coupling -C[0,1] / sqrt(C_00 C_11), which for
identical coils is the fractional mode splitting to leading order. The splitting
is the two-level sqrt(detune^2 + coupling^2), so the mixing angle turns on their
ratio alone: the modes delocalise over both towers once the coupling exceeds the
detune and localise one to a tower once it does not, which is the locking
criterion, and the participation ratio 1/sum(v^4) crosses 4/3 at that threshold.

The in-phase mode drives no current through the mutual capacitance, but it does
not sit at the isolated f0 either: two electrodes held at the same potential
screen one another, so the in-phase branch of C^-1 is c/(1 + c p12) rather than
c and the mode lifts above f0 by as much as the antiphase mode falls below it, to
first order. The lumped picture that holds each branch capacitance fixed misses
that screening; the two agree on the splitting, which is what a pair is tuned by.

A pair driven in antiphase bridges the sum of the two coils' reaches to ground,
those being the settled lengths of §3.4: the gap sees the sum of the two terminal
voltages, so each channel need only cover its own coil's reach.

## 4. Package layout

```
thirdlight/
  backend.py       array-namespace handle and njit/cuda.jit kernel dispatch
  geometry.py      coil, primary, top-load and ground descriptions; discretisation
  em/inductance.py ring mutual/self inductance kernels, L matrix
  em/capacitance.py ring-charge MoM, P/C matrices, surface field
  em/losses.py     AC resistance, ESR, dielectric loss
  secondary.py     ladder assembly, eigen-solve, modal reduction
  circuit/         bridge, tank, bus, diode/IGBT companion models, state-space builder
  control/         feedback CT, phase lead, comparator/delay, interrupter, MIDI, QCW ramp
  discharge/       breakout, Fritz load, length dynamics, DBM (phase 3)
  thermal/         loss extraction; Foster and Cauer networks, junction temperature
  solver/          expm precompute, event stepping, CUDA and CPU kernels
  batch.py         design-space expansion, sweep runner, Optuna/scipy objective glue
  acoustics.py     thermoacoustic simple source, burst rendering, WAV output
  pair.py          two coils side by side: electrode coupling and the coupled modes
  io/              YAML design schema round trip, xarray/parquet output
  viz/             waveform, mode-shape, field and streamer plots
```

Backend selection: `THIRDLIGHT_BACKEND=cuda|cpu`, default auto.

Dependencies: numpy, scipy, numba, cupy-cuda12x (extra `[cuda]`), xarray,
pyyaml, matplotlib; optional optuna; dev: pytest, pytest-xdist, pytest-cov,
black, pylint, PySpice (ngspice) for circuit cross-checks.

## 5. Validation suite

| Check | Reference | Tolerance |
|---|---|---|
| Single-ring and coaxial-loop inductance | closed form | 1e-10 rel |
| Segment self GMD | ln g = ln w - 3/2, `scipy.integrate.quad` of the double integral | 1e-12 abs |
| Rectangle self GMD | `scipy.integrate.dblquad` of ln g = <ln\|r - r'\|> over the section | Rosa 0.25 % low at worst, 0.18-0.21 % at t/w = 1, 1/2, 1/10 |
| Solenoid inductance | Wheeler, acmi published examples | 1 % |
| Isolated sphere and toroid capacitance | closed form; Kelvin image series for a sphere over a plane | 0.25/N, 1 % |
| Dielectric-coated sphere capacitance | closed form for a conducting sphere in a concentric shell | 1 %, dielectric operator alone below 1e-4 |
| Bound-charge field operator | Gauss's law: area-weighted column sums of F_bb are 1/(2 eps0) | 1e-12 rel |
| Solenoid f_res | Medhurst C_L via the eigen-solve | see §3.1a |
| Solenoid f_res | tssp measured air-cored coils [3] | f1 within 4 % rms, overtones within 2 % rms |
| Solenoid inductance | Denicolai's measured 80.22 mH on Thor [1] | 1.5 %, the derived-geometry spread |
| Coupling k | acmi | 1 % |
| Peek critical field | uniform-field limit as the electrode grows | 0.1 % |
| Sphere surface field | uniform on an isolated sphere, once corrected | 1e-9 rel |
| Electrode field functional | a direct potential solve at the same modal state | 1e-11 rel |
| Lumped 4th-order DRSSTC transient | every constant-mode interval against `scipy.integrate.solve_ivp` DOP853 at rtol 1e-13 | 1e-9 rel |
| Complete energy transfer | de Queiroz integer mode ratios [5]; achievable only when n - m is odd, so (3,2) and (5,4), not (5,3) | 1e-6 of the initial energy |
| Phase-lead/ZCS behaviour | gate edges on the current zero crossings with no delay; tau = tan(omega t_d)/omega restores them | 5 % of the gate delay |
| Winding AC resistance | Butterworth's Tables I and II [27] for the uncorrected model | 5e-4 |
| Winding AC resistance | Medhurst's Table VIII over every measured d/s and l/D [9] | 1e-3 |
| Unloaded secondary Q | Denicolai measured 326 at 65.6 kHz [1]; Kaizer tabulations [14] | within the published band |
| Switching-energy fit | the closed form of a linear transition overlap, integral v i dt | 1e-12 rel |
| Conduction loss | closed form for a sinusoid through v0 + r i over whole half cycles | 1e-6 rel, the trapezoid's own order |
| Component loss ledger | its own total against Result.dissipation | 1e-12 rel |
| Commutation attribution | the gate and state history: instant, sign and count of every event | exact |
| Switching-energy scaling | the blocking-voltage power law and the temperature coefficient reproduced | 1e-14 rel |
| IGBT switching loss | the worked example of the Renesas app note [20] §5, its inverter averaging supplied by the test | 5 %, lands at 0.2 % |
| Event locator | a functional leaving its own zero crosses half a period on | 1e-9 rel |
| Streamer length ODE | `solve_ivp` DOP853, both regimes of (|v| - E l)+ | 1e-9 rel |
| Streamer branch | `solve_ivp` DOP853 on the augmented state space | 1e-9 rel |
| Streamer branch equations | dv_m/dt = (i_m - i_s)/c_m per mode, dv_s/dt = i_s/C_s | 1e-12 rel |
| Burst energy ledger | bus energy in against dissipation and storage | 1e-4, first order in h |
| Channel capacitance levels | settled length against 10x finer quantisation | 5e-3 |
| Spark length vs power | published DRSSTC k = 1.2..2.1 in/sqrt(W) [12], [13], [14] | inside the band, §3.4a |
| Filament potential | `scipy.integrate.quad` of the point-charge integral along the segment | 1e-10 rel, lands at 4e-16 |
| Distant filament | the point charge 1/(4 pi eps0 d), second order in L/d | 4.2e-4, 4.2e-6, 4.2e-8 at d/L = 10, 100, 1000 |
| Filament self term | 2 ln(L / rw) in the thin-wire limit | (rw/L)^2 rel |
| Mixed ring/segment matrix | its own transpose | 1e-12 rel, exact by construction |
| Filament image | zero potential on the z = 0 plane | 1e-12 rel, exact |
| Closed polygon of filaments | thin-torus closed form, second order in 1/N | error quartered per doubling, 2.2e-4 at N = 128 |
| Straight channel capacitance | prolate spheroid closed form | 10 %, the cylinder-spheroid shape difference; lands at 5.1 % |
| Straight channel refinement | its own value at twice the segment count | 1e-2, lands at 1.3e-3 |
| Series resistance reduction | sum R_k for a chain charged at its tip; R_t + R_b/2 for a symmetric fork | 1e-12 rel |
| Subtree charges | a naive per-node ancestor walk on a random tree | exact |
| Propagator Φ_σ(t), Γ_σ(t) | `scipy.linalg.expm` of the augmented matrix | 1e-12 rel |
| Energy conservation | lossless circuit, float32 stepping | 1e-4 over 10^6 steps |
| Foster step response | its own closed form Zth(t) = sum R (1 - exp(-t/tau)) | 1e-12 rel |
| Cauer ladder | `scipy.integrate.solve_ivp` DOP853 at rtol 1e-13 | 1e-9 rel |
| One-rung network | the Foster and Cauer assemblies of the same rung | 1e-12 rel |
| Thermal DC limit | Ohm's law sum(R) P, through a branch and through the path two dies share | 1e-12 rel |
| Thermal energy conservation | heat in against what the capacities hold and what left by the ambient node, `expm` of the augmented block | 1e-12 rel |
| Cycle-mean temperature | the same augmented `expm` integrated over the settled cycle | 1e-9 rel |
| Periodic steady state | direct iteration of the cycle map to convergence | 1e-9 rel |
| Burst subdivision | the settled peak against four times finer; a network slow against the burst against one impulse | 1e-4 of the rise; 1e-5 rel |
| Window energies | the windows of a run against the ledger of the whole | 1e-12 rel |
| Temperature feedback | the fixed point reached from ambient and from 200 C | 1 % |
| Bus reservoir | first-order charging against its own R_s C_bus; energy traded with the circuit exactly | 1e-8, 1e-9 |
| Design-schema round trip | `from_dict(to_dict(d))` against the design, over every component shape | exact |
| Labelled run output | every dataset and frame series against the `Result` property it names | exact |
| Plot content | every artist's data against the array it stands for | exact |
| Sweep expansion | the axis product and order, and the frame's own unstacked cube | exact |
| Sweep observables | each against the built network it is read back off | exact |
| Objective | the machine built by hand at the same point; a rejected point is a wall | exact |
| Batched stepper | every design's observables and final state against `simulate` | 1e-9, lands at 1e-12 |
| Batched interval count | the intervals each design takes, against `simulate`'s own | exact |
| Batched burst edges | the analytic edge index against `Interrupter.edges` | exact |
| Batched packing | every design the model does not carry rejected, by cause | exact |
| Per-target build | each kernel compiled once, in dependency order, seeing its own siblings | exact |
| Device build | every kernel of the CUDA build constructs, which needs no device | exact |
| Thermoacoustic volume | the ideal gas heated at constant pressure, m R dT / p0 | 1e-12 rel |
| Inverse distance, retarded time | half the pressure at twice the range; the arrival shifted by r/c on the grid | 1e-12 rel; exact |
| Radiated pressure | the closed-form composite stencil of the resampling and the centred difference, for a Gaussian pulse | 1e-12 of the terms it differences |
| Differentiation truncation | its own leading 5 h^2 P'''/12 against the analytic derivative | 2 %, lands at 0.03 % |
| Source resampling | the audio samples' own sum against the ledger's channel integral | 1e-12 rel |
| Radiated acoustic energy | 2 pi^1.5 K^2 A^2 / (rho0 c s), the sphere integral of p^2 / (rho0 c) for a Gaussian pulse | 1.25 (h/s)^2, lands at 5.4e-4 |
| Burst spectrum | the DFT of an exact number of PRF periods is the PRF comb | off-comb below 1e-12 of it, lands at 7e-17 |
| Melody pitch | each note span's dominant bin against `note_frequency` | exact |
| Burst placement | the scatter-add against a per-burst loop, interrupter and melody | 1e-12 rel |
| Spark WAV | the samples read back through `wave` against the normalised waveform | the int16 step |
| Electrode centroid | the sphere's own centre with no ground plane, and below it with one | 1e-12 rel; the shift signed |
| Two-tower mutual capacitance | the far-field two-sphere 4 pi eps0 a_a a_b / s | second order in a/s: 6.7e-3, 6.7e-5, 6.7e-7 at s/a = 10, 100, 1000, on that order to 1 % |
| Ground-plane screening | strictly weaker coupling with a plane; the free-space limit far above it | first order in s/2h, to (s/2h)^2 |
| Identical-coil split | f0 sqrt(1 +- c p12), and f_anti = f_in / sqrt(1 + 2 c_mutual / c_ground) | 1e-12 rel, lands at rounding |
| Uncoupled pair | C exactly diagonal at p12 = 0, and the two isolated frequencies | exact; 1e-15 through the eigen-solve's sqrt round trip |
| Avoided crossing | sqrt(detune^2 + coupling^2) over a detune sweep at fixed coupling | second order in the coupling: 2e-6 at 0.002, 2e-4 at 0.02 |
| Mode localisation | the participation ratio's 4/3 crossing against the locking criterion | agree outside 1e-3 of the threshold |
| Pair symmetry and monotonicity | the split under a swap of the two coils; mutual capacitance and splitting against separation | 1e-12 rel; monotone |

## 6. Engineering constraints

* Any test or example script completes under 60 s CPU; GPU tests are marked
  and skipped without a device. Coverage > 85 % is achieved on the CPU path.
* black, pylint (no unused imports/vars), pytest-xdist `-n auto`.
* CI runs in Docker; multistage image: CUDA runtime base → Python deps →
  source. GitHub-hosted runners exercise the CPU path; a self-hosted GPU
  runner (or manual workflow) runs the CUDA suite.
* Dependabot for pip and GitHub Actions.
* Fixtures: measured coil data cached from public sources with attribution;
  no copyrighted datasheets committed, only fitted coefficients.

## 7. Roadmap

0. Repo scaffold, CI, Docker, schema, backend switch. Done.
1. EM matrices, eigen-solve, validation against acmi/Wheeler/Medhurst. Done.
1a. Dielectric former in the MoM, and validation against air-cored measured
   coils (tssp, Denicolai), to close the Medhurst f_res residual; Medhurst Φ
   interpolation for close-wound AC resistance. Done; both are documented in
   §3.1a.
2. Circuit + driver + exponential integrator, SSTC and DRSSTC, SPICE parity.
   Done, with `solve_ivp` in place of ngspice as the independent reference: it
   needs no optional native dependency in CI and is checked on every interval
   rather than on one waveform.
3. Streamer load and length dynamics, breakout, spark-length calibration. Done;
   the residuals and what the published data pins are in §3.4a.
4. Thermal and loss models. Done; the QCW bus ramp and the MIDI interrupter came
   with the driver in phase 2, so what was left here was the thermal side.
4a. Switching-energy device fits, commutation attribution and the
   component-resolved loss ledger. Done; §3.5. Validated on the analytic and
   self-consistent references of §5, and on [20]'s own worked example, whose
   inverter averaging the test supplies because this model has no equivalent
   of it.
4b. Foster and Cauer junction networks, the junction-temperature state and the
   between-burst thermal update, consuming 4a's per-component energies. Done;
   §3.6. Both impedance forms assemble into one state space rather than one
   being synthesised into the other, the settled interrupter cycle is solved in
   closed form rather than iterated to, and the loss/temperature loop is a
   secant on its own residual. No published worked example of a junction
   temperature was cleanly reproducible, so its validation is analytic and
   self-consistent throughout.
4c. Design-schema round trip, labelled xarray/parquet output of a run, and the
   plotting layer. Done. A sweep varies the spec mapping and records the variant,
   so what round trips is the design rather than a built machine, which has
   already resolved its tuning and its phase lead and carries a network no
   mapping describes. JavaTC import moves to phase 6: it is a browser-form tool
   whose saved format no public source documents, and a parser guessed at it
   could not be validated against anything.
5a. Design-space expansion, the sweep runner and the optimiser glue. Done. An
   axis moves the geometry the modes come out of, so every point rebuilds the
   machine, where the drive sweep of §3.4a reuses the one network it was built
   with; the sweep frame is indexed by its axes, so it unstacks into a labelled
   cube for nothing. Infeasibility is part of a design space rather than an
   error: a breakout point inside the top load and rings overlapping until P is
   indefinite are alike a NaN row, and a wall an optimiser walks away from. The
   objective is a plain callable over a vector, which is all either scipy or
   Optuna needs, so neither is imported.
5b. Batched stepping: a design space packed design-major and stepped by one
   kernel with no Python in the interval loop. Done, on both targets. What the
   batched model does not carry is rejected at pack time rather than
   approximated: a streamer, whose channel capacitance re-levels mid-run and
   rebuilds the propagators; a caller's load callback; a MIDI schedule, which is
   array data where a plain interrupter is four scalars; and any switch state
   whose propagator fell back to Pade, which has no eigenbasis to pack. What no
   machine without a GPU can check is checked anyway: the CPU build, through the
   identical per-target mechanism of §3.3, reproduces `simulate` bit for bit and
   to 1e-12 against its observables, and every kernel of the device build
   constructs. What is left to the `cuda` job is the PTX compile and the launch,
   since compiling a device call tree needs a driver.
6. DBM streamer geometry, acoustics, JavaTC import, 3D visualisation.
7. Two coils side by side. Done; §3.8. A pair is two single-coil solves and one
   potential coefficient rather than a new geometry, because at the separations
   a pair is built at neither tower perturbs the other's own axisymmetric
   problem, and the detune and the coupling it reduces to are what say whether
   the two towers lock and how far the antiphase pair reaches.

## 8. Existing tools surveyed

| Tool | What it does | Reuse |
|---|---|---|
| JavaTC (Anderson) | Full quasi-static coil design incl. mutual inductance, Medhurst, toroid C, tuning | input format, benchmark values |
| TeslaMap (Wilson) | Fast SGTC design calculator | benchmark values |
| tssp (Nicholson et al.) | Precision self-resonant solenoid model, distributed L/C, measured validation | method, validation data |
| acmi (Nicholson) | Air-core mutual inductance for concentric windings | method, validation data |
| MANDK (Denicolai) | Lumped Tesla transformer model plus measured coil | validation data |
| drsstcd / mrn (de Queiroz) | Multiple-resonance network design and simulation | mode-ratio validation |
| LTspice / ngspice + PySpice | Lumped circuit reference | validation |
| openEMS (Liebig) | EC-FDTD field solver, Python API | optional full-wave cross-check of top-load field |
| FEMM | 2D axisymmetric FEM, widely used by coilers | optional cross-check |
| NVIDIA Warp | GPU kernel DSL | evaluated; Numba CUDA chosen for NumPy compatibility and CPU fallback |

## 9. References

1. M. Denicolai, *Tesla Transformer for Experimentation and Research*, Licentiate thesis, Helsinki University of Technology, 2001. https://research.aalto.fi/en/publications/tesla-transformer-for-experimentation-and-research/
2. M. Denicolai, "Optimal performance for Tesla transformers," *Rev. Sci. Instrum.* 73(9), 2002.
3. P. Nicholson et al., Tesla Secondary Simulation Project. http://www.abelian.org/tssp/
4. P. Nicholson, acmi: Air Core Mutual Inductances. http://abelian.org/acmi/
5. A. C. M. de Queiroz, "Multiple resonance networks," *IEEE Trans. Circuits Syst. I* 49(2), 240–244, 2002; "Designing an ideal double resonance solid-state Tesla coil." https://www.coe.ufrj.br/~acmq/tesla/drsstc.html
6. A. C. M. de Queiroz, "Generalized LC multiple resonance networks." https://www.semanticscholar.org/paper/0aa234e2e59b44554bcb8032436aa5b44e24db4f
7. J. Voitkāns, A. Voitkāns, "Tesla Coil Theoretical Model and its Experimental Verification," *Electrical, Control and Communication Engineering* 7(1), 11–19, 2014. https://ecce-journals.rtu.lv/ecce/article/view/ecce-2014-0018
8. T. Fritz, streamer load model (220 kΩ + ~1 pF/ft), Tesla Coil Mailing List, 2002–2005. https://www.pupman.com/listarchives/2005/Feb/msg00064.html
9. R. G. Medhurst, "H.F. Resistance and Self-Capacitance of Single-Layer Solenoids," *Wireless Engineer*, Feb./Mar. 1947.
10. D. W. Knight, "The self-resonance and self-capacitance of solenoid coils" and "An introduction to the art of solenoid inductance calculation." https://hamwaves.com/inductance/doc/knight.p1.pdf
11. E. Fraga, C. Prados, D.-X. Chen, "Practical Model and Calculation of AC Resistance of Long Solenoids," *IEEE Trans. Magn.* 34(1), 205–212, 1998. An equivalent-tube model, not a skin/proximity split; not obtainable open access, so [27] is implemented instead.
12. S. Ward, "A General Guide to DRSSTC Design"; Universal Driver UD2.x. https://www.stevehv.4hv.org/drsstc_design.htm , https://github.com/WaskaLabs/Universal_Driver_29_X
13. Loneoceans Laboratories, UD2.7C driver and ramped/QCW SSTC notes. https://loneoceans.com/labs/sales/ud27/index.htm , https://www.loneoceans.com/labs/sstc3/
14. Kaizer Power Electronics, DRSSTC design guide and phase-lead test notes. https://kaizerpowerelectronics.dk/tesla-coils/drsstc-design-guide/
15. R. Burnett, Tesla coil and SSTC operation notes. https://www.richieburnett.co.uk/
16. J. Freau, spark length law L[in] = 1.7 √P[W]; summarised in T. Paulin, *Formulas for Tesla Coils* v3.0. https://www.mv.helsinki.fi/home/tpaulin/FormulasForTeslaCoils.pdf
17. F. W. Peek, *Dielectric Phenomena in High Voltage Engineering*, 1915 (Peek's law). https://en.wikipedia.org/wiki/Peek%27s_law
18. L. Niemeyer, L. Pietronero, H. J. Wiesmann, "Fractal Dimension of Dielectric Breakdown," *Phys. Rev. Lett.* 52(12), 1033–1036, 1984. https://doi.org/10.1103/PhysRevLett.52.1033
19. T. Kim, J. Sewall, A. Sud, M. C. Lin, "Fast Simulation of Laplacian Growth," *IEEE CG&A* 27(2), 2007. http://gamma.cs.unc.edu/FRAC/laplacian_large.pdf
20. Renesas AN, "IGBT Loss Calculation"; ROHM AN, "Estimation of switching losses in IGBTs." https://www.renesas.com/en/document/apn/igbt-loss-calculation
21. IPST 2005, "Approximate Loss Formulae for Estimation of IGBT Switching Losses." https://www.ipstconf.org/papers/Proc_IPST2005/05IPST184.pdf
22. Teraguchi et al., "Development of musical solid-state Tesla coil based on pulse repetition frequency method," *Electron. Commun. Jpn.*, 2019. https://doi.org/10.1002/ecj.12196
23. "Tesla Transformer and its Response with Square Wave and Sinusoidal Excitations," *ACES Journal*. https://journals.riverpublishers.com/index.php/ACES/article/view/10411
24. "Simulation and Analysis of the Optimal Electric Field from Modifications to the Winding Design for the Tesla Transformer," *Energies* 18(2), 339, 2025 (FEM of secondary field). https://www.mdpi.com/1996-1073/18/2/339
25. openEMS. https://www.openems.de/ ; CuPy. https://docs.cupy.dev/ ; Numba CUDA. https://numba.readthedocs.io/ ; NVIDIA Warp. https://developer.nvidia.com/warp-python
26. JavaTC. http://www.classictesla.com/java/javatc.html ; TeslaMap. https://www.teslamap.com/
27. S. Butterworth, "Effective Resistance of Inductance Coils at Radio Frequency," *Experimental Wireless & The Wireless Engineer*, Apr./May 1926 (eq. 21, Tables I, II and IV). https://www.g6yb.com/g3ynh/zdocs/refs/
28. E. B. Rosa, "The Self and Mutual Inductances of Linear Conductors," *Bulletin of the Bureau of Standards* 4(2), 301-344, 1908 (geometric mean distances, §§2-3). https://nvlpubs.nist.gov/nistpubs/bulletin/04/nbsbulletinv4n2p301_A2b.pdf
