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
| Streamer geometry (phase 3) | Dielectric breakdown model, growth probability ∝ φ^η, fast Laplacian growth on GPU; segment charges feed back into C matrix | NPW [18], Kim et al. [19] |
| Semiconductor thermal | E_on(I,Tj), E_off(I,Tj), E_rr polynomial fits from datasheet; conduction ∫Vce·I dt; Cauer RC junction→case→sink→ambient | [20], [21] |
| Acoustics (phase 3) | Spark pulse train at interrupter PRF; SPL proxy from energy per bang | Teraguchi [22] |

## 3. Numerical core

### 3.1 Geometry → matrices (once per design)

1. Discretise secondary into N sections (N = 50–400), each a ring at radius
   a_k, height z_k, carrying n_k turns. Primary: one ring per turn (flat spiral,
   helical, conical). Top load: toroid surface as rings; breakout point as a
   thin-wire charge segment.
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

State x = [i_p, v_Cp, v_bus, modal q_m, q̇_m, i_lead, thermal states...].
Each bridge configuration σ ∈ {+V, −V, freewheel, open} and each diode
conduction state gives a constant (A_σ, B_σ). Instead of tabulating propagators
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
propagator tabulated at h and h/2^j, j = 1..8, with bisection on the event
time. Both paths are checked against `scipy.linalg.expm`.

Nonlinear branches (streamer, saturating Vce, corona) enter as current
injections evaluated from x_n with their own small explicit ODE. The streamer
R_sC_s time constant (≈ 0.4 µs) is well above h, so the explicit coupling is
stable; a single fixed-point corrector is available for stiff settings.

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
* Batch of B designs, modal model (n ≤ 32): one warp per design, one lane
  per state row, matvec by warp-shuffle reduction, per-design event handling
  without divergence across warps. The per-design eigenbasis for S ≈ 4 switch
  states is 2·S·n² complex64 = 64 KB — far over the shared-memory budget, so it
  stays in global memory and streams through L1/L2, with only the state vector
  and the S·n eigenvalues resident per warp. B = 10^4 designs is then 0.64 GB
  of basis, which sets the practical batch size; wider sweeps are chunked. The
  tabulated-propagator alternative would need 2.9 GB for the same batch and
  would not fit. B = 10^4 designs × 10 ms QCW burst ≈ 10 s on a mid-range GPU.

Thermal states use their own exact-exponential update between bursts, with
per-burst energies as impulses, because their time constants are 10^3–10^6
times the electrical ones.

Elliptic integrals, potential coefficients, event stepping and DBM growth are
written as scalar and flat-array functions compiled by `numba.njit` for the CPU
and `numba.cuda.jit` for the GPU from one source; dense linear algebra
(Cholesky, eigen-solve, matvec) goes through an array-namespace handle `xp`
bound to NumPy/SciPy or CuPy. The two mechanisms are deliberately separate:
Numba's CPU and CUDA targets do not share a namespace, so kernels are
dispatched per backend and only the library-level linear algebra is
namespace-generic.

Precision: float64 for matrix assembly, inversion and eigen-solve; float32
optional for stepping, gated by a conservation-of-energy check on a lossless test
circuit. P conditioning is benign — cond(P) grows linearly at about 3.3 per ring,
reaching only 1.4e3 at N = 400 — so Cholesky retains twelve digits. The real
failure mode is geometric: P loses positive definiteness when rings overlap
(conductor radius above the ring spacing), which surfaces as a Cholesky error
rather than silent error.

### 3.4 Streamer length dynamics

Per bang: breakout when the top-load surface field exceeds the Peek
threshold; while broken out, ℓ grows at a rate proportional to the excess of
top voltage over the channel-sustaining voltage, decays between bangs with a
channel-cooling time constant that decreases with PRF. Load is Fritz
R_s + C_s(ℓ) on the top node. The two constants (growth gain, cooling time)
are the only fitted parameters and are fitted once against the published
DRSSTC and QCW spark-length datasets and Freau's law for SGTC. Phase 3
replaces the ℓ scalar with the DBM tree and derives C_s from segment charges.

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
  thermal.py       loss extraction, Cauer networks
  solver/          expm precompute, event stepping, CUDA and CPU kernels
  batch.py         design-space expansion, sweep runner, Optuna/scipy objective glue
  io/              YAML design schema, JavaTC import, xarray/parquet output
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
| Solenoid inductance | Wheeler, acmi published examples | 1 % |
| Isolated sphere and toroid capacitance | closed form; Kelvin image series for a sphere over a plane | 0.25/N, 1 % |
| Dielectric-coated sphere capacitance | closed form for a conducting sphere in a concentric shell | 1 %, dielectric operator alone below 1e-4 |
| Bound-charge field operator | Gauss's law: area-weighted column sums of F_bb are 1/(2 eps0) | 1e-12 rel |
| Solenoid f_res | Medhurst C_L via the eigen-solve | see §3.1a |
| Solenoid f_res | tssp measured air-cored coils [3] | f1 within 4 % rms, overtones within 2 % rms |
| Solenoid inductance | Denicolai's measured 80.22 mH on Thor [1] | 1.5 %, the derived-geometry spread |
| Coupling k | acmi | 1 % |
| Lumped 4th-order DRSSTC transient | ngspice via PySpice; de Queiroz mode ratios 1:2:3, 1:3:5 [5] | numerical |
| Phase-lead/ZCS behaviour | UD2.x documented behaviour [12], Kaizer static tests [14] | qualitative + timing |
| Spark length vs power | Freau law (SGTC) [16]; published DRSSTC/QCW data [13], [14] | within data spread |
| Winding AC resistance | Butterworth's Tables I and II [27] for the uncorrected model | 5e-4 |
| Winding AC resistance | Medhurst's Table VIII over every measured d/s and l/D [9] | 1e-3 |
| Unloaded secondary Q | Denicolai measured 326 at 65.6 kHz [1]; Kaizer tabulations [14] | within the published band |
| IGBT loss | datasheet curves; PLECS/PSIM published examples [20] | 5 % |
| Propagator Φ_σ(t), Γ_σ(t) | `scipy.linalg.expm` of the augmented matrix | 1e-12 rel |
| Energy conservation | lossless circuit, float32 stepping | 1e-4 over 10^6 steps |

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

0. Repo scaffold, CI, Docker, schema, backend switch.
1. EM matrices, eigen-solve, validation against acmi/Wheeler/Medhurst.
1a. Dielectric former in the MoM, and validation against air-cored measured
   coils (tssp, Denicolai), to close the Medhurst f_res residual; Medhurst Φ
   interpolation for close-wound AC resistance. Done; both are documented in
   §3.1a.
2. Circuit + driver + exponential integrator, SSTC and DRSSTC, SPICE parity.
3. Streamer load and length dynamics, breakout, spark-length calibration.
4. Thermal and loss models, QCW modulation, MIDI interrupter.
5. Batched GPU sweeps and optimisation front end.
6. DBM streamer geometry, acoustics, JavaTC import, 3D visualisation.

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
