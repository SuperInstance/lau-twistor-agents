# lau-twistor-agents

**Penrose's twistor theory in Rust — encoding spacetime events as complex geometric objects, with agent trajectories.**

This crate implements the mathematical apparatus of twistor theory: 2-component spinors, twistor space (ℂ⁴ / projective ℂP³), incidence relations linking spacetime points to twistor lines, the Penrose transform mapping cohomology to massless fields, curved twistor spaces and the nonlinear graviton, the Ward correspondence for Yang-Mills fields, null geodesics and optical scalars, conformal group actions, and agent trajectories described twistially — all backed by **122 tests**.

---

## What This Does

Twistor theory rewrites spacetime physics in terms of complex geometry. Instead of working with real spacetime coordinates (t, x, y, z), you work with complex objects called *twistors* that live in ℂ⁴. The payoff:

- **Null geodesics** (light rays) become *points* in twistor space
- **Spacetime points** become *lines* (Riemann spheres) in twistor space
- **Massless field equations** become *cohomology* problems (the Penrose transform)
- **Self-dual Yang-Mills** becomes *holomorphic bundles* (the Ward correspondence)
- **Self-dual gravity** becomes *deformed complex structures* (the nonlinear graviton)

This crate gives you all of these constructions, plus an agent trajectory system that encodes agent dynamics as curves in twistor space.

---

## Key Idea

A **twistor** Z^α = (ω^A, π_{A'}) is an element of ℂ⁴ — two spinor parts glued together. The **incidence relation** ω^A = ix^{AA'}π_{A'} links a spacetime point x to a line of twistors. Projectively, twistors live in ℂP³, and the Hermitian norm Z·Z̄ splits this space into positive-frequency (PT⁺), negative-frequency (PT⁻), and null (PN) regions — which is exactly the positive/negative frequency splitting of quantum field theory.

---

## Install

```toml
[dependencies]
lau-twistor-agents = "0.1.0"
```

Requires **Rust 2021 edition**. Dependencies: `nalgebra` (with serde), `num-complex` (with serde), `num-traits`, `serde`.

---

## Quick Start

### Spinors and null vectors

```rust
use lau_twistor_agents::spinor::*;
use num_complex::Complex64;

// Create a spinor
let s = Spinor::new(Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0));

// SL(2,C) invariant inner product
let contraction = s.contract(&s); // = 0 for this spinor

// Convert null 4-vector to spinor pair and back
let v = [Complex64::new(5.0, 0.0), Complex64::new(3.0, 0.0),
         Complex64::new(4.0, 0.0), Complex64::new(0.0, 0.0)];
let (xi, pi) = null_vector_to_spinor(v).unwrap();
let recovered = spinor_to_null_vector(&xi, &pi);
```

### Twistors and incidence

```rust
use lau_twistor_agents::twistor::*;
use lau_twistor_agents::incidence::*;

// Create a spacetime point (t, x, y, z)
let point = SpacetimePoint::from_minkowski(1.0, 2.0, 3.0, 4.0);
println!("Null? {}", point.is_null());       // false
println!("Timelike? {}", point.is_timelike()); // true

// The incidence relation: for any π_{A'}, get the corresponding twistor
let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
let twistor = point.twistor_line(&pi);
assert!(point.contains_twistor(&twistor));
```

### Projective twistor space (ℂP³)

```rust
use lau_twistor_agents::projective::*;

let pt = ProjectiveTwistor::from_homogeneous(
    Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
    Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
);

// Inhomogeneous coordinates [Z⁰/Z³, Z¹/Z³, Z²/Z³]
let (w0, w1, w2) = pt.to_inhomogeneous().unwrap();

// Region classification: positive, negative, or null
println!("Region: {:?}", pt.region());
```

### Penrose transform

```rust
use lau_twistor_agents::penrose_transform::*;

// H¹(PT⁺, O(-6)) → Maxwell field (spin-1)
let transform = PenroseTransform::new(-6);
println!("Field type: {:?}", transform.field_type()); // MaxwellField

// Transform a twistor function to spacetime
let x = SpacetimePoint::origin();
let field = transform.transform_to_spacetime(
    &x,
    |z| Complex64::new(1.0, 0.0),
    20, // integration samples
);
```

### Massless fields and frequency splitting

```rust
use lau_twistor_agents::massless::*;

// Maxwell field with positive frequency
let maxwell = MasslessField::maxwell(HelicitySign::Positive);
println!("Spin: {}", maxwell.spin()); // 1.0

// Frequency splitting for helicity-2 (gravitational field)
let split = FrequencySplitting::new(4);
// split.positive → PT⁺ part, split.negative → PT⁻ part
```

### Conformal group

```rust
use lau_twistor_agents::conformal::*;

// The conformal group SU(2,2) acts linearly on twistor space
let dilation = ConformalGroup::dilation(Complex64::new(2.0, 0.0));
let translated = ConformalGroup::translation(&point);
let combined = dilation.compose(&translated);

// 15 generators of the conformal algebra
let generators = ConformalGenerator::all_generators();
assert_eq!(generators.len(), 15);
```

### Agent trajectories

```rust
use lau_twistor_agents::agent::*;

let z = Twistor::from_components(
    Complex64::new(5.0, 0.0), Complex64::new(0.0, 0.0),
    Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
);

let mut traj = AgentTrajectory::new();
traj.push(AgentState::new(z, 0.0));
traj.push(AgentState::new(z.scale(Complex64::new(2.0, 0.0)), 1.0));

// Twistorial velocity between states
let velocities = traj.twistorial_velocity();

// Resample at uniform intervals
let resampled = traj.resample(10);
```

---

## API Reference

### `spinor` — 2-Component Weyl Spinors

| Type / Function | Description |
|---|---|
| `Spinor` | 2-component Weyl spinor with conjugation, contraction, symmetric product, index lowering |
| `PrimedSpinor` | Dotted (conjugate) spinor |
| `null_vector_to_spinor` | Decompose a null 4-vector as ξ^A π_{A'} |
| `spinor_to_null_vector` | Reconstruct null vector from spinor pair |

### `twistor` — Twistor Space ℂ⁴

| Type / Function | Description |
|---|---|
| `Twistor` | Z^α = (ω^A, π_{A'}) with Hermitian norm, frequency classification, scaling, associated spacetime point |
| `InfinityTwistor` | The dual origin I^{αβ} for Minkowski space |

### `incidence` — Incidence Relations

| Type / Function | Description |
|---|---|
| `SpacetimePoint` | Minkowski point as Hermitian matrix x^{AA'} with interval classification |
| `IncidenceRelation` | The fundamental link ω^A = ix^{AA'}π_{A'} between points and twistor lines |

### `projective` — Projective Twistor Space ℂP³

| Type / Function | Description |
|---|---|
| `ProjectiveTwistor` | Equivalence class [Z] with region classification, normalization |
| `ProjectiveLine` | Line in ℂP³ (↔ spacetime point) with containment check |
| `ProjectivePlane` | Plane in ℂP³ with 4×4 determinant test |
| `TwistorRegion` | PT⁺, PT⁻, or PN |

### `penrose_transform` — Penrose Transform

| Type / Function | Description |
|---|---|
| `LineBundle` | O(k) on ℂP³ with tensor, dual, canonical |
| `PenroseTransform` | Maps H¹(PT⁺, O(k)) to spacetime fields via contour integral |
| `TwistorCohomologyClass` | Representative (0,1)-form on twistor space |
| `FieldType` | Scalar, Weyl spinor, Maxwell, gravity, higher spin |

### `massless` — Massless Fields

| Type / Function | Description |
|---|---|
| `MasslessField` | Zero-rest-mass field with helicity, frequency sign, spin |
| `FrequencySplitting` | Decompose into positive (PT⁺) and negative (PT⁻) frequency parts |
| `evaluate_field` | Evaluate a massless field at a spacetime point |

### `curved` — Curved Twistor Spaces

| Type / Function | Description |
|---|---|
| `CurvedTwistorSpace` | Deformed twistor space with parameterized deformation terms |
| `NonlinearGraviton` | Self-dual vacuum spacetime ↔ deformed twistor space, with Weyl spinor |
| `ComplexStructureDeformation` | (0,1)-vector field deformation of the complex structure |

### `ward` — Ward Correspondence

| Type / Function | Description |
|---|---|
| `HolomorphicBundle` | Vector bundle on twistor space with transition functions, direct sum, tensor, dual |
| `WardCorrespondence` | Holomorphic bundle ↔ (anti-)self-dual Yang-Mills, with instanton ansatz |
| `YangMillsConnection` | Gauge connection with field strength, gauge transforms |
| `FieldStrength` | Curvature of the Yang-Mills connection |

### `null_geodesic` — Null Geodesics

| Type / Function | Description |
|---|---|
| `NullGeodesic` | Light ray ↔ point in PT, with affine parameterization, celestial sphere |
| `NullCongruence` | Family of null geodesics with optical scalars (expansion, shear, twist) |

### `conformal` — Conformal Group

| Type / Function | Description |
|---|---|
| `ConformalGroup` | SU(2,2) element with Lorentz, translation, dilation, special conformal, composition |
| `ConformalGenerator` | All 15 basis generators of the conformal algebra |

### `agent` — Agent Trajectories

| Type / Function | Description |
|---|---|
| `AgentState` | Twistor + affine parameter, frequency classification |
| `AgentTrajectory` | Sequence of states with interpolation, velocity, resampling, frequency purity |
| `AgentInteraction` | Pair of trajectories with closest-approach computation |

---

## How It Works

1. **Spinor decomposition**: Null vectors factor as outer products of spinors — this is the entry point to twistor theory.
2. **Twistor construction**: Each twistor Z = (ω, π) encodes both position (ω) and momentum (π) information.
3. **Incidence**: A spacetime point x determines a line of twistors via ω = ixπ; conversely, a twistor determines a spacetime point (if non-null).
4. **Projective geometry**: Working in ℂP³ means Z and λZ are equivalent — this is the physical twistor space.
5. **Field equations**: The Penrose transform turns holomorphic data on twistor space into solutions of field equations on spacetime.
6. **Curvature**: Deforming the complex structure of twistor space encodes spacetime curvature (nonlinear graviton).
7. **Gauge theory**: Holomorphic bundles on twistor space correspond to Yang-Mills fields (Ward correspondence).

---

## The Math

### Spinor Formalism

In 2-spinor notation, a spacetime vector V^{AA'} is the symmetric outer product of a spinor and its conjugate:

```
V^{AA'} = ξ^A π̄^{A'}
```

The epsilon tensor ε_{AB} provides the SL(2,ℂ)-invariant contraction: `ε_{AB} ξ^A η^B` is the spinor inner product.

### Twistor Space

A twistor Z^α = (ω^A, π_{A'}) lives in ℂ⁴. The **Hermitian norm**:

```
Z·Z̄ = ω^A π̄_A + π_{A'} ω̄^{A'} = 2 Re(ω^0 π̄₀ + ω¹ π̄₁)
```

splits ℂP³ into:
- **PT⁺** (Z·Z̄ > 0): positive frequency
- **PT⁻** (Z·Z̄ < 0): negative frequency
- **PN** (Z·Z̄ = 0): null twistors ↔ real null geodesics

### Incidence Relation

The fundamental equation linking spacetime to twistor space:

```
ω^A = ix^{AA'} π_{A'}
```

For fixed x, varying π sweeps out a ℂP¹ (Riemann sphere) in PT — the "twistor line" of x. Two spacetime points are null-separated iff their twistor lines intersect.

### Penrose Transform

The Penrose transform is a contour integral:

```
φ_{A...B}(x) = ∮ f(Z) π_{A'} ... π_{B'} π^{C'} dπ_{C'}
```

where the integral is over the Riemann sphere (twistor line) through x, and f is a representative of H¹(PT⁺, O(k)). The line bundle weight determines the spin:

| Bundle | Cohomology | Spacetime field |
|---|---|---|
| O(−2) | H¹(PT⁺, O(−2)) | Scalar (spin 0) |
| O(−4) | H¹(PT⁺, O(−4)) | Left Weyl spinor (spin ½) |
| O(−6) | H¹(PT⁺, O(−6)) | Maxwell field (spin 1) |
| O(−8) | H¹(PT⁺, O(−8)) | Linearized gravity (spin 2) |

### Nonlinear Graviton

Penrose's nonlinear graviton construction: a self-dual vacuum solution of Einstein's equations corresponds to a deformation of the complex structure of PT. The deformed incidence relation is:

```
ω^A = ix^{AA'}π_{A'} + ε f(x, π) + O(ε²)
```

The self-dual Weyl spinor Ψ_{ABCD} is reconstructed from the deformation data.

### Ward Correspondence

Ward showed that (anti-)self-dual Yang-Mills fields on spacetime correspond to holomorphic vector bundles on twistor space. The transition functions of the bundle encode the gauge connection, and the splitting problem on each twistor line reconstructs the Yang-Mills potential.

### Conformal Group

The 15-parameter conformal group SU(2,2) acts linearly on ℂ⁴:
- 6 Lorentz generators (SL(2,ℂ) × SL(2,ℂ))
- 4 translations (ω → ω + ixπ)
- 4 special conformal transformations (π → π + ibω)
- 1 dilation (ω → λω)

---

## Test Coverage

**122 tests** across all modules:

| Module | Tests | What's covered |
|---|---|---|
| `spinor` | 16 | Creation, conjugation, contraction, index lowering, normalization, projective equivalence, null vector roundtrip |
| `twistor` | 11 | Construction, scaling, addition, norm, null/positive/negative classification, infinity twistor |
| `incidence` | 12 | Minkowski roundtrip, Hermiticity, interval classification, incidence, containment, translation |
| `projective` | 11 | Homogeneous/inhomogeneous coords, equivalence, regions, lines, planes |
| `penrose_transform` | 9 | Line bundles, field types, helicity, contour integral, cohomology |
| `massless` | 11 | Field types, frequency, splitting, evaluation, twistor regions |
| `curved` | 11 | Flat/deformed spaces, incidence, linearization, graviton, Weyl spinor, complex structure |
| `ward` | 12 | Bundles, direct sum, tensor, dual, correspondence, instanton, field strength |
| `null_geodesic` | 10 | Creation, affine parameter, direction, celestial sphere, congruence, optical scalars |
| `conformal` | 9 | Identity, dilation, translation, composition, inverse, generators, special conformal |
| `agent` | 14 | States, trajectories, interpolation, velocity, resampling, frequency, closest approach |

Run them with:

```bash
cargo test
```

---

## License

MIT
