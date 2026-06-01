//! Curved twistor spaces: deformation for non-flat agent backgrounds.
//!
//! In curved spacetime, twistor space is no longer flat CP³ but a complex manifold
//! deformed according to the spacetime curvature. The nonlinear graviton construction
//! encodes self-dual solutions of Einstein's equations as deformed twistor spaces.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::incidence::SpacetimePoint;
use crate::spinor::PrimedSpinor;
use crate::twistor::Twistor;

/// A deformation of flat twistor space representing curved backgrounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvedTwistorSpace {
    /// Deformation parameters: how the ω-part is modified.
    /// In the nonlinear graviton, ω^A = ix^{AA'}π_{A'} + ε f(x,π)
    /// where ε parameterizes the deformation.
    pub deformation: Vec<DeformationTerm>,
    /// The "size" of the deformation (how far from flatness).
    pub epsilon: f64,
}

/// A single term in the deformation expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeformationTerm {
    /// Order of the deformation (1 = linear, 2 = quadratic, etc.)
    pub order: usize,
    /// Coefficient.
    pub coefficient: Complex64,
    /// Which spinor components are affected: (omega_idx, pi_idx).
    pub components: (usize, usize),
}

impl CurvedTwistorSpace {
    /// Flat (undeformed) twistor space.
    pub fn flat() -> Self {
        Self {
            deformation: Vec::new(),
            epsilon: 0.0,
        }
    }

    /// Create a deformed twistor space with a single deformation term.
    pub fn deformed(epsilon: f64, order: usize, coefficient: Complex64, components: (usize, usize)) -> Self {
        Self {
            deformation: vec![DeformationTerm {
                order,
                coefficient,
                components,
            }],
            epsilon,
        }
    }

    /// Apply the deformed incidence relation at a spacetime point.
    /// Returns the twistor corresponding to (x, π) in the curved space.
    pub fn deformed_incidence(&self, x: &SpacetimePoint, pi: &PrimedSpinor) -> Twistor {
        let flat_omega = x.incidence(pi);
        let mut deformed_omega = flat_omega;

        for term in &self.deformation {
            let factor = self.epsilon.powi(term.order as i32) * term.coefficient;
            // Simple deformation: modify omega component based on pi
            let pi_val = if term.components.1 < 2 {
                pi.components[term.components.1]
            } else {
                Complex64::new(0.0, 0.0)
            };
            let x_val = x.coords[term.components.0 / 2][term.components.0 % 2];
            let correction = factor * x_val * pi_val;
            let mut comps = deformed_omega.components;
            comps[term.components.0 % 2] += correction;
            deformed_omega = crate::spinor::Spinor::new(comps[0], comps[1]);
        }

        Twistor::new(deformed_omega, *pi)
    }

    /// Check if this twistor space is essentially flat.
    pub fn is_flat(&self) -> bool {
        self.deformation.is_empty() || self.epsilon.abs() < 1e-15
    }

    /// Compute the "curvature" of the twistor space (simplified scalar measure).
    pub fn curvature_measure(&self) -> f64 {
        let mut measure = 0.0;
        for term in &self.deformation {
            measure += self.epsilon.powi(term.order as i32) * term.coefficient.norm();
        }
        measure
    }

    /// Add a deformation term.
    pub fn add_deformation(&mut self, term: DeformationTerm) {
        self.deformation.push(term);
    }

    /// Linearize: keep only first-order deformations.
    pub fn linearize(&self) -> CurvedTwistorSpace {
        let linear_terms: Vec<_> = self
            .deformation
            .iter()
            .filter(|t| t.order == 1)
            .cloned()
            .collect();
        CurvedTwistorSpace {
            deformation: linear_terms,
            epsilon: self.epsilon,
        }
    }
}

/// The nonlinear graviton construction: self-dual vacuum spacetimes ↔ deformed twistor spaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonlinearGraviton {
    /// The underlying curved twistor space.
    pub twistor_space: CurvedTwistorSpace,
    /// Whether the construction satisfies the self-duality condition.
    pub is_self_dual: bool,
}

impl NonlinearGraviton {
    /// Create from a curved twistor space (assuming self-dual).
    pub fn self_dual(ts: CurvedTwistorSpace) -> Self {
        Self {
            twistor_space: ts,
            is_self_dual: true,
        }
    }

    /// Create an anti-self-dual solution.
    pub fn anti_self_dual(ts: CurvedTwistorSpace) -> Self {
        Self {
            twistor_space: ts,
            is_self_dual: false,
        }
    }

    /// Flat spacetime (no curvature).
    pub fn flat() -> Self {
        Self {
            twistor_space: CurvedTwistorSpace::flat(),
            is_self_dual: true,
        }
    }

    /// Reconstruct the self-dual Weyl spinor at a point.
    /// In full theory: Ψ_{ABCD} comes from the deformation of the twistor fibration.
    pub fn weyl_spinor(&self, x: &SpacetimePoint) -> [[Complex64; 2]; 2] {
        if self.twistor_space.is_flat() {
            return [[Complex64::new(0.0, 0.0); 2]; 2];
        }
        let eps = self.twistor_space.epsilon;
        let measure = self.twistor_space.curvature_measure();
        let (t, px, py, pz) = x.to_minkowski();
        // Simplified: curvature from deformation at this point
        let psi00 = Complex64::new(eps * measure * (t + pz), 0.0);
        let psi01 = Complex64::new(eps * measure * px, eps * measure * py);
        let psi11 = Complex64::new(eps * measure * (t - pz), 0.0);
        [[psi00, psi01], [psi01, psi11]]
    }

    /// Check if the Einstein equations are satisfied (simplified).
    /// In full theory: the deformed twistor space gives an exact self-dual solution.
    pub fn satisfies_einstein(&self) -> bool {
        self.is_self_dual
    }
}

/// Deformation of the complex structure of twistor space.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ComplexStructureDeformation {
    /// The (0,1)-vector field on twistor space defining the deformation.
    /// Simplified as a single complex parameter.
    pub parameter: Complex64,
}

impl ComplexStructureDeformation {
    /// Trivial (no deformation).
    pub fn trivial() -> Self {
        Self {
            parameter: Complex64::new(0.0, 0.0),
        }
    }

    /// Create with a given parameter.
    pub fn new(parameter: Complex64) -> Self {
        Self { parameter }
    }

    /// Check if trivially deformed.
    pub fn is_trivial(&self) -> bool {
        self.parameter.norm() < 1e-15
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_flat_twistor_space() {
        let ts = CurvedTwistorSpace::flat();
        assert!(ts.is_flat());
        assert_abs_diff_eq!(ts.curvature_measure(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_deformed_twistor_space() {
        let ts = CurvedTwistorSpace::deformed(
            0.1,
            1,
            Complex64::new(1.0, 0.0),
            (0, 0),
        );
        assert!(!ts.is_flat());
        assert!(ts.curvature_measure() > 0.0);
    }

    #[test]
    fn test_deformed_incidence() {
        let ts = CurvedTwistorSpace::deformed(
            0.5,
            1,
            Complex64::new(1.0, 0.0),
            (0, 0),
        );
        let x = SpacetimePoint::from_minkowski(1.0, 0.0, 0.0, 0.0);
        let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let z = ts.deformed_incidence(&x, &pi);
        // Should differ from flat
        let z_flat = x.twistor_line(&pi);
        assert!((z.omega.components[0] - z_flat.omega.components[0]).norm() > 1e-10);
    }

    #[test]
    fn test_linearize() {
        let mut ts = CurvedTwistorSpace::flat();
        ts.epsilon = 0.1;
        ts.add_deformation(DeformationTerm { order: 1, coefficient: Complex64::new(1.0, 0.0), components: (0, 0) });
        ts.add_deformation(DeformationTerm { order: 2, coefficient: Complex64::new(1.0, 0.0), components: (0, 0) });
        let lin = ts.linearize();
        assert_eq!(lin.deformation.len(), 1);
        assert_eq!(lin.deformation[0].order, 1);
    }

    #[test]
    fn test_nonlinear_graviton_flat() {
        let ng = NonlinearGraviton::flat();
        assert!(ng.satisfies_einstein());
        let x = SpacetimePoint::origin();
        let weyl = ng.weyl_spinor(&x);
        assert_abs_diff_eq!(weyl[0][0].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nonlinear_graviton_self_dual() {
        let ts = CurvedTwistorSpace::deformed(0.1, 1, Complex64::new(1.0, 0.0), (0, 0));
        let ng = NonlinearGraviton::self_dual(ts);
        assert!(ng.is_self_dual);
        assert!(ng.satisfies_einstein());
    }

    #[test]
    fn test_weyl_spinor_curved() {
        let ts = CurvedTwistorSpace::deformed(0.5, 1, Complex64::new(2.0, 0.0), (0, 0));
        let ng = NonlinearGraviton::self_dual(ts);
        let x = SpacetimePoint::from_minkowski(1.0, 0.0, 0.0, 0.0);
        let weyl = ng.weyl_spinor(&x);
        assert!(weyl[0][0].norm() > 0.0);
    }

    #[test]
    fn test_anti_self_dual() {
        let ts = CurvedTwistorSpace::flat();
        let ng = NonlinearGraviton::anti_self_dual(ts);
        assert!(!ng.is_self_dual);
    }

    #[test]
    fn test_complex_structure_trivial() {
        let cs = ComplexStructureDeformation::trivial();
        assert!(cs.is_trivial());
    }

    #[test]
    fn test_complex_structure_nontrivial() {
        let cs = ComplexStructureDeformation::new(Complex64::new(0.1, 0.2));
        assert!(!cs.is_trivial());
    }

    #[test]
    fn test_curvature_measure_nonzero() {
        let ts = CurvedTwistorSpace::deformed(1.0, 1, Complex64::new(3.0, 4.0), (0, 0));
        assert_abs_diff_eq!(ts.curvature_measure(), 5.0, epsilon = 1e-10);
    }
}
