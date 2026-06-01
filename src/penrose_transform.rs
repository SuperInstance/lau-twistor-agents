//! Penrose transform: solutions of field equations ↔ cohomology of line bundles on twistor space.
//!
//! The Penrose transform maps sheaf cohomology groups H¹(PT⁺, O(k)) to solutions
//! of zero-rest-mass field equations on spacetime.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::incidence::SpacetimePoint;
use crate::spinor::{Spinor, PrimedSpinor};
use crate::twistor::Twistor;

/// Helicity/weight of a twistor field.
pub type Helicity = i32;

/// A line bundle O(k) on projective twistor space.
/// O(k) is the k-th tensor power of the hyperplane bundle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct LineBundle {
    /// The weight/degree k.
    pub weight: i32,
}

impl LineBundle {
    /// O(k) with given weight.
    pub fn new(k: i32) -> Self {
        Self { weight: k }
    }

    /// The trivial bundle O(0).
    pub fn trivial() -> Self {
        Self { weight: 0 }
    }

    /// Tensor product: O(k) ⊗ O(l) = O(k+l).
    pub fn tensor(&self, other: &LineBundle) -> LineBundle {
        LineBundle {
            weight: self.weight + other.weight,
        }
    }

    /// Dual bundle: O(k)* = O(-k).
    pub fn dual(&self) -> LineBundle {
        LineBundle {
            weight: -self.weight,
        }
    }

    /// The canonical bundle: K = O(-4).
    pub fn canonical() -> LineBundle {
        LineBundle { weight: -4 }
    }
}

/// The Penrose transform: maps twistor cohomology to spacetime fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenroseTransform {
    /// The line bundle O(k).
    pub bundle: LineBundle,
}

impl PenroseTransform {
    /// Create a Penrose transform for O(k).
    pub fn new(k: i32) -> Self {
        Self {
            bundle: LineBundle::new(k),
        }
    }

    /// Determine the spacetime field type from the cohomological degree and line bundle weight.
    /// H¹(PT⁺, O(-2n-2)) → spin-n zero-rest-mass field on spacetime.
    pub fn field_type(&self) -> FieldType {
        match self.bundle.weight {
            -2 => FieldType::ScalarField,           // spin-0
            -4 => FieldType::LeftWeylSpinor,        // spin-½ (left-handed)
            -6 => FieldType::MaxwellField,           // spin-1
            -8 => FieldType::LinearizedGravity,      // spin-2 (linearized)
            w if w < -8 && (w + 2) % 2 == 0 => FieldType::HigherSpin(-(w as f64 / 2.0 - 1.0) as u32),
            0 => FieldType::TwistorFunction,         // H¹(PT⁺, O) → twistor function
            _ => FieldType::GenericField,
        }
    }

    /// The helicity of the resulting spacetime field.
    pub fn helicity(&self) -> Helicity {
        -self.bundle.weight / 2 - 1
    }

    /// Evaluate a twistor function (representative of a cohomology class)
    /// at a twistor point, with the appropriate transition function for O(k).
    pub fn evaluate_on_patch(
        &self,
        z: &Twistor,
        f: &dyn Fn(&Twistor) -> Complex64,
    ) -> Complex64 {
        let val = f(z);
        // For O(k), the transition function on the overlap of patches
        // involves multiplication by (Z³/Z²)^k or similar.
        if self.bundle.weight == 0 {
            val
        } else {
            let pi0 = z.pi.components[0];
            let pi1 = z.pi.components[1];
            if pi1.norm() > 1e-12 {
                let ratio = pi0 / pi1;
                val * ratio.powi(self.bundle.weight)
            } else {
                val
            }
        }
    }

    /// Perform the contour integral that defines the Penrose transform.
    /// φ(x) = ∮ f(Z) π_{A'} dπ^{A'} where Z lies on the line through x.
    pub fn transform_to_spacetime<F>(
        &self,
        x: &SpacetimePoint,
        f: F,
        n_samples: usize,
    ) -> Spinor
    where
        F: Fn(&Twistor) -> Complex64,
    {
        // Integrate over the Riemann sphere of the line through x
        // Parameterize π_{A'} = (cos(θ/2), sin(θ/2) e^{iφ})
        let d_theta = std::f64::consts::PI / n_samples as f64;
        let d_phi = 2.0 * std::f64::consts::PI / n_samples as f64;

        let mut result0 = Complex64::new(0.0, 0.0);
        let mut result1 = Complex64::new(0.0, 0.0);

        for i in 0..n_samples {
            let theta = (i as f64 + 0.5) * d_theta;
            for j in 0..n_samples {
                let phi = (j as f64 + 0.5) * d_phi;
                let pi = PrimedSpinor::new(
                    Complex64::new(theta.cos(), 0.0),
                    Complex64::new(theta.sin() * phi.cos(), theta.sin() * phi.sin()),
                );
                let z = x.twistor_line(&pi);
                let val = f(&z);

                // π_{A'} dπ^{A'} with weight for O(k)
                let weight_factor = if self.bundle.weight != 0 {
                    let pi1 = z.pi.components[1];
                    if pi1.norm() > 1e-12 {
                        (z.pi.components[0] / pi1).powi(self.bundle.weight)
                    } else {
                        Complex64::new(1.0, 0.0)
                    }
                } else {
                    Complex64::new(1.0, 0.0)
                };

                result0 += val * weight_factor * pi.components[0] * d_theta * d_phi * theta.sin();
                result1 += val * weight_factor * pi.components[1] * d_theta * d_phi * theta.sin();
            }
        }

        Spinor::new(result0, result1)
    }
}

/// Types of spacetime fields arising from the Penrose transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// Scalar (spin-0): H¹(PT⁺, O(-2))
    ScalarField,
    /// Left Weyl spinor (spin-½): H¹(PT⁺, O(-4))
    LeftWeylSpinor,
    /// Maxwell field (spin-1): H¹(PT⁺, O(-6))
    MaxwellField,
    /// Linearized gravity (spin-2): H¹(PT⁺, O(-8))
    LinearizedGravity,
    /// Higher spin field
    HigherSpin(u32),
    /// Twistor function from H¹(PT⁺, O)
    TwistorFunction,
    /// Generic field
    GenericField,
}

/// A (0,1)-form on twistor space, representing a cohomology class.
pub struct TwistorCohomologyClass {
    /// The line bundle O(k).
    pub bundle: LineBundle,
    /// Coefficient functions on patches (simplified: single function for one patch).
    pub coefficient: Box<dyn Fn(&Twistor) -> Complex64 + Send + Sync>,
}

impl TwistorCohomologyClass {
    /// Create a cohomology class with a single representative function.
    pub fn new(k: i32, f: impl Fn(&Twistor) -> Complex64 + Send + Sync + 'static) -> Self {
        Self {
            bundle: LineBundle::new(k),
            coefficient: Box::new(f),
        }
    }

    /// Evaluate at a twistor point.
    pub fn evaluate(&self, z: &Twistor) -> Complex64 {
        (self.coefficient)(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_line_bundle_tensor() {
        let o2 = LineBundle::new(2);
        let o3 = LineBundle::new(3);
        let o5 = o2.tensor(&o3);
        assert_eq!(o5.weight, 5);
    }

    #[test]
    fn test_line_bundle_dual() {
        let o3 = LineBundle::new(3);
        assert_eq!(o3.dual().weight, -3);
    }

    #[test]
    fn test_canonical_bundle() {
        assert_eq!(LineBundle::canonical().weight, -4);
    }

    #[test]
    fn test_penrose_transform_scalar() {
        let pt = PenroseTransform::new(-2);
        assert_eq!(pt.field_type(), FieldType::ScalarField);
    }

    #[test]
    fn test_penrose_transform_maxwell() {
        let pt = PenroseTransform::new(-6);
        assert_eq!(pt.field_type(), FieldType::MaxwellField);
    }

    #[test]
    fn test_penrose_transform_gravity() {
        let pt = PenroseTransform::new(-8);
        assert_eq!(pt.field_type(), FieldType::LinearizedGravity);
    }

    #[test]
    fn test_penrose_transform_helicity() {
        assert_eq!(PenroseTransform::new(-2).helicity(), 0);   // scalar
        assert_eq!(PenroseTransform::new(-4).helicity(), 1);   // spin-½
        assert_eq!(PenroseTransform::new(-6).helicity(), 2);   // spin-1
    }

    #[test]
    fn test_penrose_transform_constant_field() {
        let pt = PenroseTransform::new(-2);
        let x = SpacetimePoint::origin();
        let result = pt.transform_to_spacetime(&x, |_z| Complex64::new(1.0, 0.0), 10);
        // Should be non-zero from the integral
        assert!(result.norm_squared() > 0.0);
    }

    #[test]
    fn test_line_bundle_trivial() {
        assert_eq!(LineBundle::trivial().weight, 0);
    }

    #[test]
    fn test_twistor_cohomology_evaluate() {
        let class = TwistorCohomologyClass::new(0, |_z| Complex64::new(42.0, 0.0));
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        );
        assert_abs_diff_eq!(class.evaluate(&z).re, 42.0, epsilon = 1e-10);
    }
}
