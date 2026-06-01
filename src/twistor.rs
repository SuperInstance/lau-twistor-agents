//! Twistor space: CP³ as the space of complex lines in C⁴.
//!
//! A twistor Z^α = (ω^A, π_{A'}) is an element of C⁴.
//! Projective twistor space PT = CP³ is the space of complex lines through the origin.

use nalgebra::Vector4;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::spinor::{Spinor, PrimedSpinor};

/// A twistor: an element of twistor space C⁴.
/// Z^α = (ω^A, π_{A'}) where ω^A is a 2-spinor and π_{A'} is a primed 2-spinor.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Twistor {
    /// ω^A: the position-space spinor part
    pub omega: Spinor,
    /// π_{A'}: the momentum-space (primed) spinor part
    pub pi: PrimedSpinor,
}

impl Twistor {
    /// Create a new twistor from spinor and primed spinor.
    pub fn new(omega: Spinor, pi: PrimedSpinor) -> Self {
        Self { omega, pi }
    }

    /// Create from four complex components Z^α = (Z⁰, Z¹, Z², Z³).
    pub fn from_components(z0: Complex64, z1: Complex64, z2: Complex64, z3: Complex64) -> Self {
        Self {
            omega: Spinor::new(z0, z1),
            pi: PrimedSpinor::new(z2, z3),
        }
    }

    /// Create from a C⁴ vector.
    pub fn from_vector4(v: Vector4<Complex64>) -> Self {
        Self {
            omega: Spinor::new(v[0], v[1]),
            pi: PrimedSpinor::new(v[2], v[3]),
        }
    }

    /// Extract the C⁴ vector.
    pub fn to_vector4(&self) -> Vector4<Complex64> {
        Vector4::new(
            self.omega.components[0],
            self.omega.components[1],
            self.pi.components[0],
            self.pi.components[1],
        )
    }

    /// Zero twistor.
    pub fn zero() -> Self {
        Self {
            omega: Spinor::zero(),
            pi: PrimedSpinor::new(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)),
        }
    }

    /// Hermitian inner product: Z · Z̄ = ω^A π̄_A + π_{A'} ω̄^{A'}
    /// This defines the twistor norm.
    pub fn hermitian_norm(&self) -> f64 {
        // Z·Z̄ = 2 * Re(ω^0 * conj(π_0) + ω^1 * conj(π_1))
        let term = self.omega.components[0] * self.pi.components[0].conj()
            + self.omega.components[1] * self.pi.components[1].conj();
        2.0 * term.re
    }

    /// The twistor is null if its Hermitian norm is zero: Z·Z̄ = 0.
    /// Null twistors correspond to real null geodesics in Minkowski space.
    pub fn is_null(&self) -> bool {
        self.hermitian_norm().abs() < 1e-10
    }

    /// The twistor is positive frequency if Z·Z̄ > 0.
    pub fn is_positive_frequency(&self) -> bool {
        self.hermitian_norm() > 1e-10
    }

    /// The twistor is negative frequency if Z·Z̄ < 0.
    pub fn is_negative_frequency(&self) -> bool {
        self.hermitian_norm() < -1e-10
    }

    /// Complex scaling: Z → λZ (projective equivalence).
    pub fn scale(&self, lambda: Complex64) -> Twistor {
        Twistor {
            omega: self.omega.scale(lambda),
            pi: PrimedSpinor::new(
                self.pi.components[0] * lambda,
                self.pi.components[1] * lambda,
            ),
        }
    }

    /// Addition of twistors.
    pub fn add(&self, other: &Twistor) -> Twistor {
        Twistor {
            omega: self.omega.add(&other.omega),
            pi: PrimedSpinor::new(
                self.pi.components[0] + other.pi.components[0],
                self.pi.components[1] + other.pi.components[1],
            ),
        }
    }

    /// Twistor norm squared: Σ |Z^α|².
    pub fn norm_squared(&self) -> f64 {
        self.omega.norm_squared()
            + self.pi.components[0].norm_sqr()
            + self.pi.components[1].norm_sqr()
    }

    /// Twistor norm.
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Is this the zero twistor?
    pub fn is_zero(&self) -> bool {
        self.norm_squared() < 1e-20
    }

    /// Compute the spacetime point associated with a non-null twistor.
    /// For Z^α = (ω^A, π_{A'}), the associated point has coordinates:
    /// x^{AA'} = ω^A π̄^{A'} / (π_{B'} π̄^{B'})
    pub fn associated_spacetime_point(&self) -> Option<[[Complex64; 2]; 2]> {
        let denom = self.pi.components[0].norm_sqr() + self.pi.components[1].norm_sqr();
        if denom < 1e-15 {
            return None;
        }
        let pi_conj = self.pi.to_unprimed();
        let x = [
            [
                self.omega.components[0] * pi_conj.components[0] / denom,
                self.omega.components[0] * pi_conj.components[1] / denom,
            ],
            [
                self.omega.components[1] * pi_conj.components[0] / denom,
                self.omega.components[1] * pi_conj.components[1] / denom,
            ],
        ];
        Some(x)
    }

    /// Get the π spinor as unprimed (dual).
    pub fn pi_as_spinor(&self) -> Spinor {
        Spinor::new(self.pi.components[0], self.pi.components[1])
    }
}

/// The infinity twistor I^{αβ} — dual to the origin in twistor space.
/// It encodes the conformal structure and is used to split twistor space
/// into the two spinor parts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InfinityTwistor {
    /// I^{αβ} as a 4x4 antisymmetric complex matrix (simplified to essential data).
    pub components: [[Complex64; 4]; 4],
}

impl InfinityTwistor {
    /// Standard infinity twistor for Minkowski space.
    /// I^{αβ} has components that split PT into the ω and π parts.
    pub fn minkowski() -> Self {
        Self {
            components: [
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
            ],
        }
    }

    /// Contract with two twistors: I_{αβ} Z^α W^β.
    pub fn contract(&self, z: &Twistor, w: &Twistor) -> Complex64 {
        let zv = z.to_vector4();
        let wv = w.to_vector4();
        let mut result = Complex64::new(0.0, 0.0);
        for i in 0..4 {
            for j in 0..4 {
                result += self.components[i][j] * zv[i] * wv[j];
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_twistor_zero() {
        let t = Twistor::zero();
        assert!(t.is_zero());
    }

    #[test]
    fn test_twistor_from_components() {
        let t = Twistor::from_components(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        assert_abs_diff_eq!(t.omega.components[0].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_twistor_scale() {
        let t = Twistor::from_components(
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        );
        let s = t.scale(Complex64::new(2.0, 0.0));
        assert_abs_diff_eq!(s.omega.components[0].re, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(s.pi.components[1].re, 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_twistor_add() {
        let a = Twistor::from_components(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let b = Twistor::from_components(
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let sum = a.add(&b);
        assert_abs_diff_eq!(sum.omega.components[0].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sum.omega.components[1].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_null_twistor() {
        // ω = (1,0), π = (1,0) → Hermitian norm = Re(1*1) = 1 ≠ 0
        let t = Twistor::from_components(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        assert!(!t.is_null());
    }

    #[test]
    fn test_twistor_norm() {
        let t = Twistor::from_components(
            Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        );
        assert_abs_diff_eq!(t.norm(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vector4_roundtrip() {
        let v = Vector4::new(
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        );
        let t = Twistor::from_vector4(v);
        let v2 = t.to_vector4();
        for i in 0..4 {
            assert_abs_diff_eq!(v[i].re, v2[i].re, epsilon = 1e-10);
            assert_abs_diff_eq!(v[i].im, v2[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_infinity_twistor_minkowski() {
        let i = InfinityTwistor::minkowski();
        // I^{02} = 1, I^{20} = -1
        assert_abs_diff_eq!(i.components[0][2].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(i.components[2][0].re, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_infinity_twistor_contract() {
        let it = InfinityTwistor::minkowski();
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let w = Twistor::from_components(
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let c = it.contract(&z, &w);
        assert_abs_diff_eq!(c.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_positive_frequency() {
        // Need Z·Z̄ > 0: pick omega and pi such that the contraction is positive
        let t = Twistor::from_components(
            Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        assert!(t.is_positive_frequency());
    }

    #[test]
    fn test_associated_spacetime_point() {
        let t = Twistor::from_components(
            Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let pt = t.associated_spacetime_point().unwrap();
        // x^{00'} = ω^0 π̄^{0'} / |π|² = 2*1 / 1 = 2
        assert_abs_diff_eq!(pt[0][0].re, 2.0, epsilon = 1e-10);
    }
}
