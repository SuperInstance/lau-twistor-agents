//! 2-component spinor formalism for Lorentzian agent manifolds.
//!
//! In 2-spinor formalism, spacetime vectors are expressed as symmetric outer
//! products of spinors: V^{a} = σ^{a}_{AA'} ξ^{A} η^{A'}

use nalgebra::{Matrix2, Vector2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// A 2-component Weyl spinor (undotted index).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Spinor {
    pub components: Vector2<Complex64>,
}

impl Spinor {
    /// Create a new spinor from two complex components.
    pub fn new(upper: Complex64, lower: Complex64) -> Self {
        Self {
            components: Vector2::new(upper, lower),
        }
    }

    /// Zero spinor.
    pub fn zero() -> Self {
        Self::new(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0))
    }

    /// Unit spinor (1, 0).
    pub fn unit_upper() -> Self {
        Self::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0))
    }

    /// Unit spinor (0, 1).
    pub fn unit_lower() -> Self {
        Self::new(Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0))
    }

    /// Complex conjugate (dotted index spinor).
    pub fn conjugate(&self) -> Self {
        Self {
            components: Vector2::new(self.components[0].conj(), self.components[1].conj()),
        }
    }

    /// Spinor norm squared: |ξ⁰|² + |ξ¹|²
    pub fn norm_squared(&self) -> f64 {
        self.components[0].norm_sqr() + self.components[1].norm_sqr()
    }

    /// Spinor norm.
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Is this a null (zero) spinor?
    pub fn is_null(&self) -> bool {
        self.norm_squared() < 1e-12
    }

    /// Symmetric outer product ξ^{(A} η^{B)} — produces a symmetric 2x2 matrix.
    pub fn symmetric_product(&self, other: &Spinor) -> Matrix2<Complex64> {
        let s = self.components;
        let o = other.components;
        Matrix2::new(
            s[0] * o[0],
            0.5 * (s[0] * o[1] + s[1] * o[0]),
            0.5 * (s[0] * o[1] + s[1] * o[0]),
            s[1] * o[1],
        )
    }

    /// Contract with epsilon tensor: ε_{AB} ξ^{A} η^{B}
    /// This is the SL(2,C) invariant inner product.
    pub fn contract(&self, other: &Spinor) -> Complex64 {
        self.components[0] * other.components[1] - self.components[1] * other.components[0]
    }

    /// Raise/lower index using epsilon: ξ_A = ε_{AB} ξ^B
    pub fn lower_index(&self) -> Spinor {
        Spinor::new(-self.components[1], self.components[0])
    }

    /// Scalar multiplication.
    pub fn scale(&self, factor: Complex64) -> Spinor {
        Spinor {
            components: self.components.map(|c| c * factor),
        }
    }

    /// Spinor addition.
    pub fn add(&self, other: &Spinor) -> Spinor {
        Spinor {
            components: self.components + other.components,
        }
    }

    /// Spinor subtraction.
    pub fn sub(&self, other: &Spinor) -> Spinor {
        Spinor {
            components: self.components - other.components,
        }
    }

    /// Normalize to unit norm.
    pub fn normalize(&self) -> Option<Spinor> {
        let n = self.norm();
        if n < 1e-15 {
            None
        } else {
            Some(Spinor {
                components: self.components.map(|c| c / n),
            })
        }
    }

    /// Projective equivalence: two spinors define the same projective point
    /// if they differ by a non-zero complex scalar.
    pub fn projectively_equivalent(&self, other: &Spinor) -> bool {
        if self.is_null() && other.is_null() {
            return true;
        }
        if self.is_null() || other.is_null() {
            return false;
        }
        // Check if components are proportional
        let r = if self.components[0].norm() > 1e-12 {
            other.components[0] / self.components[0]
        } else if self.components[1].norm() > 1e-12 {
            other.components[1] / self.components[1]
        } else {
            return other.is_null();
        };
        (other.components[0] - r * self.components[0]).norm() < 1e-10
            && (other.components[1] - r * self.components[1]).norm() < 1e-10
    }
}

/// A primed (dotted) spinor — the conjugate representation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PrimedSpinor {
    pub components: Vector2<Complex64>,
}

impl PrimedSpinor {
    pub fn new(upper: Complex64, lower: Complex64) -> Self {
        Self {
            components: Vector2::new(upper, lower),
        }
    }

    /// From conjugating an unprimed spinor.
    pub fn from_conjugate(s: &Spinor) -> Self {
        Self {
            components: Vector2::new(s.components[0].conj(), s.components[1].conj()),
        }
    }

    /// Back to unprimed by conjugation.
    pub fn to_unprimed(&self) -> Spinor {
        Spinor::new(self.components[0].conj(), self.components[1].conj())
    }

    /// SL(2,C) invariant contraction of two primed spinors.
    pub fn contract(&self, other: &PrimedSpinor) -> Complex64 {
        self.components[0] * other.components[1] - self.components[1] * other.components[0]
    }
}

/// Convert a null 4-vector to spinor form: x^{AA'} = ξ^A π^{A'}
/// A null vector can be written as the outer product of a spinor and its conjugate.
pub fn null_vector_to_spinor(v: [Complex64; 4]) -> Option<(Spinor, PrimedSpinor)> {
    // For a real null vector (t, x, y, z) with t²-x²-y²-z² = 0:
    // x^{AA'} as a Hermitian matrix:
    // x^{00'} = (t+z)/2, x^{01'} = (x+iy)/2
    // x^{10'} = (x-iy)/2, x^{11'} = (t-z)/2
    // We need: x^{AA'} = ξ^A π_{A'} (outer product)
    let t = v[0].re;
    let x = v[1].re;
    let y = v[2].re;
    let z = v[3].re;

    let norm = t * t - x * x - y * y - z * z;
    if norm.abs() > 1e-10 {
        return None; // Not null
    }

    let x00 = Complex64::new((t + z) / 2.0, 0.0);
    let x01 = Complex64::new(x / 2.0, y / 2.0);
    let x10 = Complex64::new(x / 2.0, -y / 2.0);
    let x11 = Complex64::new((t - z) / 2.0, 0.0);

    // x^{AA'} = ξ^A π_{A'}
    // x^{00'} = ξ^0 π_0, x^{01'} = ξ^0 π_1
    // x^{10'} = ξ^1 π_0, x^{11'} = ξ^1 π_1
    let (xi, pi) = if x00.norm() > 1e-12 {
        // ξ^0 ≠ 0: set ξ^0 = 1
        (
            Spinor::new(Complex64::new(1.0, 0.0), x10 / x00),
            PrimedSpinor::new(x00, x01),
        )
    } else if x11.norm() > 1e-12 {
        // ξ^1 ≠ 0: set ξ^1 = 1
        (
            Spinor::new(x01 / x11, Complex64::new(1.0, 0.0)),
            PrimedSpinor::new(x10, x11),
        )
    } else {
        (
            Spinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)),
            PrimedSpinor::new(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)),
        )
    };

    Some((xi, pi))
}

/// Reconstruct a real null 4-vector from spinor pair.
pub fn spinor_to_null_vector(xi: &Spinor, pi: &PrimedSpinor) -> [f64; 4] {
    let x00 = xi.components[0] * pi.components[0];
    let x01 = xi.components[0] * pi.components[1];
    let x10 = xi.components[1] * pi.components[0];
    let x11 = xi.components[1] * pi.components[1];

    let t = (x00 + x11).re;
    let x = (x01 + x10).re;
    let y = (x01 - x10).im;
    let z = (x00 - x11).re;

    [t, x, y, z]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex64;

    #[test]
    fn test_spinor_zero() {
        let s = Spinor::zero();
        assert!(s.is_null());
    }

    #[test]
    fn test_spinor_norm() {
        let s = Spinor::new(Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0));
        assert_abs_diff_eq!(s.norm(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spinor_conjugate() {
        let s = Spinor::new(Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0));
        let c = s.conjugate();
        assert_eq!(c.components[0], Complex64::new(1.0, -2.0));
        assert_eq!(c.components[1], Complex64::new(3.0, 4.0));
    }

    #[test]
    fn test_spinor_contract_self() {
        let s = Spinor::new(Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0));
        let c = s.contract(&s);
        assert_abs_diff_eq!(c.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spinor_contract_antisymmetric() {
        let a = Spinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let b = Spinor::new(Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
        assert_abs_diff_eq!(a.contract(&b).norm(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(b.contract(&a).norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lower_index_roundtrip() {
        let s = Spinor::new(Complex64::new(3.0, -1.0), Complex64::new(2.0, 4.0));
        let lowered = s.lower_index();
        // Lowering once: ξ_A = (-ξ^1, ξ^0)
        assert_abs_diff_eq!(lowered.components[0].re, -s.components[1].re, epsilon = 1e-10);
        assert_abs_diff_eq!(lowered.components[0].im, -s.components[1].im, epsilon = 1e-10);
        assert_abs_diff_eq!(lowered.components[1].re, s.components[0].re, epsilon = 1e-10);
        // Lowering twice: should get back -ξ^A
        let double_lowered = lowered.lower_index();
        assert_abs_diff_eq!(double_lowered.components[0].re, -s.components[0].re, epsilon = 1e-10);
        assert_abs_diff_eq!(double_lowered.components[1].re, -s.components[1].re, epsilon = 1e-10);
    }

    #[test]
    fn test_spinor_scale() {
        let s = Spinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0));
        let scaled = s.scale(Complex64::new(2.0, 0.0));
        assert_eq!(scaled.components[0], Complex64::new(2.0, 0.0));
        assert_eq!(scaled.components[1], Complex64::new(0.0, 2.0));
    }

    #[test]
    fn test_spinor_add() {
        let a = Spinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let b = Spinor::new(Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
        let sum = a.add(&b);
        assert_eq!(sum.components[0], Complex64::new(1.0, 0.0));
        assert_eq!(sum.components[1], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_spinor_normalize() {
        let s = Spinor::new(Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0));
        let n = s.normalize().unwrap();
        assert_abs_diff_eq!(n.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spinor_normalize_zero_returns_none() {
        let s = Spinor::zero();
        assert!(s.normalize().is_none());
    }

    #[test]
    fn test_projective_equivalence() {
        let s = Spinor::new(Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0));
        let t = Spinor::new(Complex64::new(3.0, 0.0), Complex64::new(6.0, 0.0));
        assert!(s.projectively_equivalent(&t));
    }

    #[test]
    fn test_symmetric_product() {
        let s = Spinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let prod = s.symmetric_product(&s);
        assert_abs_diff_eq!(prod[(0, 0)].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(prod[(1, 1)].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_primed_spinor_from_conjugate() {
        let s = Spinor::new(Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0));
        let p = PrimedSpinor::from_conjugate(&s);
        let back = p.to_unprimed();
        assert_abs_diff_eq!(back.components[0].re, s.components[0].re, epsilon = 1e-10);
    }

    #[test]
    fn test_null_vector_roundtrip() {
        let v = [
            Complex64::new(5.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]; // t² = 25, x²+y²+z² = 25
        let (xi, pi) = null_vector_to_spinor(v).unwrap();
        let v2 = spinor_to_null_vector(&xi, &pi);
        assert_abs_diff_eq!(v2[0], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_non_null_vector_fails() {
        let v = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]; // t² ≠ x²+y²+z²
        assert!(null_vector_to_spinor(v).is_none());
    }
}
