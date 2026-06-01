//! Conformal group acting on twistor space.
//!
//! The conformal group SU(2,2) acts linearly on twistor space C⁴.
//! It includes Lorentz transformations, translations, dilations, and special conformal
//! transformations — all encoded as 4×4 complex matrices preserving the twistor inner product.

use nalgebra::Matrix4;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::twistor::Twistor;

/// An element of the conformal group SU(2,2), acting on twistor space.
/// Represented as a 4×4 complex matrix preserving the Hermitian form of signature (2,2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalGroup {
    /// The 4×4 matrix representation acting on twistor components Z^α.
    pub matrix: Matrix4<Complex64>,
}

impl ConformalGroup {
    /// Identity transformation.
    pub fn identity() -> Self {
        Self {
            matrix: Matrix4::identity(),
        }
    }

    /// Create from a 4×4 matrix.
    pub fn from_matrix(m: Matrix4<Complex64>) -> Self {
        Self { matrix: m }
    }

    /// Lorentz transformation: acts on the spinor parts separately.
    /// Given by SL(2,C) × SL(2,C) acting as (ω, π) → (Aω, Bπ).
    pub fn lorentz(a: &nalgebra::Matrix2<Complex64>, b: &nalgebra::Matrix2<Complex64>) -> Self {
        let mut m = Matrix4::zeros();
        // Top-left 2x2: A acting on ω
        m[(0, 0)] = a[(0, 0)];
        m[(0, 1)] = a[(0, 1)];
        m[(1, 0)] = a[(1, 0)];
        m[(1, 1)] = a[(1, 1)];
        // Bottom-right 2x2: B acting on π
        m[(2, 2)] = b[(0, 0)];
        m[(2, 3)] = b[(0, 1)];
        m[(3, 2)] = b[(1, 0)];
        m[(3, 3)] = b[(1, 1)];
        Self { matrix: m }
    }

    /// Translation by a spacetime vector encoded in spinor form.
    /// In twistor coordinates: (ω, π) → (ω + t·π, π) where t encodes the translation.
    pub fn translation(x: &crate::incidence::SpacetimePoint) -> Self {
        let mut m = Matrix4::identity();
        // Translation modifies ω based on π: ω → ω + ix^{AA'}π_{A'}
        // The i factor and the x^{AA'} matrix give the 2x2 block
        let i = Complex64::new(0.0, 1.0);
        m[(0, 2)] = i * x.coords[0][0];
        m[(0, 3)] = i * x.coords[0][1];
        m[(1, 2)] = i * x.coords[1][0];
        m[(1, 3)] = i * x.coords[1][1];
        Self { matrix: m }
    }

    /// Dilation (scaling): (ω, π) → (λω, π) scales ω but leaves π unchanged.
    pub fn dilation(lambda: Complex64) -> Self {
        let mut m = Matrix4::identity();
        m[(0, 0)] = lambda;
        m[(1, 1)] = lambda;
        Self { matrix: m }
    }

    /// Special conformal transformation (inversion-translation-inversion).
    /// Encoded as: (ω, π) → (ω, π + b·ω) for some b.
    pub fn special_conformal(x: &crate::incidence::SpacetimePoint) -> Self {
        let mut m = Matrix4::identity();
        let i = Complex64::new(0.0, 1.0);
        m[(2, 0)] = i * x.coords[0][0];
        m[(2, 1)] = i * x.coords[0][1];
        m[(3, 0)] = i * x.coords[1][0];
        m[(3, 1)] = i * x.coords[1][1];
        Self { matrix: m }
    }

    /// Apply this conformal transformation to a twistor.
    pub fn apply(&self, z: &Twistor) -> Twistor {
        let v = z.to_vector4();
        let result = self.matrix * v;
        Twistor::from_components(result[0], result[1], result[2], result[3])
    }

    /// Compose two conformal transformations.
    pub fn compose(&self, other: &ConformalGroup) -> ConformalGroup {
        ConformalGroup {
            matrix: self.matrix * other.matrix,
        }
    }

    /// Inverse transformation.
    pub fn inverse(&self) -> Option<ConformalGroup> {
        self.matrix.try_inverse().map(ConformalGroup::from_matrix)
    }

    /// Check if this transformation preserves the twistor inner product.
    /// For SU(2,2): M† G M = G where G = diag(1,1,-1,-1).
    pub fn preserves_inner_product(&self) -> bool {
        let g = Matrix4::from_diagonal(&nalgebra::Vector4::new(
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ));
        let mdag = self.matrix.adjoint();
        let product = mdag * g * self.matrix;
        let diff = product - g;
        diff.iter().all(|x| x.norm() < 1e-10)
    }

    /// The determinant of the transformation matrix.
    pub fn determinant(&self) -> Complex64 {
        self.matrix.determinant()
    }
}

/// Generators of the conformal algebra su(2,2).
/// The conformal group has 15 generators: 6 Lorentz, 4 translations, 4 special conformal, 1 dilation.
pub struct ConformalGenerator;

impl ConformalGenerator {
    /// All 15 basis generators of the conformal algebra.
    pub fn all_generators() -> Vec<Matrix4<Complex64>> {
        let mut gens = Vec::new();
        // 4 translation generators (top-right 2x2 block)
        for i in 0..2 {
            for j in 2..4 {
                let mut m = Matrix4::zeros();
                m[(i, j)] = Complex64::new(1.0, 0.0);
                gens.push(m);
            }
        }
        // 4 special conformal generators (bottom-left 2x2 block)
        for i in 2..4 {
            for j in 0..2 {
                let mut m = Matrix4::zeros();
                m[(i, j)] = Complex64::new(1.0, 0.0);
                gens.push(m);
            }
        }
        // 1 dilation
        let mut d = Matrix4::zeros();
        d[(0, 0)] = Complex64::new(1.0, 0.0);
        d[(1, 1)] = Complex64::new(1.0, 0.0);
        d[(2, 2)] = Complex64::new(-1.0, 0.0);
        d[(3, 3)] = Complex64::new(-1.0, 0.0);
        gens.push(d);
        // 6 Lorentz generators (block diagonal)
        // Left SL(2,C): 3 generators
        for i in 0..2 {
            for j in 0..2 {
                if i != j {
                    let mut m = Matrix4::zeros();
                    m[(i, j)] = Complex64::new(1.0, 0.0);
                    gens.push(m);
                }
            }
        }
        let mut m = Matrix4::zeros();
        m[(0, 0)] = Complex64::new(1.0, 0.0);
        m[(1, 1)] = Complex64::new(-1.0, 0.0);
        gens.push(m);
        // Right SL(2,C): 3 generators
        for i in 2..4 {
            for j in 2..4 {
                if i != j {
                    let mut m = Matrix4::zeros();
                    m[(i, j)] = Complex64::new(1.0, 0.0);
                    gens.push(m);
                }
            }
        }
        let mut m = Matrix4::zeros();
        m[(2, 2)] = Complex64::new(1.0, 0.0);
        m[(3, 3)] = Complex64::new(-1.0, 0.0);
        gens.push(m);
        gens
    }

    /// Count of generators (15 for the conformal group in 4D).
    pub fn count() -> usize {
        15
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_identity() {
        let id = ConformalGroup::identity();
        let z = Twistor::from_components(
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        );
        let z2 = id.apply(&z);
        assert_abs_diff_eq!(z2.omega.components[0].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(z2.omega.components[1].re, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dilation() {
        let d = ConformalGroup::dilation(Complex64::new(2.0, 0.0));
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        let z2 = d.apply(&z);
        assert_abs_diff_eq!(z2.omega.components[0].re, 2.0, epsilon = 1e-10);
        // π unchanged
        assert_abs_diff_eq!(z2.pi.components[0].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_translation() {
        let x = crate::incidence::SpacetimePoint::from_minkowski(0.0, 0.0, 0.0, 0.0);
        let t = ConformalGroup::translation(&x);
        // Translation by origin is identity
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        let z2 = t.apply(&z);
        assert_abs_diff_eq!(z2.omega.components[0].re, z.omega.components[0].re, epsilon = 1e-10);
    }

    #[test]
    fn test_compose() {
        let d1 = ConformalGroup::dilation(Complex64::new(2.0, 0.0));
        let d2 = ConformalGroup::dilation(Complex64::new(3.0, 0.0));
        let comp = d1.compose(&d2);
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let z2 = comp.apply(&z);
        assert_abs_diff_eq!(z2.omega.components[0].re, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse() {
        let d = ConformalGroup::dilation(Complex64::new(5.0, 0.0));
        let inv = d.inverse().unwrap();
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        );
        let z2 = d.apply(&z);
        let z3 = inv.apply(&z2);
        assert_abs_diff_eq!(z3.omega.components[0].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_generators_count() {
        let gens = ConformalGenerator::all_generators();
        assert_eq!(gens.len(), 15);
    }

    #[test]
    fn test_determinant() {
        let id = ConformalGroup::identity();
        assert_abs_diff_eq!(id.determinant().re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_special_conformal() {
        let x = crate::incidence::SpacetimePoint::from_minkowski(1.0, 0.0, 0.0, 0.0);
        let sc = ConformalGroup::special_conformal(&x);
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let z2 = sc.apply(&z);
        // π should be modified
        assert!(z2.pi.components[0].norm() > 0.0 || z2.pi.components[1].norm() > 0.0);
    }
}
