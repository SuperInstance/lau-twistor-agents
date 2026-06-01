//! Projective twistor space: CP³.
//!
//! Projective twistor space PT = CP³, where twistors Z and λZ are identified.
//! The topology: PT = PN ∪ PT⁺ ∪ PT⁻ where PN is the space of null projective twistors.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::twistor::Twistor;

/// A projective twistor: an equivalence class [Z^α] in CP³.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ProjectiveTwistor {
    /// The representative twistor (not normalized).
    pub representative: Twistor,
}

impl ProjectiveTwistor {
    /// Create from a twistor.
    pub fn new(z: Twistor) -> Self {
        Self { representative: z }
    }

    /// Create from homogeneous coordinates [Z⁰:Z¹:Z²:Z³].
    pub fn from_homogeneous(z0: Complex64, z1: Complex64, z2: Complex64, z3: Complex64) -> Self {
        Self {
            representative: Twistor::from_components(z0, z1, z2, z3),
        }
    }

    /// Create from inhomogeneous coordinates (z0/z3, z1/z3, z2/z3) when z3 ≠ 0.
    pub fn from_inhomogeneous(w0: Complex64, w1: Complex64, w2: Complex64) -> Option<Self> {
        Some(Self {
            representative: Twistor::from_components(w0, w1, w2, Complex64::new(1.0, 0.0)),
        })
    }

    /// Get inhomogeneous coordinates (Z⁰/Z³, Z¹/Z³, Z²/Z³).
    /// Returns None if Z³ = 0 (point at infinity).
    pub fn to_inhomogeneous(&self) -> Option<(Complex64, Complex64, Complex64)> {
        let z3 = self.representative.pi.components[1];
        if z3.norm() < 1e-15 {
            return None;
        }
        Some((
            self.representative.omega.components[0] / z3,
            self.representative.omega.components[1] / z3,
            self.representative.pi.components[0] / z3,
        ))
    }

    /// Check projective equivalence: [Z] = [W] iff Z = λW for some λ ≠ 0.
    pub fn projectively_equivalent(&self, other: &ProjectiveTwistor) -> bool {
        let z = &self.representative;
        let w = &other.representative;
        if z.is_zero() || w.is_zero() {
            return z.is_zero() && w.is_zero();
        }
        // Find ratio from first non-zero component
        let zc = z.to_vector4();
        let wc = w.to_vector4();
        let ratio = find_ratio(&zc, &wc);
        match ratio {
            Some(r) => {
                for i in 0..4 {
                    if (wc[i] - r * zc[i]).norm() > 1e-8 * wc[i].norm().max(zc[i].norm()).max(1.0) {
                        return false;
                    }
                }
                true
            }
            None => false,
        }
    }

    /// Classify the region of projective twistor space.
    pub fn region(&self) -> TwistorRegion {
        if self.representative.is_null() {
            TwistorRegion::Null
        } else if self.representative.is_positive_frequency() {
            TwistorRegion::Positive
        } else {
            TwistorRegion::Negative
        }
    }

    /// Normalize so that the C⁴ vector has unit norm.
    pub fn normalize(&self) -> Option<Self> {
        let n = self.representative.norm();
        if n < 1e-15 {
            None
        } else {
            Some(Self {
                representative: self.representative.scale(Complex64::new(1.0 / n, 0.0)),
            })
        }
    }
}

fn find_ratio(z: &nalgebra::Vector4<Complex64>, w: &nalgebra::Vector4<Complex64>) -> Option<Complex64> {
    for i in 0..4 {
        if z[i].norm() > 1e-12 {
            return Some(w[i] / z[i]);
        }
    }
    None
}

/// Regions of projective twistor space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TwistorRegion {
    /// PT⁺: positive frequency (Z·Z̄ > 0)
    Positive,
    /// PT⁻: negative frequency (Z·Z̄ < 0)
    Negative,
    /// PN: null twistors (Z·Z̄ = 0)
    Null,
}

/// A projective line in CP³ (corresponds to a spacetime point).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectiveLine {
    /// Two points defining the line.
    pub points: [ProjectiveTwistor; 2],
}

impl ProjectiveLine {
    /// Create from two projective twistors.
    pub fn new(p0: ProjectiveTwistor, p1: ProjectiveTwistor) -> Self {
        Self { points: [p0, p1] }
    }

    /// A point on the line parameterized by t: P(t) = t*P0 + (1-t)*P1.
    pub fn point_at(&self, t: Complex64) -> ProjectiveTwistor {
        let p0 = self.points[0].representative.scale(t);
        let p1 = self.points[1].representative.scale(Complex64::new(1.0, 0.0) - t);
        ProjectiveTwistor::new(p0.add(&p1))
    }

    /// Check if a projective twistor lies on this line.
    pub fn contains(&self, p: &ProjectiveTwistor) -> bool {
        // A point lies on the line through P0, P1 iff P is a linear combination of P0 and P1.
        let pv = p.representative.to_vector4();
        let p0v = self.points[0].representative.to_vector4();
        let p1v = self.points[1].representative.to_vector4();

        // Try to find α, β such that P = α*P0 + β*P1
        // Collect equations with at least one non-zero coefficient
        let mut equations: Vec<[Complex64; 3]> = Vec::new();
        let mut zero_rhs_nonzero: bool = false;
        for i in 0..4 {
            let c0 = p0v[i];
            let c1 = p1v[i];
            let rhs = pv[i];
            if c0.norm() > 1e-12 || c1.norm() > 1e-12 {
                equations.push([c0, c1, rhs]);
            } else if rhs.norm() > 1e-12 {
                // Both coefficients zero but RHS nonzero → impossible
                zero_rhs_nonzero = true;
            }
        }
        if zero_rhs_nonzero {
            return false;
        }
        if equations.len() < 2 {
            return true; // Degenerate
        }

        // Solve 2x2 system from first two equations
        let a = equations[0][0];
        let b = equations[0][1];
        let c = equations[1][0];
        let d = equations[1][1];
        let det = a * d - b * c;
        if det.norm() < 1e-12 {
            return false;
        }
        let alpha = (equations[0][2] * d - b * equations[1][2]) / det;
        let beta = (a * equations[1][2] - equations[0][2] * c) / det;

        // Check remaining equations
        for eq in &equations[2..] {
            let lhs = alpha * eq[0] + beta * eq[1];
            if (lhs - eq[2]).norm() > 1e-8 * eq[2].norm().max(1.0) {
                return false;
            }
        }
        true
    }
}

/// A plane in CP³ (corresponds to a spacetime plane / celestial sphere at a point).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectivePlane {
    /// Three points defining the plane.
    pub points: [ProjectiveTwistor; 3],
}

impl ProjectivePlane {
    /// Create from three non-collinear projective twistors.
    pub fn new(p0: ProjectiveTwistor, p1: ProjectiveTwistor, p2: ProjectiveTwistor) -> Self {
        Self {
            points: [p0, p1, p2],
        }
    }

    /// Check if a point lies on this plane.
    pub fn contains(&self, p: &ProjectiveTwistor) -> bool {
        // P lies on the plane iff [P, P0, P1, P2] are linearly dependent
        // i.e., the 4x4 determinant is zero
        let pv = p.representative.to_vector4();
        let p0v = self.points[0].representative.to_vector4();
        let p1v = self.points[1].representative.to_vector4();
        let p2v = self.points[2].representative.to_vector4();

        let det = four_by_four_det(&pv, &p0v, &p1v, &p2v);
        det.norm() < 1e-8
    }
}

fn four_by_four_det(
    a: &nalgebra::Vector4<Complex64>,
    b: &nalgebra::Vector4<Complex64>,
    c: &nalgebra::Vector4<Complex64>,
    d: &nalgebra::Vector4<Complex64>,
) -> Complex64 {
    // Cofactor expansion along first column
    let mut result = Complex64::new(0.0, 0.0);
    let cols = [b, c, d];
    for i in 0..4 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        // 3x3 minor from cols, rows {0,1,2,3}\{i}
        let mut rows = Vec::new();
        for r in 0..4 {
            if r != i {
                rows.push(r);
            }
        }
        let minor = cols[0][rows[0]]
            * (cols[1][rows[1]] * cols[2][rows[2]] - cols[1][rows[2]] * cols[2][rows[1]])
            - cols[0][rows[1]]
                * (cols[1][rows[0]] * cols[2][rows[2]] - cols[1][rows[2]] * cols[2][rows[0]])
            + cols[0][rows[2]]
                * (cols[1][rows[0]] * cols[2][rows[1]] - cols[1][rows[1]] * cols[2][rows[0]]);
        result += Complex64::new(sign, 0.0) * a[i] * minor;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_projective_twistor_from_homogeneous() {
        let pt = ProjectiveTwistor::from_homogeneous(
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        );
        let (w0, w1, w2) = pt.to_inhomogeneous().unwrap();
        assert_abs_diff_eq!(w0.re, 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(w1.re, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(w2.re, 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_projective_equivalence() {
        let p1 = ProjectiveTwistor::from_homogeneous(
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        );
        let p2 = ProjectiveTwistor::from_homogeneous(
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(8.0, 0.0),
        );
        assert!(p1.projectively_equivalent(&p2));
    }

    #[test]
    fn test_not_projectively_equivalent() {
        let p1 = ProjectiveTwistor::from_homogeneous(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        let p2 = ProjectiveTwistor::from_homogeneous(
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        assert!(!p1.projectively_equivalent(&p2));
    }

    #[test]
    fn test_region_positive() {
        let pt = ProjectiveTwistor::from_homogeneous(
            Complex64::new(5.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        assert_eq!(pt.region(), TwistorRegion::Positive);
    }

    #[test]
    fn test_region_negative() {
        // ω=(-1,0), π=(1,0): hermitian_norm = 2*Re(-1*1) = -2 < 0
        let pt = ProjectiveTwistor::from_homogeneous(
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        assert_eq!(pt.region(), TwistorRegion::Negative);
    }

    #[test]
    fn test_normalize() {
        let pt = ProjectiveTwistor::from_homogeneous(
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        let n = pt.normalize().unwrap();
        assert_abs_diff_eq!(n.representative.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_projective_line_contains() {
        let p0 = ProjectiveTwistor::from_homogeneous(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let p1 = ProjectiveTwistor::from_homogeneous(
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let line = ProjectiveLine::new(p0, p1);
        // Midpoint
        let mid = ProjectiveTwistor::from_homogeneous(
            Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        );
        assert!(line.contains(&mid));
    }

    #[test]
    fn test_projective_line_not_contains() {
        let p0 = ProjectiveTwistor::from_homogeneous(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let p1 = ProjectiveTwistor::from_homogeneous(
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let line = ProjectiveLine::new(p0, p1);
        let off = ProjectiveTwistor::from_homogeneous(
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        assert!(!line.contains(&off));
    }

    #[test]
    fn test_inhomogeneous_at_infinity() {
        let pt = ProjectiveTwistor::from_homogeneous(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        );
        assert!(pt.to_inhomogeneous().is_none());
    }

    #[test]
    fn test_from_inhomogeneous() {
        let pt = ProjectiveTwistor::from_inhomogeneous(
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ).unwrap();
        let (w0, _w1, _w2) = pt.to_inhomogeneous().unwrap();
        assert_abs_diff_eq!(w0.re, 1.0, epsilon = 1e-10);
    }
}
