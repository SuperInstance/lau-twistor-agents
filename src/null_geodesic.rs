//! Null geodesics: light rays as points in twistor space.
//!
//! A null geodesic in Minkowski space corresponds to a point in projective twistor space.
//! The set of null geodesics through a spacetime point forms a Riemann sphere (celestial sphere).

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::incidence::SpacetimePoint;
use crate::projective::ProjectiveTwistor;
use crate::spinor::PrimedSpinor;
use crate::twistor::Twistor;

/// A null geodesic (light ray) in Minkowski spacetime.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct NullGeodesic {
    /// A point on the geodesic.
    pub point: SpacetimePoint,
    /// The null direction as a spinor pair (ξ^A, π_{A'}).
    pub direction: PrimedSpinor,
}

impl NullGeodesic {
    /// Create a null geodesic through a point in a given direction.
    pub fn new(point: SpacetimePoint, direction: PrimedSpinor) -> Self {
        Self { point, direction }
    }

    /// Create from Minkowski coordinates and a null direction vector.
    #[allow(clippy::too_many_arguments)]
    pub fn from_minkowski_with_direction(
        t: f64, x: f64, y: f64, z: f64,
        dt: f64, dx: f64, dy: f64, dz: f64,
    ) -> Option<Self> {
        // Check null direction
        if (dt * dt - dx * dx - dy * dy - dz * dz).abs() > 1e-10 {
            return None;
        }
        let point = SpacetimePoint::from_minkowski(t, x, y, z);
        // Convert direction to primed spinor
        let norm = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-15);
        let direction = PrimedSpinor::new(
            Complex64::new((dt + dz) / (2.0 * norm).sqrt(), 0.0),
            Complex64::new(dx / (2.0 * norm).sqrt(), dy / (2.0 * norm).sqrt()),
        );
        Some(Self { point, direction })
    }

    /// The corresponding projective twistor.
    /// A null geodesic ↔ a point in projective twistor space.
    pub fn to_twistor(&self) -> Twistor {
        self.point.twistor_line(&self.direction)
    }

    /// The corresponding projective twistor point.
    pub fn to_projective_twistor(&self) -> ProjectiveTwistor {
        ProjectiveTwistor::new(self.to_twistor())
    }

    /// Evaluate the geodesic at affine parameter λ.
    /// x(λ) = x₀ + λ * n where n is the null direction.
    pub fn at_affine_parameter(&self, lambda: f64) -> SpacetimePoint {
        let n = self.direction_vector();
        self.point.translate(lambda * n[0], lambda * n[1], lambda * n[2], lambda * n[3])
    }

    /// Extract the null direction as a 4-vector (dt, dx, dy, dz).
    pub fn direction_vector(&self) -> [f64; 4] {
        let pi = self.direction;
        // Direction from π: n^{AA'} = ω^A π̄^{A'} where ω comes from incidence
        // Simplified: extract from primed spinor
        let pi0 = pi.components[0];
        let pi1 = pi.components[1];
        let pi0c = pi0.conj();
        let pi1c = pi1.conj();
        // n^{00'} = |π0|²
        // n^{01'} = π0 π̄1
        // n^{10'} = π1 π̄0
        // n^{11'} = |π1|²
        let n00 = pi0 * pi0c;
        let n01 = pi0 * pi1c;
        let n10 = pi1 * pi0c;
        let n11 = pi1 * pi1c;
        let dt = (n00 + n11).re;
        let dx = (n01 + n10).re;
        let dy = (n01 - n10).im;
        let dz = (n00 - n11).re;
        [dt, dx, dy, dz]
    }

    /// Check if two null geodesics intersect.
    pub fn intersects(&self, other: &NullGeodesic) -> bool {
        // Two null geodesics intersect iff the corresponding twistors
        // satisfy a certain incidence relation
        let _z1 = self.to_twistor();
        let _z2 = other.to_twistor();
        // They intersect if the twistor lines in PT share a point
        // Simplified: check if the spacetime points are connected by a null interval
        let diff = self.point.translate(
            -other.point.coords[0][0].re - other.point.coords[1][1].re,
            -other.point.coords[0][1].re - other.point.coords[1][0].re,
            -other.point.coords[0][1].im + other.point.coords[1][0].im,
            -other.point.coords[0][0].re + other.point.coords[1][1].re,
        );
        diff.is_null()
    }

    /// The celestial sphere at this point: all null directions from this point.
    /// Returns a collection of null geodesics parameterized by (θ, φ).
    pub fn celestial_sphere(&self, n_directions: usize) -> Vec<NullGeodesic> {
        let mut geodesics = Vec::new();
        for i in 0..n_directions {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / n_directions as f64;
            for j in 0..n_directions {
                let phi = 2.0 * std::f64::consts::PI * j as f64 / n_directions as f64;
                let pi = PrimedSpinor::new(
                    Complex64::new((theta / 2.0).cos(), 0.0),
                    Complex64::new((theta / 2.0).sin() * phi.cos(), (theta / 2.0).sin() * phi.sin()),
                );
                geodesics.push(NullGeodesic::new(self.point, pi));
            }
        }
        geodesics
    }
}

/// A congruence of null geodesics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NullCongruence {
    /// The geodesics in the congruence.
    pub geodesics: Vec<NullGeodesic>,
}

impl NullCongruence {
    /// Create from a collection of geodesics.
    pub fn new(geodesics: Vec<NullGeodesic>) -> Self {
        Self { geodesics }
    }

    /// A null congruence from a single spacetime point (the celestial sphere).
    pub fn from_point(p: SpacetimePoint, n_directions: usize) -> Self {
        let base = NullGeodesic::new(p, PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)));
        Self::new(base.celestial_sphere(n_directions))
    }

    /// Number of geodesics.
    pub fn len(&self) -> usize {
        self.geodesics.len()
    }

    /// Is the congruence empty?
    pub fn is_empty(&self) -> bool {
        self.geodesics.is_empty()
    }

    /// Compute the expansion, shear, and twist of the congruence (simplified).
    pub fn optical_scalars(&self) -> OpticalScalars {
        if self.geodesics.is_empty() {
            return OpticalScalars::zero();
        }
        // Simplified: average over the directions
        let n = self.geodesics.len() as f64;
        let mut expansion = 0.0;
        let mut shear = 0.0;
        let mut twist = 0.0;
        for g in &self.geodesics {
            let d = g.direction_vector();
            let norm = (d[1] * d[1] + d[2] * d[2] + d[3] * d[3]).sqrt();
            if norm > 1e-10 {
                expansion += 1.0 / norm;
                shear += (d[1] * d[1] + d[2] * d[2] - 2.0 * d[3] * d[3]).abs() / (norm * norm);
                twist += (d[1] * d[2]).abs() / (norm * norm);
            }
        }
        OpticalScalars {
            expansion: expansion / n,
            shear: shear / n,
            twist: twist / n,
        }
    }
}

/// Optical scalars of a null congruence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OpticalScalars {
    /// Expansion: ρ (real, divergence of the congruence).
    pub expansion: f64,
    /// Shear: σ (complex modulus, distortion of the cross-section).
    pub shear: f64,
    /// Twist: ω (rotation of the congruence).
    pub twist: f64,
}

impl OpticalScalars {
    /// Zero optical scalars.
    pub fn zero() -> Self {
        Self {
            expansion: 0.0,
            shear: 0.0,
            twist: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_null_geodesic_creation() {
        let p = SpacetimePoint::origin();
        let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let ng = NullGeodesic::new(p, pi);
        assert_abs_diff_eq!(
            ng.to_twistor().omega.components[0].norm(), 0.0, epsilon = 1e-10
        );
    }

    #[test]
    fn test_null_geodesic_affine_parameter() {
        let p = SpacetimePoint::origin();
        let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let ng = NullGeodesic::new(p, pi);
        let q = ng.at_affine_parameter(1.0);
        let (t, _, _, _z) = q.to_minkowski();
        // Light ray: t = z for π = (1,0)
        assert!(t > 0.0);
    }

    #[test]
    fn test_direction_vector_null() {
        let p = SpacetimePoint::origin();
        let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0));
        let ng = NullGeodesic::new(p, pi);
        let d = ng.direction_vector();
        let interval = d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3];
        assert_abs_diff_eq!(interval, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_projective_twistor() {
        let p = SpacetimePoint::from_minkowski(1.0, 0.0, 0.0, 0.0);
        let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let ng = NullGeodesic::new(p, pi);
        let pt = ng.to_projective_twistor();
        assert!(pt.representative.norm() > 0.0);
    }

    #[test]
    fn test_celestial_sphere() {
        let p = SpacetimePoint::origin();
        let ng = NullGeodesic::new(p, PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)));
        let sphere = ng.celestial_sphere(3);
        assert_eq!(sphere.len(), 9); // 3×3 grid
    }

    #[test]
    fn test_null_congruence() {
        let p = SpacetimePoint::origin();
        let nc = NullCongruence::from_point(p, 4);
        assert_eq!(nc.len(), 16); // 4×4
    }

    #[test]
    fn test_optical_scalars_zero() {
        let os = OpticalScalars::zero();
        assert_abs_diff_eq!(os.expansion, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(os.shear, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_optical_scalars_congruence() {
        let p = SpacetimePoint::origin();
        let nc = NullCongruence::from_point(p, 3);
        let os = nc.optical_scalars();
        // From a single point, expansion is positive (diverging)
        assert!(os.expansion > 0.0);
    }

    #[test]
    fn test_from_minkowski_with_direction_null() {
        let ng = NullGeodesic::from_minkowski_with_direction(0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.8, 0.0);
        assert!(ng.is_some());
    }

    #[test]
    fn test_from_minkowski_with_direction_timelike_fails() {
        let ng = NullGeodesic::from_minkowski_with_direction(0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0);
        assert!(ng.is_none());
    }
}
