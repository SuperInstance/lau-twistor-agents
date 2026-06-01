//! Incidence relation: point in spacetime ↔ line in twistor space.
//!
//! The fundamental incidence relation: ω^A = ix^{AA'} π_{A'}
//! A point x in Minkowski space corresponds to a line (Riemann sphere) in twistor space.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::spinor::{Spinor, PrimedSpinor};
use crate::twistor::Twistor;

/// A point in Minkowski spacetime represented in spinor form.
/// x^{AA'} is a 2x2 Hermitian matrix encoding the spacetime coordinates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SpacetimePoint {
    /// x^{AA'} as a 2x2 Hermitian matrix
    pub coords: [[Complex64; 2]; 2],
}

impl SpacetimePoint {
    /// Create from Minkowski coordinates (t, x, y, z).
    pub fn from_minkowski(t: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            coords: [
                [Complex64::new((t + z) / 2.0, 0.0), Complex64::new(x / 2.0, y / 2.0)],
                [Complex64::new(x / 2.0, -y / 2.0), Complex64::new((t - z) / 2.0, 0.0)],
            ],
        }
    }

    /// The origin of Minkowski space.
    pub fn origin() -> Self {
        Self::from_minkowski(0.0, 0.0, 0.0, 0.0)
    }

    /// Extract Minkowski coordinates (t, x, y, z).
    pub fn to_minkowski(&self) -> (f64, f64, f64, f64) {
        let t = self.coords[0][0].re + self.coords[1][1].re;
        let x = self.coords[0][1].re + self.coords[1][0].re;
        let y = self.coords[0][1].im - self.coords[1][0].im;
        let z = self.coords[0][0].re - self.coords[1][1].re;
        (t, x, y, z)
    }

    /// Check Hermiticity: the matrix should be Hermitian.
    pub fn is_hermitian(&self) -> bool {
        (self.coords[0][1] - self.coords[1][0].conj()).norm() < 1e-10
            && (self.coords[0][0].im).abs() < 1e-10
            && (self.coords[1][1].im).abs() < 1e-10
    }

    /// Lorentzian interval: t² - x² - y² - z².
    pub fn interval(&self) -> f64 {
        let (t, x, y, z) = self.to_minkowski();
        t * t - x * x - y * y - z * z
    }

    /// Is this point null (lightlike)?
    pub fn is_null(&self) -> bool {
        self.interval().abs() < 1e-10
    }

    /// Is this point timelike?
    pub fn is_timelike(&self) -> bool {
        self.interval() > 1e-10
    }

    /// Is this point spacelike?
    pub fn is_spacelike(&self) -> bool {
        self.interval() < -1e-10
    }

    /// Compute ω^A = ix^{AA'} π_{A'} (the incidence relation).
    pub fn incidence(&self, pi: &PrimedSpinor) -> Spinor {
        let omega0 = Complex64::new(0.0, 1.0)
            * (self.coords[0][0] * pi.components[0] + self.coords[0][1] * pi.components[1]);
        let omega1 = Complex64::new(0.0, 1.0)
            * (self.coords[1][0] * pi.components[0] + self.coords[1][1] * pi.components[1]);
        Spinor::new(omega0, omega1)
    }

    /// The twistor line: for a fixed spacetime point, varying π gives all twistors
    /// lying on the corresponding line in twistor space.
    pub fn twistor_line(&self, pi: &PrimedSpinor) -> Twistor {
        Twistor::new(self.incidence(pi), *pi)
    }

    /// Check if a twistor lies on the line corresponding to this point.
    pub fn contains_twistor(&self, z: &Twistor) -> bool {
        let omega_expected = self.incidence(&z.pi);
        (z.omega.components[0] - omega_expected.components[0]).norm() < 1e-10
            && (z.omega.components[1] - omega_expected.components[1]).norm() < 1e-10
    }

    /// Compute the line in twistor space as two points (parameterized by π).
    /// Returns two points on the line: π = (1,0) and π = (0,1).
    pub fn twistor_line_endpoints(&self) -> [Twistor; 2] {
        let pi0 = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let pi1 = PrimedSpinor::new(Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
        [self.twistor_line(&pi0), self.twistor_line(&pi1)]
    }

    /// Spacetime translation by a 4-vector.
    pub fn translate(&self, dt: f64, dx: f64, dy: f64, dz: f64) -> SpacetimePoint {
        let (t, x, y, z) = self.to_minkowski();
        SpacetimePoint::from_minkowski(t + dt, x + dx, y + dy, z + dz)
    }
}

/// The incidence relation linking spacetime points and twistor lines.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IncidenceRelation {
    /// The spacetime point.
    pub point: SpacetimePoint,
}

impl IncidenceRelation {
    /// Create an incidence relation for a given spacetime point.
    pub fn new(point: SpacetimePoint) -> Self {
        Self { point }
    }

    /// Create from Minkowski coordinates.
    pub fn from_minkowski(t: f64, x: f64, y: f64, z: f64) -> Self {
        Self::new(SpacetimePoint::from_minkowski(t, x, y, z))
    }

    /// Generate the twistor for a given π_{A'}.
    pub fn twistor_for_pi(&self, pi: &PrimedSpinor) -> Twistor {
        self.point.twistor_line(pi)
    }

    /// Check incidence: does a twistor lie on this spacetime point's line?
    pub fn check(&self, z: &Twistor) -> bool {
        self.point.contains_twistor(z)
    }

    /// Two spacetime points are null-separated if their twistor lines intersect.
    pub fn are_null_separated(p1: &SpacetimePoint, p2: &SpacetimePoint) -> bool {
        // Two lines in PT intersect iff the points are null-separated
        p1.translate(
            -p2.coords[0][0].re - p2.coords[1][1].re,
            -p2.coords[0][1].re - p2.coords[1][0].re,
            -p2.coords[0][1].im + p2.coords[1][0].im,
            -p2.coords[0][0].re + p2.coords[1][1].re,
        )
        .is_null()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_origin() {
        let o = SpacetimePoint::origin();
        let (t, x, _y, _z) = o.to_minkowski();
        assert_abs_diff_eq!(t, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_minkowski_roundtrip() {
        let p = SpacetimePoint::from_minkowski(1.0, 2.0, 3.0, 4.0);
        let (t, x, y, z) = p.to_minkowski();
        assert_abs_diff_eq!(t, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(z, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hermiticity() {
        let p = SpacetimePoint::from_minkowski(1.0, 2.0, 3.0, 4.0);
        assert!(p.is_hermitian());
    }

    #[test]
    fn test_null_point() {
        // Lightlike: t=5, r=5
        let p = SpacetimePoint::from_minkowski(5.0, 3.0, 4.0, 0.0);
        assert!(p.is_null());
    }

    #[test]
    fn test_timelike_point() {
        let p = SpacetimePoint::from_minkowski(10.0, 0.0, 0.0, 0.0);
        assert!(p.is_timelike());
    }

    #[test]
    fn test_spacelike_point() {
        let p = SpacetimePoint::from_minkowski(0.0, 1.0, 0.0, 0.0);
        assert!(p.is_spacelike());
    }

    #[test]
    fn test_incidence_at_origin() {
        let o = SpacetimePoint::origin();
        let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let omega = o.incidence(&pi);
        // At origin, ω = 0
        assert_abs_diff_eq!(omega.components[0].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_incidence_and_contains() {
        let p = SpacetimePoint::from_minkowski(1.0, 0.0, 0.0, 0.0);
        let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let t = p.twistor_line(&pi);
        assert!(p.contains_twistor(&t));
    }

    #[test]
    fn test_twistor_line_endpoints() {
        let p = SpacetimePoint::from_minkowski(1.0, 2.0, 0.0, 0.0);
        let [t0, t1] = p.twistor_line_endpoints();
        assert!(p.contains_twistor(&t0));
        assert!(p.contains_twistor(&t1));
    }

    #[test]
    fn test_incidence_relation_check() {
        let p = SpacetimePoint::from_minkowski(2.0, 1.0, 0.0, 0.0);
        let ir = IncidenceRelation::new(p);
        let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0));
        let z = ir.twistor_for_pi(&pi);
        assert!(ir.check(&z));
    }

    #[test]
    fn test_translate() {
        let p = SpacetimePoint::origin();
        let q = p.translate(1.0, 2.0, 3.0, 4.0);
        let (t, x, y, z) = q.to_minkowski();
        assert_abs_diff_eq!(t, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(z, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interval() {
        let p = SpacetimePoint::from_minkowski(3.0, 0.0, 0.0, 0.0);
        assert_abs_diff_eq!(p.interval(), 9.0, epsilon = 1e-10);
    }
}
