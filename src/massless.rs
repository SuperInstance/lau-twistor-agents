//! Massless fields: positive/negative frequency splitting via twistor geometry.
//!
//! The twistor space splits into PT⁺ and PT⁻, giving a natural decomposition
//! of massless fields into positive and negative frequency parts.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::incidence::SpacetimePoint;
use crate::penrose_transform::{PenroseTransform, LineBundle, Helicity};
use crate::twistor::Twistor;


/// Helicity classification for massless fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HelicitySign {
    Positive,
    Negative,
    Zero,
}

/// A massless field on spacetime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasslessField {
    /// The helicity/weight of the field.
    pub helicity: Helicity,
    /// The sign (positive or negative frequency).
    pub sign: HelicitySign,
    /// The line bundle weight: O(-2s-2) for helicity s.
    pub bundle: LineBundle,
}

impl MasslessField {
    /// Create a massless field with given helicity.
    pub fn new(helicity: Helicity, sign: HelicitySign) -> Self {
        let bundle_weight = -2 * helicity - 2;
        Self {
            helicity,
            sign,
            bundle: LineBundle::new(bundle_weight),
        }
    }

    /// Scalar field (helicity 0).
    pub fn scalar(sign: HelicitySign) -> Self {
        Self::new(0, sign)
    }

    /// Left-handed Weyl spinor (helicity +½ → s=1).
    pub fn left_weyl(sign: HelicitySign) -> Self {
        Self::new(1, sign)
    }

    /// Right-handed Weyl spinor (helicity -½ → s=-1).
    pub fn right_weyl(sign: HelicitySign) -> Self {
        Self::new(-1, sign)
    }

    /// Maxwell field (helicity ±1 → s=±2).
    pub fn maxwell(sign: HelicitySign) -> Self {
        Self::new(2, sign)
    }

    /// Linearized gravitational field (helicity ±2 → s=±4).
    pub fn gravity(sign: HelicitySign) -> Self {
        Self::new(4, sign)
    }

    /// The Penrose transform for this field.
    pub fn penrose_transform(&self) -> PenroseTransform {
        PenroseTransform::new(self.bundle.weight)
    }

    /// Check if this field uses PT⁺ (positive frequency) or PT⁻ (negative frequency).
    pub fn twistor_region(&self) -> TwistorRegion {
        match self.sign {
            HelicitySign::Positive => TwistorRegion::Positive,
            HelicitySign::Negative => TwistorRegion::Negative,
            HelicitySign::Zero => TwistorRegion::Null,
        }
    }

    /// The spin of the field (helicity / 2).
    pub fn spin(&self) -> f64 {
        self.helicity as f64 / 2.0
    }

    /// Number of independent components.
    pub fn num_components(&self) -> usize {
        (2 * self.helicity.unsigned_abs() as usize + 1).max(1)
    }

    /// Is this a field of positive frequency?
    pub fn is_positive_frequency(&self) -> bool {
        matches!(self.sign, HelicitySign::Positive)
    }
}

/// Region of twistor space for the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TwistorRegion {
    Positive,
    Negative,
    Null,
}

/// Frequency splitting of a massless field into positive and negative parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencySplitting {
    /// The positive frequency part.
    pub positive: MasslessField,
    /// The negative frequency part.
    pub negative: MasslessField,
}

impl FrequencySplitting {
    /// Create a frequency splitting for a given helicity.
    pub fn new(helicity: Helicity) -> Self {
        Self {
            positive: MasslessField::new(helicity, HelicitySign::Positive),
            negative: MasslessField::new(helicity, HelicitySign::Negative),
        }
    }

    /// Evaluate both parts at a spacetime point.
    pub fn evaluate<F>(
        &self,
        x: &SpacetimePoint,
        f: &F,
        n_samples: usize,
    ) -> (crate::spinor::Spinor, crate::spinor::Spinor)
    where
        F: Fn(&Twistor) -> Complex64,
    {
        let pos = self.positive.penrose_transform().transform_to_spacetime(x, f, n_samples);
        let neg = self.negative.penrose_transform().transform_to_spacetime(x, f, n_samples);
        (pos, neg)
    }
}

/// Evaluate a massless field at a spacetime point using the Penrose transform.
pub fn evaluate_field<F>(
    field: &MasslessField,
    x: &SpacetimePoint,
    twistor_function: F,
    n_samples: usize,
) -> crate::spinor::Spinor
where
    F: Fn(&Twistor) -> Complex64,
{
    field
        .penrose_transform()
        .transform_to_spacetime(x, twistor_function, n_samples)
}

/// Check if a twistor lies in the appropriate region for a massless field.
pub fn twistor_in_region(z: &Twistor, sign: HelicitySign) -> bool {
    match sign {
        HelicitySign::Positive => z.is_positive_frequency(),
        HelicitySign::Negative => z.is_negative_frequency(),
        HelicitySign::Zero => z.is_null(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_scalar_field() {
        let f = MasslessField::scalar(HelicitySign::Positive);
        assert_eq!(f.helicity, 0);
        assert_eq!(f.spin(), 0.0);
    }

    #[test]
    fn test_maxwell_field() {
        let f = MasslessField::maxwell(HelicitySign::Negative);
        assert_eq!(f.helicity, 2);
        assert_eq!(f.spin(), 1.0);
    }

    #[test]
    fn test_gravity_field() {
        let f = MasslessField::gravity(HelicitySign::Positive);
        assert_eq!(f.helicity, 4);
        assert_eq!(f.spin(), 2.0);
    }

    #[test]
    fn test_positive_frequency_region() {
        let f = MasslessField::scalar(HelicitySign::Positive);
        assert_eq!(f.twistor_region(), TwistorRegion::Positive);
    }

    #[test]
    fn test_negative_frequency_region() {
        let f = MasslessField::scalar(HelicitySign::Negative);
        assert_eq!(f.twistor_region(), TwistorRegion::Negative);
    }

    #[test]
    fn test_frequency_splitting() {
        let split = FrequencySplitting::new(2);
        assert_eq!(split.positive.helicity, 2);
        assert_eq!(split.negative.helicity, 2);
        assert!(split.positive.is_positive_frequency());
        assert!(!split.negative.is_positive_frequency());
    }

    #[test]
    fn test_penrose_transform_consistency() {
        let f = MasslessField::maxwell(HelicitySign::Positive);
        let pt = f.penrose_transform();
        assert_eq!(pt.bundle.weight, -6);
    }

    #[test]
    fn test_num_components() {
        assert_eq!(MasslessField::scalar(HelicitySign::Positive).num_components(), 1);
        assert_eq!(MasslessField::maxwell(HelicitySign::Positive).num_components(), 5);
    }

    #[test]
    fn test_left_right_weyl() {
        let left = MasslessField::left_weyl(HelicitySign::Positive);
        let right = MasslessField::right_weyl(HelicitySign::Positive);
        assert_eq!(left.helicity, 1);
        assert_eq!(right.helicity, -1);
    }

    #[test]
    fn test_twistor_in_region_positive() {
        let z = Twistor::from_components(
            Complex64::new(5.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        assert!(twistor_in_region(&z, HelicitySign::Positive));
    }

    #[test]
    fn test_evaluate_scalar() {
        let f = MasslessField::scalar(HelicitySign::Positive);
        let x = SpacetimePoint::origin();
        let result = evaluate_field(&f, &x, |_z| Complex64::new(1.0, 0.0), 8);
        assert!(result.norm_squared() > 0.0);
    }
}
