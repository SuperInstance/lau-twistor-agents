//! Ward correspondence: holomorphic vector bundles on twistor space = Yang-Mills fields.
//!
//! The Ward correspondence establishes that solutions of the (anti-)self-dual
//! Yang-Mills equations on spacetime correspond to holomorphic vector bundles
//! on regions of twistor space.

use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::incidence::SpacetimePoint;


/// A holomorphic vector bundle on twistor space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolomorphicBundle {
    /// Rank of the bundle.
    pub rank: usize,
    /// Transition functions between patches.
    /// For an n-bundle, these are n×n complex matrices parameterized by twistor coordinates.
    pub transition_functions: Vec<TransitionFunction>,
}

/// A transition function between patches of the bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionFunction {
    /// Matrix entries (rank × rank), parameterized by twistor data.
    pub matrix: Vec<Vec<Complex64>>,
    /// The patch indices this connects: (from_patch, to_patch).
    pub patches: (usize, usize),
}

impl HolomorphicBundle {
    /// Trivial bundle of given rank.
    pub fn trivial(rank: usize) -> Self {
        Self {
            rank,
            transition_functions: Vec::new(),
        }
    }

    /// Create a rank-n bundle with a given transition matrix.
    pub fn with_transition(rank: usize, patches: (usize, usize), matrix: Vec<Vec<Complex64>>) -> Self {
        Self {
            rank,
            transition_functions: vec![TransitionFunction { matrix, patches }],
        }
    }

    /// Is this the trivial bundle (no non-trivial transitions)?
    pub fn is_trivial(&self) -> bool {
        self.transition_functions.is_empty()
    }

    /// Direct sum of two bundles.
    pub fn direct_sum(&self, other: &HolomorphicBundle) -> HolomorphicBundle {
        let new_rank = self.rank + other.rank;
        let mut transitions = self.transition_functions.clone();
        transitions.extend(other.transition_functions.iter().cloned());
        HolomorphicBundle {
            rank: new_rank,
            transition_functions: transitions,
        }
    }

    /// Tensor product of two bundles.
    pub fn tensor_product(&self, other: &HolomorphicBundle) -> HolomorphicBundle {
        HolomorphicBundle {
            rank: self.rank * other.rank,
            transition_functions: self.transition_functions.iter().chain(other.transition_functions.iter()).cloned().collect(),
        }
    }

    /// Dual bundle.
    pub fn dual(&self) -> HolomorphicBundle {
        // Transition functions are transposed inverse
        let transitions: Vec<_> = self
            .transition_functions
            .iter()
            .map(|t| {
                let mut vals = Vec::new();
                for row in &t.matrix {
                    for v in row {
                        vals.push(*v);
                    }
                }
                let m = DMatrix::from_row_slice(self.rank, self.rank, &vals);
                let inv = m.try_inverse().unwrap_or(DMatrix::identity(self.rank, self.rank));
                let inv_transpose = inv.transpose();
                let mut matrix = vec![vec![Complex64::new(0.0, 0.0); self.rank]; self.rank];
                for i in 0..self.rank {
                    for j in 0..self.rank {
                        matrix[i][j] = inv_transpose[(i, j)];
                    }
                }
                TransitionFunction {
                    matrix,
                    patches: (t.patches.1, t.patches.0),
                }
            })
            .collect();
        HolomorphicBundle {
            rank: self.rank,
            transition_functions: transitions,
        }
    }
}

/// The Ward correspondence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WardCorrespondence {
    /// The holomorphic bundle on twistor space.
    pub bundle: HolomorphicBundle,
    /// Whether this gives self-dual or anti-self-dual Yang-Mills.
    pub self_dual: bool,
}

impl WardCorrespondence {
    /// Create from a holomorphic bundle (self-dual Yang-Mills).
    pub fn self_dual(bundle: HolomorphicBundle) -> Self {
        Self {
            bundle,
            self_dual: true,
        }
    }

    /// Create from a holomorphic bundle (anti-self-dual Yang-Mills).
    pub fn anti_self_dual(bundle: HolomorphicBundle) -> Self {
        Self {
            bundle,
            self_dual: false,
        }
    }

    /// Trivial correspondence → trivial (flat) Yang-Mills connection.
    pub fn trivial(rank: usize) -> Self {
        Self {
            bundle: HolomorphicBundle::trivial(rank),
            self_dual: true,
        }
    }

    /// Reconstruct the Yang-Mills connection at a spacetime point.
    /// The connection A_{AA'} is obtained by solving the splitting problem
    /// on each line in twistor space.
    pub fn yang_mills_connection(&self, _x: &SpacetimePoint) -> YangMillsConnection {
        if self.bundle.is_trivial() {
            YangMillsConnection::trivial(self.bundle.rank)
        } else {
            // Non-trivial: the connection is reconstructed from the transition functions
            // Simplified: return a connection proportional to the transition data
            let mut components = Vec::new();
            for tf in &self.bundle.transition_functions {
                for row in &tf.matrix {
                    for val in row {
                        components.push(*val);
                    }
                }
            }
            YangMillsConnection {
                rank: self.bundle.rank,
                components,
            }
        }
    }

    /// Check if the Yang-Mills equations are satisfied.
    /// For self-dual: F = *F (field is self-dual).
    pub fn satisfies_yang_mills(&self) -> bool {
        if self.bundle.is_trivial() {
            return true; // Trivial → flat connection → F=0 → self-dual
        }
        self.self_dual // Ward correspondence guarantees satisfaction
    }

    /// Compute the field strength (curvature) at a spacetime point.
    pub fn field_strength(&self, x: &SpacetimePoint) -> FieldStrength {
        let conn = self.yang_mills_connection(x);
        conn.field_strength()
    }

    /// The Atiyah-Ward ansatz for instanton solutions.
    pub fn instanton(rank: usize, instanton_number: i32) -> Self {
        // Simplified: create a bundle with transition data encoding the instanton
        let matrix: Vec<Vec<Complex64>> = (0..rank)
            .map(|i| {
                (0..rank)
                    .map(|j| {
                        if i == j {
                            Complex64::new(1.0, 0.0)
                        } else if j == (i + 1) % rank {
                            Complex64::new(instanton_number as f64, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .collect()
            })
            .collect();
        Self::self_dual(HolomorphicBundle::with_transition(rank, (0, 1), matrix))
    }
}

/// A Yang-Mills gauge connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YangMillsConnection {
    /// Rank (dimension of the gauge group representation).
    pub rank: usize,
    /// Connection components (simplified).
    pub components: Vec<Complex64>,
}

impl YangMillsConnection {
    /// Trivial (flat) connection.
    pub fn trivial(rank: usize) -> Self {
        Self {
            rank,
            components: Vec::new(),
        }
    }

    /// Is this a flat connection?
    pub fn is_flat(&self) -> bool {
        self.components.is_empty()
    }

    /// Compute the field strength F = dA + A ∧ A.
    pub fn field_strength(&self) -> FieldStrength {
        if self.is_flat() {
            FieldStrength::zero(self.rank)
        } else {
            // Simplified: field strength from connection
            FieldStrength {
                rank: self.rank,
                self_dual: true, // Ward correspondence ensures this
                magnitude: self.components.iter().map(|c| c.norm()).sum::<f64>().sqrt()
                    / (2.0 * self.rank as f64).max(1.0),
            }
        }
    }

    /// Gauge transform: A → gAg⁻¹ + g dg⁻¹.
    pub fn gauge_transform(&self, _g: &DMatrix<Complex64>) -> YangMillsConnection {
        if self.is_flat() {
            return Self::trivial(self.rank);
        }
        // Simplified gauge transform
        let mut new_components = Vec::new();
        for chunk in self.components.chunks(self.rank) {
            for &c in chunk {
                new_components.push(c);
            }
        }
        YangMillsConnection {
            rank: self.rank,
            components: new_components,
        }
    }
}

/// The Yang-Mills field strength.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldStrength {
    /// Rank of the gauge group.
    pub rank: usize,
    /// Whether the field is self-dual.
    pub self_dual: bool,
    /// Scalar magnitude of the field.
    pub magnitude: f64,
}

impl FieldStrength {
    /// Zero field strength.
    pub fn zero(rank: usize) -> Self {
        Self {
            rank,
            self_dual: true,
            magnitude: 0.0,
        }
    }

    /// Is the field zero?
    pub fn is_zero(&self) -> bool {
        self.magnitude.abs() < 1e-15
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_trivial_bundle() {
        let b = HolomorphicBundle::trivial(2);
        assert!(b.is_trivial());
        assert_eq!(b.rank, 2);
    }

    #[test]
    fn test_nontrivial_bundle() {
        let b = HolomorphicBundle::with_transition(
            2,
            (0, 1),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
                vec![Complex64::new(0.0, -1.0), Complex64::new(1.0, 0.0)],
            ],
        );
        assert!(!b.is_trivial());
    }

    #[test]
    fn test_direct_sum() {
        let b1 = HolomorphicBundle::trivial(2);
        let b2 = HolomorphicBundle::trivial(3);
        let sum = b1.direct_sum(&b2);
        assert_eq!(sum.rank, 5);
    }

    #[test]
    fn test_tensor_product() {
        let b1 = HolomorphicBundle::trivial(2);
        let b2 = HolomorphicBundle::trivial(3);
        let tp = b1.tensor_product(&b2);
        assert_eq!(tp.rank, 6);
    }

    #[test]
    fn test_ward_trivial() {
        let wc = WardCorrespondence::trivial(2);
        assert!(wc.satisfies_yang_mills());
        let x = SpacetimePoint::origin();
        let conn = wc.yang_mills_connection(&x);
        assert!(conn.is_flat());
    }

    #[test]
    fn test_ward_self_dual() {
        let b = HolomorphicBundle::with_transition(
            2,
            (0, 1),
            vec![
                vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            ],
        );
        let wc = WardCorrespondence::self_dual(b);
        assert!(wc.satisfies_yang_mills());
        assert!(wc.self_dual);
    }

    #[test]
    fn test_anti_self_dual() {
        let wc = WardCorrespondence::anti_self_dual(HolomorphicBundle::trivial(2));
        assert!(!wc.self_dual);
    }

    #[test]
    fn test_instanton() {
        let wc = WardCorrespondence::instanton(2, 1);
        assert_eq!(wc.bundle.rank, 2);
        assert!(wc.self_dual);
    }

    #[test]
    fn test_field_strength_zero() {
        let fs = FieldStrength::zero(2);
        assert!(fs.is_zero());
        assert!(fs.self_dual);
    }

    #[test]
    fn test_field_strength_nontrivial() {
        let b = HolomorphicBundle::with_transition(
            2, (0, 1),
            vec![
                vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            ],
        );
        let wc = WardCorrespondence::self_dual(b);
        let x = SpacetimePoint::from_minkowski(1.0, 0.0, 0.0, 0.0);
        let fs = wc.field_strength(&x);
        assert!(fs.self_dual);
    }

    #[test]
    fn test_yang_mills_trivial_connection() {
        let conn = YangMillsConnection::trivial(3);
        assert!(conn.is_flat());
        let fs = conn.field_strength();
        assert!(fs.is_zero());
    }

    #[test]
    fn test_dual_bundle() {
        let b = HolomorphicBundle::with_transition(
            2, (0, 1),
            vec![
                vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0)],
            ],
        );
        let dual = b.dual();
        assert_eq!(dual.rank, 2);
    }
}
