//! Agent trajectories: twistor-theoretic description of agent dynamics.
//!
//! An agent's state is encoded as a twistor, and its trajectory through spacetime
//! is a curve whose twistor description captures both position and momentum information.
//! This enables holomorphic (complex-analytic) descriptions of agent dynamics.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::incidence::SpacetimePoint;
use crate::spinor::PrimedSpinor;
use crate::twistor::Twistor;

/// An agent's twistorial state at a moment in time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AgentState {
    /// The twistor encoding the agent's state.
    pub twistor: Twistor,
    /// Timestamp (affine parameter along the trajectory).
    pub parameter: f64,
}

impl AgentState {
    /// Create a new agent state.
    pub fn new(twistor: Twistor, parameter: f64) -> Self {
        Self { twistor, parameter }
    }

    /// Create from a spacetime point and momentum direction.
    pub fn from_spacetime(x: &SpacetimePoint, pi: &PrimedSpinor, parameter: f64) -> Self {
        let twistor = x.twistor_line(pi);
        Self { twistor, parameter }
    }

    /// The spacetime point associated with this agent state.
    pub fn spacetime_point(&self) -> Option<SpacetimePoint> {
        // Reconstruct from the twistor's associated spacetime point
        let x_matrix = self.twistor.associated_spacetime_point()?;
        Some(SpacetimePoint { coords: x_matrix })
    }

    /// Is this a null state (on the light cone)?
    pub fn is_null(&self) -> bool {
        self.twistor.is_null()
    }

    /// The frequency sign of this state.
    pub fn frequency_sign(&self) -> FrequencySign {
        if self.twistor.is_positive_frequency() {
            FrequencySign::Positive
        } else if self.twistor.is_negative_frequency() {
            FrequencySign::Negative
        } else {
            FrequencySign::Zero
        }
    }
}

/// Frequency sign of an agent state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrequencySign {
    Positive,
    Negative,
    Zero,
}

/// An agent trajectory: a sequence of twistorial states parameterized by time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTrajectory {
    /// The states along the trajectory.
    pub states: Vec<AgentState>,
}

impl Default for AgentTrajectory {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentTrajectory {
    /// Create an empty trajectory.
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Create from a vector of states.
    pub fn from_states(states: Vec<AgentState>) -> Self {
        Self { states }
    }

    /// Create a null geodesic trajectory from a starting point and direction.
    pub fn null_geodesic(
        start: &SpacetimePoint,
        direction: &PrimedSpinor,
        n_steps: usize,
        step_size: f64,
    ) -> Self {
        let mut states = Vec::new();
        for i in 0..n_steps {
            let lambda = i as f64 * step_size;
            let x = start.translate(
                lambda * 1.0, // simplified: actual direction from spinor
                0.0,
                0.0,
                0.0,
            );
            let twistor = x.twistor_line(direction);
            states.push(AgentState::new(twistor, lambda));
        }
        Self { states }
    }

    /// Add a state to the trajectory.
    pub fn push(&mut self, state: AgentState) {
        self.states.push(state);
    }

    /// Number of states.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Is the trajectory empty?
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Interpolate between two agent states.
    pub fn interpolate(s1: &AgentState, s2: &AgentState, t: f64) -> AgentState {
        let z1 = s1.twistor.to_vector4();
        let z2 = s2.twistor.to_vector4();
        let interpolated = z1 * Complex64::new(1.0 - t, 0.0) + z2 * Complex64::new(t, 0.0);
        AgentState::new(
            Twistor::from_vector4(interpolated),
            s1.parameter * (1.0 - t) + s2.parameter * t,
        )
    }

    /// Compute the twistorial "velocity" between consecutive states.
    pub fn twistorial_velocity(&self) -> Vec<Twistor> {
        let mut velocities = Vec::new();
        for i in 1..self.states.len() {
            let s0 = &self.states[i - 1];
            let s1 = &self.states[i];
            let dt = s1.parameter - s0.parameter;
            if dt.abs() > 1e-15 {
                let v0 = s0.twistor.to_vector4();
                let v1 = s1.twistor.to_vector4();
                let vel = (v1 - v0) * Complex64::new(1.0 / dt, 0.0);
                velocities.push(Twistor::from_vector4(vel));
            }
        }
        velocities
    }

    /// Resample the trajectory at uniform parameter intervals.
    pub fn resample(&self, n_samples: usize) -> AgentTrajectory {
        if self.states.len() < 2 || n_samples < 2 {
            return self.clone();
        }
        let p_min = self.states.first().unwrap().parameter;
        let p_max = self.states.last().unwrap().parameter;
        let mut new_states = Vec::new();
        for i in 0..n_samples {
            let t = i as f64 / (n_samples - 1) as f64;
            let p = p_min + t * (p_max - p_min);
            // Find bracketing states
            let idx = self.states.iter().position(|s| s.parameter >= p).unwrap_or(self.states.len() - 1);
            if idx == 0 {
                new_states.push(self.states[0]);
            } else {
                let s0 = &self.states[idx - 1];
                let s1 = &self.states[idx];
                let local_t = (p - s0.parameter) / (s1.parameter - s0.parameter);
                new_states.push(Self::interpolate(s0, s1, local_t));
            }
        }
        AgentTrajectory { states: new_states }
    }

    /// The total parameter range.
    pub fn parameter_range(&self) -> Option<(f64, f64)> {
        if self.states.is_empty() {
            return None;
        }
        let p_min = self.states.first().unwrap().parameter;
        let p_max = self.states.last().unwrap().parameter;
        Some((p_min, p_max))
    }

    /// Check if the trajectory stays within a frequency region.
    pub fn is_pure_frequency(&self) -> bool {
        if self.states.is_empty() {
            return true;
        }
        let first_sign = self.states[0].frequency_sign();
        self.states.iter().all(|s| s.frequency_sign() == first_sign)
    }
}

/// Interaction between two agent trajectories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInteraction {
    /// The first agent's trajectory.
    pub trajectory_a: AgentTrajectory,
    /// The second agent's trajectory.
    pub trajectory_b: AgentTrajectory,
}

impl AgentInteraction {
    /// Create a new interaction between two trajectories.
    pub fn new(a: AgentTrajectory, b: AgentTrajectory) -> Self {
        Self {
            trajectory_a: a,
            trajectory_b: b,
        }
    }

    /// Find points where the two trajectories are "closest" in twistor space.
    pub fn closest_approach(&self) -> Option<(AgentState, AgentState, f64)> {
        let mut best: Option<(AgentState, AgentState, f64)> = None;
        for sa in &self.trajectory_a.states {
            for sb in &self.trajectory_b.states {
                let va = sa.twistor.to_vector4();
                let vb = sb.twistor.to_vector4();
                let diff = va - vb;
                let dist = diff.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                if best.is_none() || dist < best.as_ref().unwrap().2 {
                    best = Some((*sa, *sb, dist));
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_agent_state_creation() {
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        let s = AgentState::new(z, 0.0);
        assert_abs_diff_eq!(s.parameter, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_agent_state_from_spacetime() {
        let x = SpacetimePoint::from_minkowski(1.0, 0.0, 0.0, 0.0);
        let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let s = AgentState::from_spacetime(&x, &pi, 0.0);
        assert!(s.twistor.norm() > 0.0);
    }

    #[test]
    fn test_agent_state_frequency() {
        let z = Twistor::from_components(
            Complex64::new(5.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        let s = AgentState::new(z, 0.0);
        assert_eq!(s.frequency_sign(), FrequencySign::Positive);
    }

    #[test]
    fn test_empty_trajectory() {
        let t = AgentTrajectory::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn test_trajectory_push() {
        let mut t = AgentTrajectory::new();
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        );
        t.push(AgentState::new(z, 0.0));
        t.push(AgentState::new(z, 1.0));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn test_trajectory_parameter_range() {
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        );
        let mut t = AgentTrajectory::new();
        t.push(AgentState::new(z, 0.0));
        t.push(AgentState::new(z, 1.0));
        let (p_min, p_max) = t.parameter_range().unwrap();
        assert_abs_diff_eq!(p_min, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p_max, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolate() {
        let z1 = Twistor::from_components(
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let z2 = Twistor::from_components(
            Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let s1 = AgentState::new(z1, 0.0);
        let s2 = AgentState::new(z2, 2.0);
        let mid = AgentTrajectory::interpolate(&s1, &s2, 0.5);
        assert_abs_diff_eq!(mid.twistor.omega.components[0].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mid.parameter, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_twistorial_velocity() {
        let z1 = Twistor::from_components(
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let z2 = Twistor::from_components(
            Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let mut t = AgentTrajectory::new();
        t.push(AgentState::new(z1, 0.0));
        t.push(AgentState::new(z2, 1.0));
        let vels = t.twistorial_velocity();
        assert_eq!(vels.len(), 1);
        assert_abs_diff_eq!(vels[0].omega.components[0].re, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pure_frequency() {
        let z = Twistor::from_components(
            Complex64::new(5.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        );
        let mut t = AgentTrajectory::new();
        t.push(AgentState::new(z, 0.0));
        t.push(AgentState::new(z, 1.0));
        assert!(t.is_pure_frequency());
    }

    #[test]
    fn test_closest_approach() {
        let z1 = Twistor::from_components(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        );
        let z2 = Twistor::from_components(
            Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        );
        let mut t1 = AgentTrajectory::new();
        t1.push(AgentState::new(z1, 0.0));
        let mut t2 = AgentTrajectory::new();
        t2.push(AgentState::new(z2, 0.0));
        let interaction = AgentInteraction::new(t1, t2);
        let (_, _, dist) = interaction.closest_approach().unwrap();
        assert!(dist > 0.0);
    }

    #[test]
    fn test_null_geodesic_trajectory() {
        let x = SpacetimePoint::origin();
        let pi = PrimedSpinor::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let t = AgentTrajectory::null_geodesic(&x, &pi, 5, 1.0);
        assert_eq!(t.len(), 5);
    }

    #[test]
    fn test_resample() {
        let z = Twistor::from_components(
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        );
        let mut t = AgentTrajectory::new();
        t.push(AgentState::new(z, 0.0));
        t.push(AgentState::new(z, 1.0));
        let resampled = t.resample(5);
        assert_eq!(resampled.len(), 5);
    }
}
