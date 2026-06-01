//! # lau-twistor-agents
//!
//! Penrose's twistor theory for agents — encoding spacetime events as complex geometric objects.
//!
//! Twistor theory replaces spacetime points with complex lines in twistor space.
//! For agents: replace agent states with twistors, and the dynamics become holomorphic
//! (complex-analytic).

pub mod spinor;
pub mod twistor;
pub mod incidence;
pub mod projective;
pub mod penrose_transform;
pub mod massless;
pub mod curved;
pub mod ward;
pub mod null_geodesic;
pub mod conformal;
pub mod agent;

pub use spinor::Spinor;
pub use twistor::Twistor;
pub use incidence::IncidenceRelation;
pub use projective::ProjectiveTwistor;
pub use penrose_transform::PenroseTransform;
pub use massless::MasslessField;
pub use penrose_transform::Helicity;
pub use curved::CurvedTwistorSpace;
pub use ward::WardCorrespondence;
pub use null_geodesic::NullGeodesic;
pub use conformal::ConformalGroup;
pub use agent::AgentTrajectory;
