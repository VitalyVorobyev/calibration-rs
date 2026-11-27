//! The commands of the table, by what they serve. Each is a function of the
//! [`Ctx`](super::Ctx) and its argument object; the argument object's
//! rustdoc is the command's documentation in the generated client.

pub mod files;
pub mod forms;
pub mod jobs;
pub mod project;
pub mod resources;
pub mod runs;
pub mod session;
pub mod setup;
pub mod viewer;

#[cfg(test)]
mod fixture_tests;
#[cfg(test)]
mod project_tests;
