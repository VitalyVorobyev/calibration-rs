//! The commands of the table, by what they serve. Each is a function of the
//! [`Ctx`](super::Ctx) and its argument object; the argument object's
//! rustdoc is the command's documentation in the generated client.

pub mod compare;
pub mod detection;
pub mod device;
pub mod device_series;
pub mod files;
pub mod forms;
pub mod jobs;
mod map_support;
pub mod maps;
pub mod project;
pub mod resources;
pub mod runs;
pub mod seed;
pub mod session;
pub mod setup;
pub mod viewer;
pub mod views;

#[cfg(test)]
mod compare_tests;
#[cfg(test)]
mod detection_tests;
#[cfg(test)]
mod device_tests;
#[cfg(test)]
mod fixture_tests;
#[cfg(test)]
mod maps_tests;
#[cfg(test)]
mod project_tests;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod views_tests;
