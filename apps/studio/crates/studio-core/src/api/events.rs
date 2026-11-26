//! The event stream: every job's state changes and progress, numbered, to
//! each subscriber. A subscriber that comes back with the last number it saw
//! gets what it missed from a short history; one that comes fresh (or
//! fell too far behind) gets each live job's latest state first.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard};

use schemars::JsonSchema;
use serde::Serialize;

use super::jobs::JobState;

/// Events kept for subscribers that resume.
const HISTORY: usize = 512;

/// One numbered event.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ServerEvent {
    /// Process-wide, increasing; a subscriber resumes after the last one it
    /// saw.
    pub seq: u64,
    pub body: EventBody,
}

/// What happened.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventBody {
    /// A job changed state or reported progress.
    Job(JobEvent),
}

/// A job's state, and the progress report that came with it.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct JobEvent {
    /// The job's id.
    pub id: String,
    /// The job command that started it.
    pub command: String,
    pub state: JobState,
    /// The job's progress report (its command's event type), when this
    /// event carries one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<serde_json::Value>,
}

/// A subscription: its id (to end it) and the events, in order.
pub struct Subscription {
    pub id: u64,
    pub events: Receiver<ServerEvent>,
}

#[derive(Default)]
struct BusState {
    seq: u64,
    history: VecDeque<ServerEvent>,
    next_subscriber: u64,
    subscribers: Vec<(u64, Sender<ServerEvent>)>,
}

/// The process's event stream. Cheap to clone; every clone is the same bus.
#[derive(Clone, Default)]
pub struct EventBus(Arc<Mutex<BusState>>);

impl EventBus {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, BusState> {
        // The state is a counter, a ring and a list; a panic elsewhere
        // leaves each of them whole.
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Number `body`, keep it for resuming subscribers and send it to every
    /// subscriber; one whose receiver is gone is dropped.
    pub fn publish(&self, body: EventBody) {
        let mut bus = self.lock();
        bus.seq += 1;
        let event = ServerEvent { seq: bus.seq, body };
        if bus.history.len() == HISTORY {
            bus.history.pop_front();
        }
        bus.history.push_back(event.clone());
        bus.subscribers
            .retain(|(_, sender)| sender.send(event.clone()).is_ok());
    }

    /// A new subscription. With `after`, the events after it from the
    /// history when it still holds them all; otherwise `live` — each live
    /// job's latest state, numbered as the newest event so far — first.
    /// `live` runs under the bus lock, so no event published meanwhile is
    /// lost or sent before it.
    pub fn subscribe(
        &self,
        after: Option<u64>,
        live: impl FnOnce() -> Vec<EventBody>,
    ) -> Subscription {
        let (sender, events) = channel();
        let mut bus = self.lock();
        let oldest = bus.history.front().map_or(bus.seq + 1, |e| e.seq);
        match after {
            Some(after) if after + 1 >= oldest && after <= bus.seq => {
                for event in bus.history.iter().filter(|e| e.seq > after) {
                    let _ = sender.send(event.clone());
                }
            }
            _ => {
                for body in live() {
                    let _ = sender.send(ServerEvent { seq: bus.seq, body });
                }
            }
        }
        bus.next_subscriber += 1;
        let id = bus.next_subscriber;
        bus.subscribers.push((id, sender));
        Subscription { id, events }
    }

    /// End subscription `id`: its receiver sees the stream close.
    pub fn unsubscribe(&self, id: u64) {
        self.lock().subscribers.retain(|(sub, _)| *sub != id);
    }

    /// The newest event's number (0 before any).
    pub fn last_seq(&self) -> u64 {
        self.lock().seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str, state: JobState) -> EventBody {
        EventBody::Job(JobEvent {
            id: id.into(),
            command: "c".into(),
            state,
            event: None,
        })
    }

    fn drain(sub: &Subscription) -> Vec<ServerEvent> {
        sub.events.try_iter().collect()
    }

    #[test]
    fn a_resuming_subscriber_gets_what_it_missed_in_order() {
        let bus = EventBus::new();
        for i in 0..5 {
            bus.publish(job(&i.to_string(), JobState::Running));
        }
        let sub = bus.subscribe(Some(3), Vec::new);
        let seqs: Vec<u64> = drain(&sub).iter().map(|e| e.seq).collect();
        assert_eq!(seqs, [4, 5]);
        bus.publish(job("x", JobState::Cancelled));
        assert_eq!(drain(&sub)[0].seq, 6);
        bus.unsubscribe(sub.id);
        bus.publish(job("y", JobState::Cancelled));
        assert!(sub.events.recv().is_err(), "the stream closes");
    }

    #[test]
    fn a_fresh_or_lost_subscriber_gets_the_live_jobs_first() {
        let bus = EventBus::new();
        for i in 0..(HISTORY + 10) {
            bus.publish(job(&i.to_string(), JobState::Running));
        }
        let live = || vec![job("live", JobState::Running)];
        // Too old to replay: the live states instead.
        let sub = bus.subscribe(Some(1), live);
        let got = drain(&sub);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].seq, bus.last_seq());
        // Fresh.
        let sub = bus.subscribe(None, Vec::new);
        assert!(drain(&sub).is_empty());
        // Ahead of the bus (a server restart): the live states too.
        let sub = bus.subscribe(Some(u64::MAX - 1), live);
        assert_eq!(drain(&sub).len(), 1);
    }

    #[test]
    fn events_serialize_with_their_kind_and_state() {
        let event = ServerEvent {
            seq: 7,
            body: job("1-ab", JobState::Queued { position: 2 }),
        };
        assert_eq!(
            serde_json::to_value(&event).unwrap(),
            serde_json::json!({
                "seq": 7,
                "body": {
                    "kind": "job", "id": "1-ab", "command": "c",
                    "state": {"kind": "queued", "position": 2}
                }
            })
        );
    }
}
