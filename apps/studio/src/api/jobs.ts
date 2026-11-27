/** Jobs on the client side. Starting a job answers at once with its id; its
 * progress and its end arrive on the back end's event stream. A `JobHub`
 * keeps one subscription open while a job it follows is live, routes each
 * job's events to its `Job`, and settles the job's `result` when it ends. */
import type { JobInfo, JobState, ServerEvent } from "./generated/types";
import { StudioError, type Subscription, type Transport } from "./transport";

/** A job being followed. */
export interface Job<E, R> {
  id: string;
  /** Hear each progress report; the reports sent before the first listener
   * are replayed to it. Returns the unsubscribe function. */
  onEvent(listener: (event: E) => void): () => void;
  /** The output once the job is done; rejects with its `StudioError` when
   * it fails, and with kind `cancelled` when it is cancelled. */
  result: Promise<R>;
  /** Ask the back end to cancel the job (a no-op once it has ended). */
  cancel(): Promise<void>;
}

/** The job commands a `Job` needs, from the generated client. */
export interface JobControl {
  status(id: string): Promise<JobInfo>;
  cancel(id: string): Promise<JobInfo>;
}

/** Reports kept for a job nobody listens to yet. */
const MAX_BUFFERED = 1000;
/** Jobs kept that nobody follows (another tab's, a finished earlier one). */
const MAX_UNFOLLOWED = 32;

interface Entry {
  followed: boolean;
  events: unknown[];
  listeners: Set<(event: unknown) => void>;
  terminal: JobState | null;
  settle: ((state: JobState) => void) | null;
}

function isTerminal(state: JobState): boolean {
  return state.kind === "done" || state.kind === "failed" || state.kind === "cancelled";
}

/** Whether the event stream is up. */
export type Connection = "connecting" | "open" | "lost";

/** One subscription to the event stream, shared by every job followed and
 * by the app's standing watch (`connect`). */
export class JobHub {
  private readonly transport: Transport;
  private subscription: Subscription | null = null;
  private lastSeq: number | undefined;
  private readonly entries = new Map<string, Entry>();
  /** Starts in flight: the stream stays open for them. */
  private holds = 0;
  /** The app watches the stream for good (`connect`). */
  private standing = false;
  private connection: Connection = "connecting";
  private readonly listeners = new Set<(event: ServerEvent) => void>();
  private readonly connectionListeners = new Set<(c: Connection) => void>();

  constructor(transport: Transport) {
    this.transport = transport;
  }

  /** Forget everything and close the stream (the transport changed). */
  reset(): void {
    this.subscription?.close();
    this.subscription = null;
    this.lastSeq = undefined;
    this.entries.clear();
    this.holds = 0;
    this.standing = false;
    this.setConnection("connecting");
  }

  /** Keep the stream open for good, whatever jobs come and go: the app's
   * standing watch of the back end. Resolves once it is registered. */
  connect(): Promise<void> {
    this.standing = true;
    return this.open();
  }

  /** End the standing watch (the stream closes when no job needs it). */
  disconnect(): void {
    this.standing = false;
    this.closeIfIdle();
  }

  /** Hear every event of the stream, jobs' and projects' alike. Returns
   * the unsubscribe function. */
  listen(listener: (event: ServerEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Hear the stream go up and down; `listener` is told the current state
   * at once. Returns the unsubscribe function. */
  onConnection(listener: (c: Connection) => void): () => void {
    this.connectionListeners.add(listener);
    listener(this.connection);
    return () => this.connectionListeners.delete(listener);
  }

  private setConnection(next: Connection): void {
    if (this.connection === next) return;
    this.connection = next;
    for (const listener of [...this.connectionListeners]) listener(next);
  }

  private open(): Promise<void> {
    if (!this.subscription) {
      this.subscription = this.transport.subscribe(
        (e) => this.dispatch(e),
        this.lastSeq,
        (open) => this.setConnection(open ? "open" : "lost"),
      );
    }
    return this.subscription.ready;
  }

  /** Keep the stream open for a job about to be followed (until
   * `release`); resolves once the stream is registered. */
  hold(): Promise<void> {
    this.holds += 1;
    return this.open();
  }

  release(): void {
    this.holds -= 1;
    this.closeIfIdle();
  }

  private closeIfIdle(): void {
    const live = [...this.entries.values()].some((e) => e.followed && !e.terminal);
    if (!live && this.holds === 0 && !this.standing) {
      this.subscription?.close();
      this.subscription = null;
      this.setConnection("connecting");
    }
  }

  private entry(id: string): Entry {
    let entry = this.entries.get(id);
    if (!entry) {
      entry = {
        followed: false,
        events: [],
        listeners: new Set(),
        terminal: null,
        settle: null,
      };
      this.entries.set(id, entry);
      this.forgetUnfollowed();
    }
    return entry;
  }

  private forgetUnfollowed(): void {
    const unfollowed = [...this.entries].filter(([, e]) => !e.followed);
    for (const [id] of unfollowed.slice(
      0,
      Math.max(0, unfollowed.length - MAX_UNFOLLOWED),
    )) {
      this.entries.delete(id);
    }
  }

  private dispatch(event: ServerEvent): void {
    this.lastSeq = event.seq;
    for (const listener of [...this.listeners]) listener(event);
    const body = event.body;
    if (body.kind !== "job") return;
    const entry = this.entry(body.id);
    if (entry.terminal) return;
    if (body.event !== undefined && body.event !== null) {
      if (entry.listeners.size > 0) {
        for (const listener of entry.listeners) listener(body.event);
      } else if (entry.events.length < MAX_BUFFERED) {
        entry.events.push(body.event);
      }
    }
    if (isTerminal(body.state)) this.end(body.id, body.state);
  }

  /** Job `id` reached `state`: settle it and close the stream when no
   * followed job is live. */
  end(id: string, state: JobState): void {
    const entry = this.entry(id);
    if (entry.terminal) return;
    entry.terminal = state;
    entry.settle?.(state);
    if (entry.followed) this.entries.delete(id);
    this.closeIfIdle();
  }

  /** Follow job `id`: its events and its end. */
  follow<E, R>(id: string, control: JobControl): Job<E, R> {
    const entry = this.entry(id);
    entry.followed = true;
    const result = new Promise<R>((resolve, reject) => {
      entry.settle = (state) => {
        if (state.kind === "done") resolve(state.output as R);
        else if (state.kind === "failed") reject(new StudioError(state.error));
        else reject(new StudioError({ kind: "cancelled", message: "cancelled" }));
      };
    });
    // Settled before anyone followed it (it ended while the start was in
    // flight).
    if (entry.terminal) {
      const state = entry.terminal;
      entry.terminal = null;
      this.end(id, state);
    }
    return {
      id,
      result,
      onEvent(listener) {
        const typed = listener as (event: unknown) => void;
        if (entry.listeners.size === 0) {
          for (const event of entry.events.splice(0)) typed(event);
        }
        entry.listeners.add(typed);
        return () => entry.listeners.delete(typed);
      },
      cancel: async () => {
        await control.cancel(id);
      },
    };
  }
}

/** Start job `name` with `args` and follow it. */
export async function startJob<E, R>(
  transport: Transport,
  hub: JobHub,
  control: JobControl,
  name: string,
  args: unknown,
): Promise<Job<E, R>> {
  // Subscribe first, so no event of the job can come before the stream.
  await hub.hold();
  let info: JobInfo;
  try {
    info = (await transport.invoke(name, args)) as JobInfo;
  } catch (e) {
    hub.release();
    throw e;
  }
  const job = hub.follow<E, R>(info.id, control);
  hub.release();
  return job;
}

/** Follow job `id`, started earlier (before a reload, by another tab). */
export async function attachJob<E, R>(
  hub: JobHub,
  control: JobControl,
  id: string,
): Promise<Job<E, R>> {
  await hub.hold();
  const job = hub.follow<E, R>(id, control);
  hub.release();
  try {
    const info = await control.status(id);
    if (isTerminal(info.state)) hub.end(id, info.state);
  } catch (e) {
    // No such job (any more): it ends with that error.
    const error =
      e instanceof StudioError
        ? e
        : new StudioError({ kind: "internal", message: String(e) });
    hub.end(id, { kind: "failed", error });
  }
  return job;
}
