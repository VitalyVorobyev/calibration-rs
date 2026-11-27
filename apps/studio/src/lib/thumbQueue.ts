/** Thumbnails for a grid of hundreds of cells: at most `limit` loads at a
 * time, the most recently asked for first (the cells in view when the user
 * scrolled), a load nobody waits for any more never started, and the
 * answers kept (least recently used dropped past `keep`) so scrolling back
 * shows them at once. The loader and the release of what it made are
 * injected: the browser's is `fetch` into an object URL (`imageLoader`).
 * The grid never waits on any of this: a cell shows a placeholder until its
 * state says `ready`. */

export type ThumbState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready"; src: string }
  | { kind: "error"; message: string };

const IDLE: ThumbState = { kind: "idle" };

interface Entry {
  state: ThumbState;
  listeners: Set<() => void>;
  /** Asked for, not started. */
  queued: boolean;
  /** Last use, for the LRU. */
  used: number;
}

export interface ThumbLoader {
  /** Load `url`; resolves to what the cell can show. */
  load: (url: string) => Promise<string>;
  /** Release what `load` made (an object URL). */
  release: (src: string) => void;
}

export class ThumbQueue {
  private readonly entries = new Map<string, Entry>();
  /** Urls waiting to start, the newest last. */
  private pending: string[] = [];
  private running = 0;
  private tick = 0;

  constructor(
    private readonly loader: ThumbLoader,
    private readonly limit = 6,
    private readonly keep = 800,
  ) {}

  /** The state of `url` (a stable object until it changes). */
  get = (url: string): ThumbState => this.entries.get(url)?.state ?? IDLE;

  /** Ask for `url` and hear of its changes; the returned function stops
   * listening (and drops the load if it has not started and no one else
   * waits for it). */
  subscribe(url: string, listener: () => void): () => void {
    let entry = this.entries.get(url);
    if (!entry) {
      entry = { state: IDLE, listeners: new Set(), queued: false, used: 0 };
      this.entries.set(url, entry);
    }
    entry.listeners.add(listener);
    entry.used = ++this.tick;
    if (entry.state.kind === "idle" && !entry.queued) {
      entry.queued = true;
      this.pending.push(url);
      this.pump();
    } else if (entry.queued) {
      // Asked for again: it moves to the front of the line.
      this.pending = this.pending.filter((u) => u !== url);
      this.pending.push(url);
    }
    return () => {
      const e = this.entries.get(url);
      if (!e) return;
      e.listeners.delete(listener);
      if (e.listeners.size === 0 && e.queued) {
        e.queued = false;
        this.pending = this.pending.filter((u) => u !== url);
        if (e.state.kind === "idle") this.entries.delete(url);
      }
    };
  }

  /** Forget everything (a new session, a changed back end). */
  clear(): void {
    for (const e of this.entries.values()) {
      if (e.state.kind === "ready") this.loader.release(e.state.src);
    }
    this.entries.clear();
    this.pending = [];
  }

  private set(entry: Entry, state: ThumbState): void {
    entry.state = state;
    for (const l of [...entry.listeners]) l();
  }

  private pump(): void {
    while (this.running < this.limit && this.pending.length > 0) {
      const url = this.pending.pop()!;
      const entry = this.entries.get(url);
      if (!entry?.queued) continue;
      entry.queued = false;
      this.running++;
      this.set(entry, { kind: "loading" });
      this.loader.load(url).then(
        (src) => {
          this.running--;
          this.set(entry, { kind: "ready", src });
          this.evict();
          this.pump();
        },
        (e: unknown) => {
          this.running--;
          this.set(entry, {
            kind: "error",
            message: e instanceof Error ? e.message : String(e),
          });
          this.pump();
        },
      );
    }
  }

  /** Drop the least recently used answers past `keep` that nobody shows. */
  private evict(): void {
    if (this.entries.size <= this.keep) return;
    const idle = [...this.entries.entries()]
      .filter(([, e]) => e.listeners.size === 0 && e.state.kind !== "loading")
      .sort(([, a], [, b]) => a.used - b.used);
    for (const [url, e] of idle) {
      if (this.entries.size <= this.keep) break;
      if (e.state.kind === "ready") this.loader.release(e.state.src);
      this.entries.delete(url);
    }
  }
}

/** The browser's loader: `fetch` the image into an object URL. */
export const imageLoader: ThumbLoader = {
  load: async (url) => {
    const response = await fetch(url, { credentials: "same-origin" });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`.trim());
    return URL.createObjectURL(await response.blob());
  },
  release: (src) => URL.revokeObjectURL(src),
};
