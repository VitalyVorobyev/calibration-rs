/** A small query cache over the typed client: one entry per `(command,
 * arguments)`, shared by every panel that asks, fetched once, and marked
 * stale (not dropped) when the back end says the data changed, so a panel
 * keeps showing the last good answer under a veil while the new one
 * arrives.
 *
 *     const q = useQuery("list_runs", { dir }, () => api.listRuns({ dir }));
 *
 * `project_changed { dir }` invalidates every entry that names `dir`
 * (`invalidateProject`, called by the shell's event handler). */
import { useCallback, useEffect, useSyncExternalStore } from "react";
import { asStudioError, type StudioError } from "../api";

/** What the cache holds for one key. */
export interface Entry<T> {
  /** The last good answer; kept while a newer one is fetched or fails. */
  data: T | undefined;
  /** The last failure, cleared by the next success. */
  error: StudioError | null;
  /** A fetch is in flight. */
  loading: boolean;
  /** The data changed since it was fetched. */
  stale: boolean;
}

const EMPTY: Entry<never> = { data: undefined, error: null, loading: false, stale: true };

/** JSON with sorted keys, so the same arguments are the same key. */
function canonical(value: unknown): string {
  return JSON.stringify(value, (_k, v: unknown) =>
    v !== null && typeof v === "object" && !Array.isArray(v)
      ? Object.fromEntries(
          Object.entries(v as Record<string, unknown>).sort(([a], [b]) =>
            a < b ? -1 : a > b ? 1 : 0,
          ),
        )
      : v,
  );
}

export function queryKey(command: string, args: unknown): string {
  return `${command}\u0000${canonical(args ?? null)}`;
}

/** The project folders a command's arguments name (`dir`, `run.dir`,
 * `project`), where the arguments are an object. */
function dirsOf(args: unknown): string[] {
  if (typeof args !== "object" || args === null) return [];
  const a = args as Record<string, unknown>;
  const dirs: string[] = [];
  if (typeof a["dir"] === "string") dirs.push(a["dir"]);
  if (typeof a["project"] === "string") dirs.push(a["project"]);
  const run = a["run"];
  if (typeof run === "object" && run !== null) {
    const dir = (run as Record<string, unknown>)["dir"];
    if (typeof dir === "string") dirs.push(dir);
  }
  return dirs;
}

interface Slot {
  entry: Entry<unknown>;
  args: unknown;
  command: string;
  inflight: Promise<void> | null;
  /** Invalidated while a fetch was in flight: fetch again after it. */
  again: boolean;
}

export class QueryCache {
  private readonly slots = new Map<string, Slot>();
  private readonly listeners = new Set<() => void>();

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  private notify(): void {
    for (const listener of [...this.listeners]) listener();
  }

  private slot(command: string, args: unknown): Slot {
    const key = queryKey(command, args);
    let slot = this.slots.get(key);
    if (!slot) {
      slot = { entry: EMPTY, args, command, inflight: null, again: false };
      this.slots.set(key, slot);
    }
    return slot;
  }

  /** The entry of `(command, args)`: a stable object until it changes. */
  get<T>(command: string, args: unknown): Entry<T> {
    return (this.slots.get(queryKey(command, args))?.entry ?? EMPTY) as Entry<T>;
  }

  /** Fetch `(command, args)` with `fetcher` unless a fresh answer is held
   * or a fetch is in flight (it joins it). Resolves when the entry
   * settles. */
  fetch<T>(command: string, args: unknown, fetcher: () => Promise<T>): Promise<void> {
    const slot = this.slot(command, args);
    if (slot.inflight) return slot.inflight;
    if (!slot.entry.stale && slot.entry.error === null) return Promise.resolve();
    slot.again = false;
    slot.entry = { ...slot.entry, loading: true };
    this.notify();
    slot.inflight = fetcher().then(
      (data) => {
        slot.entry = { data, error: null, loading: false, stale: slot.again };
      },
      (e: unknown) => {
        slot.entry = { ...slot.entry, error: asStudioError(e), loading: false };
      },
    );
    const settled = slot.inflight.finally(() => {
      slot.inflight = null;
      this.notify();
    });
    slot.inflight = settled;
    return settled;
  }

  /** Mark stale every entry `matches` accepts; mounted queries refetch. */
  invalidate(matches: (command: string, args: unknown) => boolean): void {
    let changed = false;
    for (const slot of this.slots.values()) {
      if (!matches(slot.command, slot.args)) continue;
      if (slot.inflight) slot.again = true;
      if (!slot.entry.stale) {
        slot.entry = { ...slot.entry, stale: true };
        changed = true;
      }
    }
    if (changed) this.notify();
  }

  /** Mark stale every entry about the project in `dir`. */
  invalidateProject(dir: string): void {
    this.invalidate((_command, args) => dirsOf(args).includes(dir));
  }

  /** Forget everything (a new session). */
  clear(): void {
    this.slots.clear();
    this.notify();
  }
}

export const queryCache = new QueryCache();

export interface Query<T> extends Entry<T> {
  refetch: () => void;
}

/** The answer of `fetcher` for `(command, args)`, shared and cached; pass
 * `enabled: false` to wait (no project open yet). `fetcher` need not be
 * stable: the key is what identifies the query. */
export function useQuery<T>(
  command: string,
  args: unknown,
  fetcher: () => Promise<T>,
  enabled = true,
): Query<T> {
  const key = queryKey(command, args);
  const entry = useSyncExternalStore(
    queryCache.subscribe,
    () => queryCache.get<T>(command, args),
    () => EMPTY as Entry<T>,
  );
  // The effect may run on every render (an unstable `args` or `fetcher`);
  // the guard and the cache's own join of a fetch in flight make that cheap.
  useEffect(() => {
    if (enabled && entry.stale && !entry.loading && entry.error === null) {
      void queryCache.fetch(command, args, fetcher);
    }
  }, [command, args, fetcher, enabled, entry.stale, entry.loading, entry.error]);
  const refetch = useCallback(() => {
    queryCache.invalidate((c, a) => queryKey(c, a) === key);
    void queryCache.fetch(command, args, fetcher);
  }, [command, args, fetcher, key]);
  return { ...entry, refetch };
}
