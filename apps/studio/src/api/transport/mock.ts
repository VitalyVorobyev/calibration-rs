/** A transport for tests: plain commands answer from typed handlers, jobs
 * run their handler and announce their states and events as the back end
 * would, and every call is recorded. A handler's argument and reply types
 * are the command table's, so a wrong fixture shape fails `tsc`. */
import type { Commands, Jobs, Resources } from "../generated/client";
import type { JobInfo, JobState, ServerEvent, ViewerSession } from "../generated/types";
import {
  asStudioError,
  type BackendKind,
  type OpenDialog,
  type Transport,
} from "../transport";

type Awaitable<T> = T | Promise<T>;

export type CommandHandlers = {
  [K in keyof Commands]?: (args: Commands[K]["args"]) => Awaitable<Commands[K]["reply"]>;
};

/** What a job handler gets besides its arguments. */
export interface MockJobControl<E> {
  emit(event: E): void;
  /** Resolves when the job is cancelled. */
  cancelled: Promise<void>;
}

export type JobHandlers = {
  [K in keyof Jobs]?: (
    args: Jobs[K]["args"],
    control: MockJobControl<Jobs[K]["event"]>,
  ) => Awaitable<Jobs[K]["output"]>;
};

export type ResourceHandlers = {
  [K in keyof Resources]?: (query: Resources[K]) => Awaitable<ArrayBuffer>;
};

export interface MockOptions {
  kind?: BackendKind;
  commands?: CommandHandlers;
  jobs?: JobHandlers;
  resources?: ResourceHandlers;
  dialog?: (options: OpenDialog) => string | null;
  session?: ViewerSession;
}

export interface MockTransport extends Transport {
  /** Every command and job started, in order. */
  calls: { command: string; args: unknown }[];
  /** Send an event to the subscribers. */
  emit(event: ServerEvent["body"]): void;
}

export function createMockTransport(options: MockOptions = {}): MockTransport {
  const subscribers = new Set<(event: ServerEvent) => void>();
  const jobs = new Map<string, { info: JobInfo; cancel: () => void }>();
  let seq = 0;
  let counter = 0;

  const emit = (body: ServerEvent["body"]) => {
    seq += 1;
    for (const subscriber of [...subscribers]) subscriber({ seq, body });
  };
  const announce = (info: JobInfo, state: JobState, event?: unknown) => {
    info.state = state;
    emit({ kind: "job", id: info.id, command: info.command, state, event });
  };

  const startJob = (name: string, args: unknown): JobInfo => {
    const handler = (options.jobs as Record<string, unknown> | undefined)?.[name] as
      | ((args: unknown, control: MockJobControl<unknown>) => Awaitable<unknown>)
      | undefined;
    if (!handler)
      throw asStudioError({ kind: "unknown_command", message: `no mock job ${name}` });
    counter += 1;
    const info: JobInfo = {
      id: `${counter}-mock`,
      command: name,
      state: { kind: "queued", position: 1 },
    };
    let cancel!: () => void;
    const cancelled = new Promise<void>((resolve) => (cancel = resolve));
    let isCancelled = false;
    jobs.set(info.id, {
      info,
      cancel: () => {
        isCancelled = true;
        cancel();
      },
    });
    // Run once the caller has the job's id, as a real back end would.
    setTimeout(() => {
      announce(info, { kind: "running" });
      const control: MockJobControl<unknown> = {
        emit: (event) => {
          info.last_event = event;
          announce(info, { kind: "running" }, event);
        },
        cancelled,
      };
      Promise.resolve()
        .then(() => handler(args, control))
        .then(
          (output) =>
            announce(
              info,
              isCancelled ? { kind: "cancelled" } : { kind: "done", output },
            ),
          (e: unknown) => {
            const error = asStudioError(e);
            announce(
              info,
              error.kind === "cancelled"
                ? { kind: "cancelled" }
                : { kind: "failed", error: { kind: error.kind, message: error.message } },
            );
          },
        );
    }, 0);
    return { ...info };
  };

  const { dialog, session } = options;
  const transport: MockTransport = {
    kind: options.kind ?? "tauri",
    calls: [],
    emit,

    async invoke(name, args) {
      transport.calls.push({ command: name, args });
      if (name === "job_status" || name === "cancel_job") {
        const { id } = args as { id: string };
        const job = jobs.get(id);
        if (!job) throw asStudioError({ kind: "job_not_found", message: `no job ${id}` });
        if (name === "cancel_job") job.cancel();
        return { ...job.info };
      }
      if (options.jobs && name in options.jobs) return startJob(name, args);
      const handler = (options.commands as Record<string, unknown> | undefined)?.[
        name
      ] as ((args: unknown) => Awaitable<unknown>) | undefined;
      if (!handler) return null;
      try {
        return await handler(args);
      } catch (e) {
        throw asStudioError(e);
      }
    },

    subscribe(onEvent) {
      subscribers.add(onEvent);
      return { ready: Promise.resolve(), close: () => subscribers.delete(onEvent) };
    },

    async fetchResource(name, query) {
      const handler = (options.resources as Record<string, unknown> | undefined)?.[
        name
      ] as ((query: unknown) => Awaitable<ArrayBuffer>) | undefined;
      if (!handler)
        throw asStudioError({ kind: "unknown_command", message: `no mock ${name}` });
      return handler(query);
    },

    // A one-pixel PNG stands in for every source image.
    fileUrl: () =>
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",

    ...(dialog ? { openDialog: (o: OpenDialog) => Promise.resolve(dialog(o)) } : {}),
    ...(session ? { session: () => Promise.resolve(session) } : {}),
  };
  return transport;
}
