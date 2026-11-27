/** The state of the New project dialog: one object the steps read and
 * patch. Nothing here talks to the back end; the steps do, and the project
 * is written by the last one. */
import { useCallback, useState } from "react";
import type { DatasetSpec } from "../../../api/generated/types";
import { NO_DEVICE_DRAFT, type DeviceDraft } from "../../../project/deviceDraft";
import type { DataSource, NewStep } from "./newProjectModel";

export interface NewProjectState {
  step: NewStep;
  source: DataSource | null;
  /** The draft manifest (the sniffer's or the file's), edited by the steps. */
  manifest: DatasetSpec | null;
  device: DeviceDraft;
  /** The project's name; `null` until the Create step offers the default. */
  name: string | null;
  /** The project folder; `null` until chosen or defaulted. */
  projectDir: string | null;
  /** The project file lives in the data folder itself. */
  inDataFolder: boolean;
  /** The farthest step reached, so Back never forgets what is ahead. */
  reached: number;
}

const INITIAL: NewProjectState = {
  step: "data",
  source: null,
  manifest: null,
  device: NO_DEVICE_DRAFT,
  name: null,
  projectDir: null,
  inDataFolder: false,
  reached: 0,
};

export function useNewProject(start?: Partial<NewProjectState>) {
  const [state, setState] = useState<NewProjectState>({ ...INITIAL, ...start });
  const patch = useCallback(
    (next: Partial<NewProjectState>) => setState((s) => ({ ...s, ...next })),
    [],
  );
  return { state, patch };
}

export type NewProjectApi = ReturnType<typeof useNewProject>;
