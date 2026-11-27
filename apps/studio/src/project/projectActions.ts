/** The ways to a project, as the menu and the Run workspace offer them:
 * open one, make one of a dataset folder (the sniffer drafts its
 * manifest) or of a manifest file (referenced, never written), fetch a
 * Quick Start sample as one, and import an export into the open one. Each
 * failure lands in the project store's `error`, which the shell shows. */
import { toast } from "@vitavision/ui";
import { api } from "../api";
import type { ProjectSpec } from "../api/generated/types";
import { dirnamePath } from "../lib/paths";
import {
  pickExportFile,
  pickFolder,
  pickManifestFile,
  pickProject,
} from "../shell/dialogs";
import { asProjectError, folderName, inlineSpec, referencedSpec } from "./projectModel";
import { useProject } from "./projectStore";

function fail(e: unknown): false {
  useProject.setState({ error: asProjectError(e) });
  return false;
}

/** Make `dir` a project of `spec`; a folder that already holds one is
 * opened as it is. */
async function createOrOpen(dir: string, spec: ProjectSpec): Promise<boolean> {
  const store = useProject.getState();
  if (await store.create(dir, spec)) {
    toast({ title: "Project created", description: dir, tone: "success" });
    return true;
  }
  if (useProject.getState().error?.kind !== "project_exists") return false;
  if (!(await store.open(dir))) return false;
  toast({ title: "This folder already holds a project: opened it", description: dir });
  return true;
}

/** Open a project chosen in the picker (or `dir`). */
export async function openProject(dir?: string): Promise<boolean> {
  const chosen = dir ?? (await pickProject());
  if (chosen === null) return false;
  return useProject.getState().open(chosen);
}

/** A new project in a dataset folder, its manifest drafted from the
 * folder's files; what the sniffer leaves open stays in `_unresolved` and
 * blocks a run until it is filled in. */
export async function newProjectFromFolder(): Promise<boolean> {
  const folder = await pickFolder("Pick the dataset folder of the new project");
  if (folder === null) return false;
  try {
    const sniffed = await api.sniffFolder({ folder });
    return await createOrOpen(folder, inlineSpec(sniffed.spec));
  } catch (e) {
    return fail(e);
  }
}

/** A new project beside a manifest file, referencing it. */
export async function newProjectFromManifest(): Promise<boolean> {
  const path = await pickManifestFile();
  if (path === null) return false;
  return createOrOpen(dirnamePath(path), referencedSpec(folderName(path)));
}

/** The Quick Start sample `id`, fetched (on first use) as a project of
 * its own and opened. */
export async function openSample(id: string): Promise<boolean> {
  try {
    const sample = await (await api.jobs.fetchSample({ id })).result;
    return await useProject.getState().open(sample.dir);
  } catch (e) {
    return fail(e);
  }
}

/** Add an export chosen in the picker to the open project as its next
 * run, and show it. */
export async function importExport(): Promise<boolean> {
  if (!useProject.getState().project) return false;
  const path = await pickExportFile();
  if (path === null) return false;
  return useProject.getState().importExport(path);
}
