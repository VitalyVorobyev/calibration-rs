import { afterEach, describe, expect, it } from "vitest";
import type { DatasetSpec, ProjectView } from "../api/generated/types";
import { MANIFEST, projectView } from "../test/projectFixtures";
import { isDirty, useDraft } from "./draftStore";
import { useProject } from "./projectStore";

afterEach(() => {
  useProject.setState({ project: null });
  useDraft.getState().resetFrom(null);
});

const manifestWith = (patch: Partial<DatasetSpec>): DatasetSpec => ({
  ...MANIFEST,
  ...patch,
});

/** `projectView` at `revision`, its manifest `manifest`. */
function version(revision: string, manifest: DatasetSpec): ProjectView {
  const view = projectView("/work/p");
  return {
    ...view,
    revision,
    spec: { ...view.spec, dataset: { root: ".", manifest } },
    dataset: { kind: "ok", spec: manifest },
  };
}

function open(view: ProjectView) {
  useProject.setState({ project: view });
}

const target = (m: DatasetSpec) => (m.target as unknown as { rows: number }).rows;
const withRows = (m: DatasetSpec, rows: number) =>
  manifestWith({ target: { ...m.target, rows } as DatasetSpec["target"] });

describe("the draft over a project that changes on disk", () => {
  it("starts again from a project it has no edits on", () => {
    open(version("r1", MANIFEST));
    expect(useDraft.getState().baseManifest).toEqual(MANIFEST);
    open(version("r2", withRows(MANIFEST, 9)));
    expect(target(useDraft.getState().manifest!)).toBe(9);
    expect(useDraft.getState().disk).toBeNull();
  });

  it("keeps unsaved edits when only another part of the project file changed", () => {
    open(version("r1", MANIFEST));
    useDraft.getState().setManifest(withRows(MANIFEST, 7));
    // The device was set: the manifest is what it was.
    const next = { ...version("r2", MANIFEST), name: "renamed" };
    open(next);
    const draft = useDraft.getState();
    expect(isDirty(draft)).toBe(true);
    expect(target(draft.manifest!)).toBe(7);
    expect(draft.disk).toBeNull();
    expect(draft.base).toBe("/work/p|r2");
  });

  it("keeps unsaved edits and says so when the manifest changed on disk too", () => {
    open(version("r1", MANIFEST));
    useDraft.getState().setManifest(withRows(MANIFEST, 7));
    open(version("r2", manifestWith({ description: "edited elsewhere" })));
    const draft = useDraft.getState();
    expect(target(draft.manifest!)).toBe(7);
    expect(draft.disk?.revision).toBe("r2");
    // Still on the version it started from: a save over r1 would be refused.
    expect(draft.base).toBe("/work/p|r1");
  });

  it("puts my edits back on the disk's version: mine win where both changed", () => {
    open(version("r1", MANIFEST));
    useDraft.getState().setManifest(withRows(MANIFEST, 7));
    useDraft.getState().setManifest({
      ...withRows(MANIFEST, 7),
      description: "mine",
    });
    open(
      version("r2", {
        ...withRows(MANIFEST, 3),
        description: "theirs",
        detector: { min_features_per_view: 12 },
      }),
    );
    useDraft.getState().rebase();
    const draft = useDraft.getState();
    expect(target(draft.manifest!)).toBe(7);
    expect(draft.manifest?.description).toBe("mine");
    // What only the disk changed is taken.
    expect(draft.manifest?.detector).toEqual({ min_features_per_view: 12 });
    // The baseline is the disk's, so the forms mark what differs from it.
    expect(target(draft.baseManifest!)).toBe(3);
    expect(draft.baseManifest?.description).toBe("theirs");
    expect(draft.manifestDirty).toBe(true);
    expect(draft.disk).toBeNull();
    expect(draft.base).toBe("/work/p|r2");
  });

  it("is clean again when the disk has what I had edited to", () => {
    open(version("r1", MANIFEST));
    useDraft.getState().setManifest(withRows(MANIFEST, 7));
    open(version("r2", withRows(MANIFEST, 7)));
    useDraft.getState().rebase();
    expect(useDraft.getState().manifestDirty).toBe(false);
  });

  it("can keep editing over the disk's version, and revert to it", () => {
    open(version("r1", MANIFEST));
    useDraft.getState().setManifest(withRows(MANIFEST, 7));
    open(version("r2", withRows(MANIFEST, 3)));
    useDraft.getState().keepEditing();
    expect(useDraft.getState().disk).toBeNull();
    expect(useDraft.getState().base).toBe("/work/p|r2");
    expect(target(useDraft.getState().manifest!)).toBe(7);
    useDraft.getState().revert();
    expect(target(useDraft.getState().manifest!)).toBe(3);
    expect(isDirty(useDraft.getState())).toBe(false);
  });
});
