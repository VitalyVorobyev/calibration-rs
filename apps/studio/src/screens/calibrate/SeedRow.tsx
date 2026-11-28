import { Button, Callout, Select } from "@vitavision/ui";
import { useState } from "react";
import { Link } from "react-router";
import { api, asStudioError } from "../../api";
import type { ProjectView } from "../../api/generated/types";
import { folderName } from "../../project/projectModel";
import { applyProjectSeed } from "../../project/seed";
import { useReadOnly } from "../../session/jobs";
import { useQuery } from "../../session/query";
import { placePath } from "../../shell/screens";
import { targetOfDevice, toParams } from "../device/deviceModel";
import { describeSource, lensLine, sourceKind, sourceRunRef } from "../device/seedModel";

/** The device screen's Seed tab for the project's device, with the project
 * kept in the address. */
function deviceSeedPath(project: ProjectView): string {
  const base = placePath({
    screen: "device",
    tab: null,
    project: project.dir,
    run: null,
  });
  const params = toParams(new URLSearchParams(base.split("?")[1] ?? ""), {
    target: targetOfDevice(project.spec.device),
    tab: "seed",
    starredOnly: false,
    xMode: "order",
  });
  return `${base.split("?")[0]}?${params.toString()}`;
}

/** The project's seed in the "This run" panel: what the next run starts
 * from (the project's `[seed]`, or nothing), a choice among the project's
 * own calibrated runs, clearing it, and the way on to the Device screen for
 * a device or a type's seed. A seed is a starting value, never a prior. */
export function SeedRow({ project }: { project: ProjectView }) {
  const readOnly = useReadOnly();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const seed = project.spec.seed ?? null;
  const ref = sourceRunRef(seed);
  const preview = useQuery(
    "seed_preview",
    { from: ref ? { kind: "run", run: ref } : null },
    () => api.seedPreview({ from: { kind: "run", run: ref! } }),
    ref !== null,
  );
  // Any run whose files read can seed, as on the Device screen.
  const runs = project.runs.filter((r) => r.state.kind !== "unreadable");
  const chosen = runs.find((r) => seed !== null && "run" in seed && seed.run === r.dir);

  const set = async (next: Parameters<typeof applyProjectSeed>[1]) => {
    setBusy(true);
    setError(null);
    try {
      await applyProjectSeed(project, next);
    } catch (e) {
      setError(asStudioError(e).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-fg">
        {seed === null ? (
          <>
            None{" "}
            <span className="text-fg-subtle">
              · each camera starts from its own images
            </span>
          </>
        ) : (
          <>
            From {sourceKind(seed)}{" "}
            <span
              className="block min-w-0 truncate font-mono text-[11px] text-fg-muted"
              title={describeSource(seed)}
            >
              {"run" in seed
                ? ref
                  ? `${ref.id} of ${folderName(ref.dir)}`
                  : seed.run
                : "device_spec" in seed
                  ? seed.device_spec
                  : seed.file}
            </span>
          </>
        )}
      </p>
      {ref !== null && preview.data !== undefined && (
        <ul aria-label="What the seed starts from" className="text-[11px] text-fg-muted">
          {preview.data.seed.cameras.map((camera) => (
            <li key={camera.id}>
              <span className="font-mono text-fg">{camera.id}</span> · {lensLine(camera)}
              {camera.cam_se3_rig ? " · mount" : ""}
              {camera.laser_plane ? " · laser plane" : ""}
            </li>
          ))}
          {preview.data.seed.handeye && (
            <li>
              <span className="font-mono text-fg">hand-eye</span> ·{" "}
              {preview.data.seed.handeye.mode.replace(/_/g, " ")}
            </li>
          )}
        </ul>
      )}
      {ref !== null && preview.error && preview.data === undefined && (
        <p className="text-[11px] text-defect">{preview.error.message}</p>
      )}
      {seed !== null && ref === null && (
        <p className="text-[11px] text-fg-muted">
          Its values are read when the run starts.
        </p>
      )}
      <div className="flex flex-wrap items-center gap-1.5">
        <Select
          aria-label="Seed from a run of this project"
          className="h-7 min-w-0 flex-1 text-xs"
          value={chosen?.id ?? ""}
          placeholder={runs.length === 0 ? "No run to seed from yet" : "Seed from a run…"}
          disabled={readOnly || busy || runs.length === 0}
          options={runs.map((r) => ({
            value: r.id,
            label: r.id,
            note: r.notes.label ?? "",
          }))}
          onValueChange={(id) => {
            const run = runs.find((r) => r.id === id);
            if (run) void set({ run: run.dir });
          }}
        />
        {seed !== null && (
          <Button
            size="sm"
            variant="ghost"
            disabled={readOnly || busy}
            onClick={() => void set(null)}
          >
            Clear
          </Button>
        )}
      </div>
      {error && (
        <Callout tone="error" title="The project cannot take this seed">
          <span className="text-xs">{error}</span>
        </Callout>
      )}
      <p className="text-[11px] text-fg-subtle">
        A starting value: the run still estimates everything from its own images; only a
        tilted (Scheimpflug) sensor is held near the seeded tilt.{" "}
        <Link
          className="text-signal underline-offset-2 hover:underline"
          to={deviceSeedPath(project)}
        >
          Seed from the device or its type
        </Link>
      </p>
    </div>
  );
}
