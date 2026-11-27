import { Button, ConfirmDialog, Dialog, toast } from "@vitavision/ui";
import { Stepper, type StepperStep } from "@vitavision/workbench";
import { useState } from "react";
import { useNavigate } from "react-router";
import { deviceIdentity } from "../../../project/deviceDraft";
import { useProjectDialog } from "../../../project/dialogs";
import type { ProjectError } from "../../../project/projectModel";
import { useProject } from "../../../project/projectStore";
import { useReadOnly } from "../../../session/jobs";
import { userRoots, useSession } from "../../../session/session";
import { placePath, screenById } from "../../../shell/screens";
import { CreateStep } from "./CreateStep";
import { DataStep } from "./DataStep";
import { DeviceStep } from "./DeviceStep";
import { LayoutStep } from "./LayoutStep";
import { blockedAt, createSpec, NEW_STEPS, resolveProjectDir } from "./newProjectModel";
import { TargetStep } from "./TargetStep";
import { useNewProject } from "./useNewProject";

/** The guided New project dialog: five steps, nothing written until the
 * last. Back keeps what was entered; Cancel after the layout step asks
 * first. The project is created through the project store, and the dialog
 * ends on the Dataset screen (or Calibrate while that screen is not
 * built). */
export function NewProjectDialog({ manifest }: { manifest?: string }) {
  const close = useProjectDialog((s) => s.close);
  const navigate = useNavigate();
  const readOnly = useReadOnly();
  const status = useSession((s) => s.status);
  const roots = status.kind === "ready" ? userRoots(status.info) : [];
  const project = useNewProject();
  const { state, patch } = project;
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<ProjectError | null>(null);
  const [creating, setCreating] = useState(false);

  const index = NEW_STEPS.findIndex((s) => s.id === state.step);
  const steps: StepperStep[] = NEW_STEPS.map((s, i) => ({
    id: s.id,
    label: s.label,
    complete: i < index,
    ...(i > state.reached ? { blockedBy: "Finish the earlier steps first." } : {}),
  }));
  const blocked = blockedAt(state.step, state);
  const last = state.step === "create";

  const go = (to: number) => {
    const next = NEW_STEPS[to];
    if (next) patch({ step: next.id, reached: Math.max(state.reached, to) });
  };

  const create = async () => {
    if (!state.source || !state.manifest) return;
    const dir = resolveProjectDir({ ...state, source: state.source }, roots[0] ?? null);
    setCreating(true);
    setError(null);
    const spec = createSpec({
      source: state.source,
      manifest: state.manifest,
      name: state.name ?? "",
      device: state.device,
      projectDir: dir,
    });
    const ok = await useProject.getState().create(dir, spec);
    setCreating(false);
    if (!ok) {
      // Said here, in the step, not by the shell's strip over the screens.
      setError(useProject.getState().error);
      useProject.getState().clearError();
      return;
    }
    const screen = screenById("dataset").built ? "dataset" : "calibrate";
    toast({ title: "Project created", description: dir, tone: "success" });
    close();
    void navigate(placePath({ screen, tab: null, project: dir, run: null }));
  };

  const leave = () => (index >= 1 ? setConfirming(true) : close());

  return (
    <>
      <Dialog
        open
        onOpenChange={(open) => {
          if (!open) leave();
        }}
        title="New project"
        description="Describe the data, choose the target, name the project. Nothing is written until the last step."
        className="w-[min(60rem,calc(100vw-2rem))]"
        footer={
          <>
            <Button variant="ghost" onClick={leave}>
              Cancel
            </Button>
            <Button
              variant="secondary"
              disabled={index === 0}
              onClick={() => go(index - 1)}
            >
              Back
            </Button>
            {last ? (
              <Button
                variant="primary"
                disabled={readOnly || creating || state.manifest === null}
                onClick={() => void create()}
              >
                {creating ? "Creating…" : "Create project"}
              </Button>
            ) : (
              <Button
                variant="primary"
                disabled={blocked !== null}
                title={blocked ?? undefined}
                onClick={() => go(index + 1)}
              >
                {state.step === "device" && deviceIdentity(state.device) === null
                  ? "Set later"
                  : "Next"}
              </Button>
            )}
          </>
        }
      >
        <div className="mt-3 flex flex-col gap-4">
          <Stepper
            aria-label="New project steps"
            steps={steps}
            value={state.step}
            onValueChange={(id) => go(NEW_STEPS.findIndex((s) => s.id === id))}
          />
          <div className="min-h-[22rem]">
            {state.step === "data" && (
              <DataStep project={project} initialManifest={manifest} />
            )}
            {state.step === "layout" && <LayoutStep project={project} />}
            {state.step === "target" && <TargetStep project={project} />}
            {state.step === "device" && <DeviceStep project={project} />}
            {state.step === "create" && <CreateStep project={project} error={error} />}
          </div>
          {blocked !== null && !last && state.manifest !== null && (
            <p className="text-xs text-fg-muted">{blocked}</p>
          )}
        </div>
      </Dialog>
      <ConfirmDialog
        open={confirming}
        onOpenChange={setConfirming}
        title="Discard this new project?"
        description="What you entered is lost. Nothing was written."
        confirmLabel="Discard"
        destructive
        onConfirm={() => {
          setConfirming(false);
          close();
        }}
      />
    </>
  );
}
