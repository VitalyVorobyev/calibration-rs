import { Button, Dialog, ErrorBox } from "@vitavision/ui";
import { useState } from "react";
import { useProjectDialog } from "../project/dialogs";
import { useProject } from "../project/projectStore";
import { useReadOnly } from "../session/jobs";
import { useLibraryScan } from "../project/useLibraryScan";
import { DeviceFields } from "../components/DeviceFields";
import { deviceIdentity, deviceProblem, type DeviceDraft } from "../project/deviceDraft";

/** "Set device": the device of the open project, written to its `[device]`
 * (a project edit, saved at once over the revision it was opened at).
 * Clearing all three fields removes the device. */
export function SetDeviceDialog() {
  const close = useProjectDialog((s) => s.close);
  const project = useProject((s) => s.project);
  const readOnly = useReadOnly();
  const scan = useLibraryScan();
  const [value, setValue] = useState<DeviceDraft>({
    product: project?.spec.device?.product ?? "",
    id: project?.spec.device?.id ?? "",
    label: project?.spec.device?.label ?? "",
  });
  const [error, setError] = useState<string | null>(null);
  const products = [
    ...new Set(
      (scan.data?.projects ?? [])
        .map((p) => p.device?.product)
        .filter((p): p is string => typeof p === "string" && p !== ""),
    ),
  ].sort();
  const problem = deviceProblem(value);

  const save = async () => {
    if (!project) return;
    const { device: _old, ...rest } = project.spec;
    const identity = deviceIdentity(value);
    const ok = await useProject
      .getState()
      .save(identity ? { ...rest, device: identity } : rest);
    if (ok) {
      close();
      return;
    }
    // Said here; the shell's strip would say it over the screens.
    setError(useProject.getState().error?.message ?? "The project could not be saved.");
    useProject.getState().clearError();
  };

  return (
    <Dialog
      open
      onOpenChange={(o) => {
        if (!o) close();
      }}
      title="Set device"
      description="The device this project calibrates. Projects of one device form its history in the Library."
      footer={
        <>
          <Button variant="ghost" onClick={close}>
            Cancel
          </Button>
          <Button
            variant="primary"
            disabled={readOnly || problem !== null || !project}
            onClick={() => void save()}
          >
            Save
          </Button>
        </>
      }
    >
      <div className="mt-4 flex flex-col gap-3">
        <DeviceFields value={value} onChange={setValue} products={products} autoFocus />
        {error && <ErrorBox>{error}</ErrorBox>}
      </div>
    </Dialog>
  );
}
