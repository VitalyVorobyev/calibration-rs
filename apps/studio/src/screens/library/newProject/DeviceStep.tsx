import { useMemo } from "react";
import { useLibraryScan } from "../../../project/useLibraryScan";
import { DeviceFields } from "../../../components/DeviceFields";
import type { NewProjectApi } from "./useNewProject";

/** Step 4, optional: the device this project calibrates. Projects of one
 * device (any folder) form its history, grouped under its type in the
 * Library. Skippable; it can be set from the context bar later. */
export function DeviceStep({ project }: { project: NewProjectApi }) {
  const scan = useLibraryScan();
  const products = useMemo(
    () =>
      [
        ...new Set(
          (scan.data?.projects ?? [])
            .map((p) => p.device?.product)
            .filter((p): p is string => typeof p === "string" && p !== ""),
        ),
      ].sort(),
    [scan.data],
  );
  return (
    <div className="flex max-w-lg flex-col gap-3">
      <p className="text-sm text-fg-muted">
        Which device is this calibration for? Leave it empty to set it later; a project
        with no device is listed under “(no device)”.
      </p>
      <DeviceFields
        value={project.state.device}
        onChange={(device) => project.patch({ device })}
        products={products}
        autoFocus
      />
    </div>
  );
}
