import { Button } from "@vitavision/ui";
import { ChevronRight } from "lucide-react";
import { useMemo } from "react";
import { Link, useNavigate } from "react-router";
import { isDirty, useDraft } from "../project/draftStore";
import { useProjectDialog } from "../project/dialogs";
import { useProject } from "../project/projectStore";
import { useLibraryScan } from "../project/useLibraryScan";
import {
  groupLibrary,
  projectRow,
  UNSPECIFIED_TYPE,
} from "../screens/library/libraryModel";
import { useReadOnly } from "../session/jobs";
import { deviceParamsOf } from "../screens/device/deviceModel";
import { ProjectMenu } from "./ProjectMenu";
import { placePath, screenById, screenPlace, type Place } from "./screens";
import { SiblingPopover } from "./SiblingPopover";

/** `Product ▸ Device ▸ Project`, from the project's device and name. The
 * device type and the device are each a list of their siblings (choosing
 * one shows it on the Device screen); the project is the menu that opens,
 * creates and reopens projects, with its device's other projects at hand.
 * A project with no device says `(no device)` quietly, and clicking it sets
 * one. An unsaved draft is a dot that goes to where it is saved. */
export function Breadcrumb({ place }: { place: Place }) {
  const project = useProject((s) => s.project);
  const dirty = useDraft(isDirty);
  const manifestDirty = useDraft((s) => s.manifestDirty);
  const device = project?.spec.device ?? null;
  const readOnly = useReadOnly();
  const navigate = useNavigate();
  const show = useProjectDialog((s) => s.show);
  const scan = useLibraryScan();

  const groups = useMemo(
    () => groupLibrary((scan.data?.projects ?? []).map(projectRow)),
    [scan.data],
  );
  const product = device ? device.product?.trim() || UNSPECIFIED_TYPE : null;
  const productGroup = groups.products.find((p) => p.label === product);

  /** The Device screen, on the library node `id` (a device type or a
   * device). A type that names no product has no units to compare: its
   * first device is shown instead. */
  const showOnDevice = (id: string) => {
    const node =
      groups.products.find((p) => p.id === id && p.label === UNSPECIFIED_TYPE)?.devices[0]
        ?.id ?? id;
    const params = deviceParamsOf(groups, node);
    if (params === null) return;
    const base = placePath(screenPlace("device", place));
    void navigate(`${base}${base.includes("?") ? "&" : "?"}${params.toString()}`);
  };

  return (
    <nav aria-label="Project" className="flex min-w-0 items-center gap-0.5 text-xs">
      {project && (
        <>
          {device ? (
            <>
              {product && (
                <>
                  <SiblingPopover
                    label={product}
                    title="Device type: choose another to open its units on the Device screen"
                    muted
                    current={productGroup?.id ?? null}
                    options={groups.products.map((p) => ({
                      id: p.id,
                      label: p.label,
                      note: `${p.devices.length} ${p.devices.length === 1 ? "device" : "devices"}`,
                    }))}
                    onChoose={showOnDevice}
                  />
                  <Separator />
                </>
              )}
              <SiblingPopover
                label={device.label ?? device.id}
                title="Device: choose another to open its history on the Device screen"
                muted
                current={
                  productGroup?.devices.find((d) => d.device.id === device.id)?.id ?? null
                }
                options={(productGroup?.devices ?? []).map((d) => ({
                  id: d.id,
                  label: d.device.label ?? d.device.id,
                  note: d.device.label ? d.device.id : undefined,
                }))}
                onChoose={showOnDevice}
              />
            </>
          ) : (
            <Button
              size="sm"
              variant="ghost"
              disabled={readOnly}
              title="This project names no device. Set one."
              onClick={() => show({ kind: "device", dir: project.dir })}
            >
              <span className="text-fg-subtle">(no device)</span>
            </Button>
          )}
          <Separator />
        </>
      )}
      <ProjectMenu />
      {project && dirty && (
        <Link
          to={placePath(
            screenPlace(
              manifestDirty && screenById("dataset").built ? "dataset" : "calibrate",
              place,
            ),
          )}
          aria-label="Unsaved changes: go to where they are saved"
          title="Unsaved changes"
          className="ml-1.5 size-2 shrink-0 rounded-full bg-signal outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal"
        />
      )}
    </nav>
  );
}

function Separator() {
  return <ChevronRight aria-hidden className="size-3 shrink-0 text-fg-subtle" />;
}
