import { ChevronRight } from "lucide-react";
import { Link } from "react-router";
import { isDirty, useDraft } from "../project/draftStore";
import { useProject } from "../project/projectStore";
import { ProjectMenu } from "./ProjectMenu";
import { placePath, screenPlace, type Place } from "./screens";

/** `Product ▸ Device ▸ Project`, from the project's device and name. The
 * project is a menu (open, new, recent); a project with no device says
 * `(no device)` quietly; an unsaved draft is a dot that goes to where it is
 * saved. */
export function Breadcrumb({ place }: { place: Place }) {
  const project = useProject((s) => s.project);
  const dirty = useDraft(isDirty);
  const device = project?.spec.device ?? null;

  return (
    <nav aria-label="Project" className="flex min-w-0 items-center gap-1.5 text-xs">
      {project && (
        <>
          {device?.product && (
            <>
              <span className="truncate text-fg-muted">{device.product}</span>
              <Separator />
            </>
          )}
          <span
            className={device ? "truncate text-fg-muted" : "truncate text-fg-subtle"}
            title={device ? device.id : "This project names no device"}
          >
            {device ? (device.label ?? device.id) : "(no device)"}
          </span>
          <Separator />
        </>
      )}
      <ProjectMenu />
      {project && dirty && (
        <Link
          to={placePath(screenPlace("calibrate", place))}
          aria-label="Unsaved changes: go to where they are saved"
          title="Unsaved changes"
          className="size-2 shrink-0 rounded-full bg-signal outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal"
        />
      )}
    </nav>
  );
}

function Separator() {
  return <ChevronRight aria-hidden className="size-3 shrink-0 text-fg-subtle" />;
}
