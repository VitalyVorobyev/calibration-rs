import { ImportDialog } from "../screens/library/ImportDialog";
import { NewProjectDialog } from "../screens/library/newProject/NewProjectDialog";
import { useProjectDialog } from "../project/dialogs";
import { SetDeviceDialog } from "./SetDeviceDialog";

/** The project dialogs, opened from the Library, the project menu and the
 * context bar through `useProjectDialog`. Each is mounted only while open,
 * so it starts from nothing every time. */
export function ProjectDialogs() {
  const open = useProjectDialog((s) => s.open);
  if (open === null) return null;
  switch (open.kind) {
    case "new":
      return (
        <NewProjectDialog
          key="new"
          {...(open.manifest ? { manifest: open.manifest } : {})}
        />
      );
    case "import":
      return <ImportDialog key="import" {...(open.dir ? { dir: open.dir } : {})} />;
    case "device":
      return <SetDeviceDialog key={`device:${open.dir}`} />;
  }
}
