import { Dialog, ShortcutList } from "@vitavision/ui";
import { useUi } from "../session/ui";
import { GLOBAL_SHORTCUTS } from "./shortcuts";

/** `?`: the keys that work everywhere. */
export function ShortcutsDialog() {
  const open = useUi((s) => s.dialog === "shortcuts");
  const close = useUi((s) => s.close);
  return (
    <Dialog
      open={open}
      onOpenChange={(next) => !next && close()}
      title="Keyboard shortcuts"
      description="⌘ is Ctrl outside macOS. Keys that type (a bracket, a comma) do nothing while a text field has focus."
      className="w-[min(40rem,92vw)]"
    >
      <ShortcutList groups={GLOBAL_SHORTCUTS} />
    </Dialog>
  );
}
