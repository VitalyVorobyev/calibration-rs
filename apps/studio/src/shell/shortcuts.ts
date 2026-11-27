/** The global shortcuts as the help lists them: what `keymap.ts` binds,
 * grouped. A screen adds its own groups beside these. `⌘` is `Ctrl` off
 * macOS. */
import type { ShortcutGroup } from "@vitavision/ui";

export const GLOBAL_SHORTCUTS: readonly ShortcutGroup[] = [
  {
    title: "Go to",
    shortcuts: [
      { keys: ["⌘1…7"], description: "Library … Device, in the rail's order" },
      { keys: ["⌘K"], description: "Command palette: project, run, camera, screen" },
    ],
  },
  {
    title: "Selection",
    shortcuts: [
      { keys: ["[", "]"], description: "Previous / next view" },
      { keys: [",", "."], description: "Previous / next camera" },
      { keys: ["⇧[", "⇧]"], description: "Previous / next run" },
      { keys: ["Esc"], description: "Clear the selection, then close panels" },
    ],
  },
  {
    title: "Project",
    shortcuts: [
      { keys: ["⌘S"], description: "Save the project's unsaved edits" },
      { keys: ["?"], description: "This list" },
    ],
  },
];
