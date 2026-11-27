import { Button, DensityProvider, ThemeToggle } from "@vitavision/ui";
import { Keyboard, Settings } from "lucide-react";
import { THEME_STORAGE_KEY } from "../lib/theme";
import { useUi } from "../session/ui";
import { Breadcrumb } from "./Breadcrumb";
import { JobIndicator } from "./JobIndicator";
import { RunSelector } from "./RunSelector";
import type { Place } from "./screens";

/** The bar under which every screen sits: where the user is (the project,
 * with its device), which run the app is about and what that run is (state,
 * verdict, stale), then the jobs, the shortcut list, the theme and Settings.
 * Its name, for assistive technology, is the app's. */
export function ContextBar({ place }: { place: Place }) {
  const open = useUi((s) => s.open);
  return (
    <DensityProvider value="compact">
      <div className="flex items-center gap-4 px-3 py-2">
        <span className="sr-only">Calibration Studio</span>
        <Breadcrumb place={place} />
        <RunSelector />
        <div className="ml-auto flex shrink-0 items-center gap-1">
          <JobIndicator />
          <Button
            size="sm"
            variant="ghost"
            aria-label="Keyboard shortcuts"
            title="Keyboard shortcuts (?)"
            onClick={() => open("shortcuts")}
          >
            <Keyboard aria-hidden className="size-4" />
          </Button>
          <ThemeToggle storageKey={THEME_STORAGE_KEY} />
          <Button
            size="sm"
            variant="ghost"
            aria-label="Settings"
            title="Settings"
            onClick={() => open("settings")}
          >
            <Settings aria-hidden className="size-4" />
          </Button>
        </div>
      </div>
    </DensityProvider>
  );
}
