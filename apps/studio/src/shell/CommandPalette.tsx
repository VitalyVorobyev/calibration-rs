import { Dialog, Input } from "@vitavision/ui";
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useLocation, useNavigate } from "react-router";
import { filterItems, type PaletteItem } from "../lib/palette";
import { openProject } from "../project/projectActions";
import { folderName, runLabel, runsNewestFirst } from "../project/projectModel";
import { useProject } from "../project/projectStore";
import { useSelection } from "../session/selection";
import { useUi } from "../session/ui";
import { useStore } from "../store";
import {
  INSPECT_TABS,
  isEnabled,
  parsePlace,
  placePath,
  screenPlace,
  SCREENS,
} from "./screens";

interface Command extends PaletteItem {
  run: () => void;
}

/** `⌘K`: jump to a screen (Inspect's tabs too), a recent project, a run of
 * the open project or a camera of the shown run, by typing a few letters.
 * A combobox over a listbox: arrows move, Enter runs, Esc closes. */
export function CommandPalette() {
  const open = useUi((s) => s.dialog === "palette");
  const close = useUi((s) => s.close);
  return (
    <Dialog
      open={open}
      onOpenChange={(next) => !next && close()}
      title="Command palette"
      className="w-[min(34rem,92vw)]"
    >
      {open && <PaletteBody onDone={close} />}
    </Dialog>
  );
}

function useCommands(): Command[] {
  const { pathname, search } = useLocation();
  const navigate = useNavigate();
  const project = useProject((s) => s.project);
  const recent = useProject((s) => s.recent);
  const cameras = useStore((s) => s.cameraValues);
  const hasRun = useStore((s) => s.data !== null);
  const openRun = useProject((s) => s.openRun);
  const selectCamera = useSelection((s) => s.selectCamera);
  const place = useMemo(() => parsePlace(pathname, search), [pathname, search]);

  return useMemo(() => {
    const go = (to: string) => () => void navigate(to);
    const commands: Command[] = [];
    for (const screen of SCREENS) {
      if (!isEnabled(screen, project !== null)) continue;
      commands.push({
        id: `screen:${screen.id}`,
        group: "Screens",
        label: screen.label,
        run: go(placePath(screenPlace(screen.id, place))),
      });
      if (screen.id === "inspect") {
        for (const tab of INSPECT_TABS) {
          commands.push({
            id: `screen:inspect:${tab}`,
            group: "Screens",
            label: `Inspect · ${tab[0]!.toUpperCase()}${tab.slice(1)}`,
            run: go(placePath({ ...screenPlace("inspect", place), tab })),
          });
        }
      }
    }
    for (const run of runsNewestFirst(project?.runs ?? [])) {
      commands.push({
        id: `run:${run.id}`,
        group: "Runs",
        label: runLabel(run),
        keywords: run.id,
        run: () => void openRun(run.id),
      });
    }
    if (hasRun) {
      for (const camera of cameras) {
        commands.push({
          id: `camera:${camera}`,
          group: "Cameras",
          label: `Camera ${camera}`,
          run: () => selectCamera(camera),
        });
      }
    }
    for (const dir of recent) {
      commands.push({
        id: `project:${dir}`,
        group: "Projects",
        label: folderName(dir),
        keywords: dir,
        run: () => void openProject(dir),
      });
    }
    return commands;
  }, [navigate, place, project, recent, cameras, hasRun, openRun, selectCamera]);
}

function PaletteBody({ onDone }: { onDone: () => void }) {
  const commands = useCommands();
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLUListElement>(null);
  const shown = useMemo(() => filterItems(commands, query), [commands, query]);
  const at = Math.min(active, Math.max(shown.length - 1, 0));

  useEffect(() => {
    listRef.current
      ?.querySelector('[aria-selected="true"]')
      ?.scrollIntoView({ block: "nearest" });
  }, [at, shown]);

  const choose = (command: Command | undefined) => {
    if (!command) return;
    onDone();
    command.run();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive(shown.length === 0 ? 0 : (at + 1) % shown.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive(shown.length === 0 ? 0 : (at - 1 + shown.length) % shown.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      choose(shown[at]);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <Input
        autoFocus
        role="combobox"
        aria-label="Search commands"
        aria-expanded
        aria-controls="palette-list"
        aria-activedescendant={shown[at] ? `palette-option-${at}` : undefined}
        aria-autocomplete="list"
        placeholder="Go to a screen, project, run or camera…"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
        }}
        onKeyDown={onKeyDown}
      />
      <ul
        id="palette-list"
        ref={listRef}
        role="listbox"
        aria-label="Commands"
        className="max-h-72 overflow-y-auto"
      >
        {shown.length === 0 && (
          <li role="presentation" className="px-3 py-6 text-center text-xs text-fg-muted">
            Nothing matches “{query}”.
          </li>
        )}
        {shown.map((command, i) => (
          <li
            key={command.id}
            id={`palette-option-${i}`}
            role="option"
            aria-selected={i === at}
            onMouseMove={() => setActive(i)}
            onClick={() => choose(command)}
            className="flex cursor-pointer items-center justify-between gap-3 rounded-control px-3 py-1.5 text-sm text-fg aria-selected:bg-raised aria-selected:ring-1 aria-selected:ring-line-strong"
          >
            <span className="truncate">{command.label}</span>
            <span className="shrink-0 text-[10px] tracking-wider text-fg-subtle uppercase">
              {command.group}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
