import { NavRail, NavRailItem } from "@vitavision/workbench";
import { Box, Columns2, Crosshair, Layers, Play } from "lucide-react";
import { useEffect, useMemo, type ReactNode } from "react";
import { Link, useLocation, useNavigate } from "react-router";
import { capabilities } from "../api";
import { availableWorkspaces, workspaceForShortcut } from "./workspaces";

const ICONS: Record<string, ReactNode> = {
  "/diagnose": <Crosshair />,
  "/viewer3d": <Box />,
  "/epipolar": <Columns2 />,
  "/depth": <Layers />,
  "/run": <Play />,
};

/** The workspace rail: one router link per workspace the back end offers,
 * the current route's marked `aria-current="page"`, labels as tooltips.
 * `⌘1`…`⌘5` (`Ctrl` off macOS) switch workspaces from anywhere. */
export function WorkspaceRail() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const { desktop } = capabilities();
  const workspaces = useMemo(() => availableWorkspaces(desktop), [desktop]);
  const current = workspaces.find((w) => pathname.startsWith(w.to))?.to ?? null;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const to = workspaceForShortcut(e, workspaces);
      if (to === null) return;
      e.preventDefault();
      void navigate(to);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, workspaces]);

  return (
    <NavRail value={current} labels="tooltip">
      {workspaces.map((w) => (
        <NavRailItem key={w.to} value={w.to} label={w.label} icon={ICONS[w.to]} asChild>
          <Link to={w.to} />
        </NavRailItem>
      ))}
    </NavRail>
  );
}
