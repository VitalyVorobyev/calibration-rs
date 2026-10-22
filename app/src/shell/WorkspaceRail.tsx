import { NavRail, NavRailItem } from "@vitavision/workbench";
import { Box, Columns2, Crosshair, Layers, Play } from "lucide-react";
import { useEffect, type ReactNode } from "react";
import { Link, useLocation, useNavigate } from "react-router";
import { WORKSPACES, workspaceForShortcut } from "./workspaces";

const ICONS: Record<string, ReactNode> = {
  "/diagnose": <Crosshair />,
  "/viewer3d": <Box />,
  "/epipolar": <Columns2 />,
  "/depth": <Layers />,
  "/run": <Play />,
};

/** The workspace rail: one router link per workspace, the current route's
 * marked `aria-current="page"`, labels as tooltips. `⌘1`…`⌘5` (`Ctrl` off
 * macOS) switch workspaces from anywhere. */
export function WorkspaceRail() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const current = WORKSPACES.find((w) => pathname.startsWith(w.to))?.to ?? null;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const to = workspaceForShortcut(e);
      if (to === null) return;
      e.preventDefault();
      void navigate(to);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate]);

  return (
    <NavRail value={current} labels="tooltip">
      {WORKSPACES.map((w) => (
        <NavRailItem key={w.to} value={w.to} label={w.label} icon={ICONS[w.to]} asChild>
          <Link to={w.to} />
        </NavRailItem>
      ))}
    </NavRail>
  );
}
