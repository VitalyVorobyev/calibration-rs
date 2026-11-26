import type { ReactNode } from "react";
import { Navigate } from "react-router";
import { capabilities } from "../api";

/** A route the desktop app offers and the browser viewer does not: its
 * workspace there, Diagnose elsewhere. */
export function DesktopOnly({ children }: { children: ReactNode }) {
  return capabilities().desktop ? children : <Navigate to="/diagnose" replace />;
}
