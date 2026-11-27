import { Navigate, useLocation } from "react-router";
import { parsePlace, type ScreenId } from "../shell/screens";
import { NotYet } from "./NotYet";

/** Redirect to `to`, keeping the query (the project and run). */
export function Redirect({ to }: { to: string }) {
  const { search } = useLocation();
  return <Navigate to={{ pathname: to, search }} replace />;
}

/** A screen that is not built yet, for its address. */
export function Unbuilt({ id }: { id: ScreenId }) {
  const { pathname, search } = useLocation();
  return <NotYet id={id} place={parsePlace(pathname, search)} />;
}
