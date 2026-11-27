/** The placeholder screens of the shell's tests: a body that says which
 * screen it is, and the address the router is at. */
import { useLocation } from "react-router";
import { LibraryScreen } from "../screens/library";

/** The current address, for assertions. */
function Where() {
  const { pathname, search } = useLocation();
  return <output data-testid="where">{`${pathname}${search}`}</output>;
}

export function Body({ text }: { text: string }) {
  return (
    <>
      <p>{text}</p>
      <Where />
    </>
  );
}

/** The real Library, with the marker the tests wait for. */
export function LibraryBody() {
  return (
    <>
      <p className="sr-only">library body</p>
      <LibraryScreen />
      <Where />
    </>
  );
}
