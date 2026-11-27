import { Button, Empty } from "@vitavision/ui";
import { Link } from "react-router";
import {
  placePath,
  screenById,
  screenPlace,
  type Place,
  type ScreenId,
} from "../shell/screens";

/** What an address of a screen that does not exist yet shows: its name and
 * where the work can be done meanwhile. Never a broken page. */
export function NotYet({ id, place }: { id: ScreenId; place: Place }) {
  const screen = screenById(id);
  return (
    <div className="grid h-full place-items-center">
      <Empty
        action={
          <Button asChild variant="secondary" size="sm">
            <Link to={placePath(screenPlace("library", place))}>Go to the Library</Link>
          </Button>
        }
      >
        <span className="block text-sm font-semibold text-fg">{screen.label}</span>
        Coming in a later release.
      </Empty>
    </div>
  );
}
