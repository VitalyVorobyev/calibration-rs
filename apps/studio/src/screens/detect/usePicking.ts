/** Which features of the image on the stage are picked and which one the
 * pointer is over. Both belong to one image (`resetKey`): another image, or
 * another detection of it, starts with nothing picked, without an effect
 * to reset them. */
import type { TargetHit } from "@vitavision/overlays";
import { useCallback, useState } from "react";
import { useSelection } from "../../session/selection";

const NONE: ReadonlySet<number> = new Set();

interface State {
  key: string;
  picked: ReadonlySet<number>;
  hovered: number | null;
}

export function usePicking(resetKey: string) {
  const [state, setState] = useState<State>({
    key: resetKey,
    picked: NONE,
    hovered: null,
  });
  const current =
    state.key === resetKey ? state : { key: resetKey, picked: NONE, hovered: null };

  const update = useCallback(
    (change: (now: State) => Partial<State>) =>
      setState((prev) => {
        const now =
          prev.key === resetKey ? prev : { key: resetKey, picked: NONE, hovered: null };
        return { ...now, ...change(now) };
      }),
    [resetKey],
  );

  /** Press on feature `index`; `additive` toggles it in the picked set. */
  const pick = useCallback(
    (index: number, additive: boolean) => {
      update((now) => {
        if (additive) {
          const next = new Set(now.picked);
          if (!next.delete(index)) next.add(index);
          return { picked: next };
        }
        return {
          picked:
            now.picked.size === 1 && now.picked.has(index) ? NONE : new Set([index]),
        };
      });
      useSelection.getState().selectFeature(index);
    },
    [update],
  );
  /** A box over the stage picked `indices`. */
  const box = useCallback(
    (indices: number[], additive: boolean) =>
      update((now) => ({
        picked: new Set(additive ? [...now.picked, ...indices] : indices),
      })),
    [update],
  );
  const clear = useCallback(() => {
    update(() => ({ picked: NONE }));
    useSelection.getState().selectFeature(null);
  }, [update]);
  const hoverIndex = useCallback(
    (index: number | null) => update(() => ({ hovered: index })),
    [update],
  );
  /** The overlay's hover: a feature is a number id. */
  const hoverHit = useCallback(
    (hit: TargetHit | null) =>
      hoverIndex(hit && typeof hit.id === "number" ? hit.id : null),
    [hoverIndex],
  );

  return {
    picked: current.picked,
    hovered: current.hovered,
    pick,
    box,
    clear,
    hoverIndex,
    hoverHit,
  };
}
