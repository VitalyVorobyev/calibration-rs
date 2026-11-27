/** The Device screen's queries: a device's (or a type's) history and
 * series, and the seed a choice makes. Shared through the query cache, and
 * marked stale with the library scan (`session/sync.ts`). */
import { api } from "../../api";
import type {
  DeviceHistories,
  DeviceSeriesSet,
  SeedFrom,
  SeedPreview,
} from "../../api/generated/types";
import { useQuery, type Query } from "../../session/query";
import { useSession } from "../../session/session";
import { selectorOf, type DeviceTarget } from "./deviceModel";

export function useDeviceHistory(target: DeviceTarget | null): Query<DeviceHistories> {
  const ready = useSession((s) => s.status.kind === "ready");
  const device = target ? selectorOf(target) : null;
  return useQuery(
    "device_history",
    { device },
    () => api.deviceHistory({ device: device! }),
    ready && device !== null,
  );
}

export function useDeviceSeries(
  target: DeviceTarget | null,
  starredOnly: boolean,
): Query<DeviceSeriesSet> {
  const ready = useSession((s) => s.status.kind === "ready");
  const device = target ? selectorOf(target) : null;
  return useQuery(
    "device_series",
    { device, starred_only: starredOnly },
    () => api.deviceSeries({ device: device!, starred_only: starredOnly }),
    ready && device !== null,
  );
}

export function useSeedPreview(from: SeedFrom | null): Query<SeedPreview> {
  const ready = useSession((s) => s.status.kind === "ready");
  return useQuery(
    "seed_preview",
    { from },
    () => api.seedPreview({ from: from! }),
    ready && from !== null,
  );
}
