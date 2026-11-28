import { SegmentedControl, Select } from "@vitavision/ui";
import { UNSPECIFIED_TYPE, type LibraryGroups } from "../library/libraryModel";
import { productOf, productValue, type DeviceTarget } from "./deviceModel";

/** Device type, then device, from the library's grouping; and whether to
 * look at that one unit or at every unit of its type. */
export function DevicePicker({
  groups,
  target,
  lastDevice,
  onChange,
}: {
  groups: LibraryGroups;
  target: DeviceTarget | null;
  /** The device shown last, for "This device" after "Whole type". */
  lastDevice: string | null;
  onChange: (target: DeviceTarget) => void;
}) {
  const product = target ? productValue(target.product) : "";
  const group = groups.products.find((p) => p.label === product);
  const devices = group?.devices ?? [];
  const typeOk = target !== null && target.product !== null;

  return (
    <div className="flex flex-wrap items-end gap-3" role="group" aria-label="Device">
      <label className="flex flex-col gap-1 text-xs text-fg-muted">
        Device type
        <Select
          aria-label="Device type"
          className="min-w-44"
          value={product}
          placeholder="Choose a type"
          options={groups.products.map((p) => ({
            value: p.label,
            label: p.label,
            note: `${p.devices.length} ${p.devices.length === 1 ? "unit" : "units"}`,
          }))}
          onValueChange={(value) => {
            const next = groups.products.find((p) => p.label === value);
            const first = next?.devices[0];
            if (!next || !first) return;
            onChange({
              scope: "device",
              product: productOf(next.label),
              id: first.device.id,
            });
          }}
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-fg-muted">
        Device
        <Select
          aria-label="Device"
          className="min-w-44"
          value={target?.scope === "device" ? (target.id ?? "") : ""}
          placeholder={
            target?.scope === "type" ? "Every unit of the type" : "Choose a device"
          }
          disabled={devices.length === 0}
          options={devices.map((d) => ({
            value: d.device.id,
            label: d.device.label ?? d.device.id,
            ...(d.device.label ? { note: d.device.id } : {}),
          }))}
          onValueChange={(id) =>
            onChange({ scope: "device", product: target?.product ?? null, id })
          }
        />
      </label>
      <div className="flex flex-col gap-1 text-xs text-fg-muted">
        Show
        <SegmentedControl
          aria-label="Show one device or the whole type"
          value={target?.scope ?? "device"}
          options={[
            { value: "device", label: "This device" },
            { value: "type", label: "Whole type" },
          ]}
          disabled={target === null || (!typeOk && target.scope === "device")}
          onValueChange={(value) => {
            if (!target) return;
            if (value === "type" && target.product) {
              onChange({ scope: "type", product: target.product, id: null });
            } else if (value === "device") {
              const id =
                devices.find((d) => d.device.id === lastDevice)?.device.id ??
                devices[0]?.device.id ??
                null;
              if (id) onChange({ scope: "device", product: target.product, id });
            }
          }}
        />
      </div>
      {target?.product === null && target.scope === "device" && (
        <p className="max-w-xs pb-1.5 text-xs text-fg-subtle">
          This device names no type ({UNSPECIFIED_TYPE}), so there is no type to compare
          it with.
        </p>
      )}
    </div>
  );
}
