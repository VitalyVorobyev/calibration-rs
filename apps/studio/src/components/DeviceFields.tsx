import { Field, Input } from "@vitavision/ui";
import { useId } from "react";
import { deviceProblem, type DeviceDraft } from "../project/deviceDraft";

/** The three fields of a device identity: its type, its id and a label.
 * `products` are the types already in the library, offered as completions
 * of the free text. The id is the key the library groups projects by; the
 * type and label are for the reader. */
export function DeviceFields({
  value,
  onChange,
  products,
  autoFocus,
}: {
  value: DeviceDraft;
  onChange: (next: DeviceDraft) => void;
  products: readonly string[];
  autoFocus?: boolean;
}) {
  const list = useId();
  const problem = deviceProblem(value);
  return (
    <div className="flex flex-col gap-3">
      <Field
        label="Device type"
        description="The product line or model; devices of one type are grouped in the Library."
      >
        <Input
          list={list}
          value={value.product}
          autoFocus={autoFocus}
          placeholder="Stereo-X1"
          onChange={(e) => onChange({ ...value, product: e.target.value })}
        />
        <datalist id={list}>
          {products.map((p) => (
            <option key={p} value={p} />
          ))}
        </datalist>
      </Field>
      <Field
        label="Device id"
        description="A serial number or MAC address: what tells this device from others of its type."
        error={problem}
      >
        <Input
          value={value.id}
          className="font-mono"
          placeholder="SN-00123"
          onChange={(e) => onChange({ ...value, id: e.target.value })}
        />
      </Field>
      <Field label="Label" description="A name for display; optional.">
        <Input
          value={value.label}
          placeholder="Bench cell 3"
          onChange={(e) => onChange({ ...value, label: e.target.value })}
        />
      </Field>
    </div>
  );
}
