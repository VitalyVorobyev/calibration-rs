/** A device identity as a form fills it in: three text fields, the id the
 * one the library requires. Shared by the New project dialog and "Set
 * device". */
import type { DeviceIdentity } from "../api/generated/types";

/** A device the dialog is filling in. */
export interface DeviceDraft {
  product: string;
  id: string;
  label: string;
}

export const NO_DEVICE_DRAFT: DeviceDraft = { product: "", id: "", label: "" };

/** The identity to write, or `null` when no id was given (the device step
 * is skippable; the library requires an id). */
export function deviceIdentity(d: DeviceDraft): DeviceIdentity | null {
  const id = d.id.trim();
  if (id === "") return null;
  const identity: DeviceIdentity = { id };
  if (d.product.trim() !== "") identity.product = d.product.trim();
  if (d.label.trim() !== "") identity.label = d.label.trim();
  return identity;
}

/** Why the device draft cannot be written; `null` when it can (or is empty). */
export function deviceProblem(d: DeviceDraft): string | null {
  const touched = d.product.trim() !== "" || d.label.trim() !== "";
  if (d.id.trim() === "" && touched) {
    return "A device needs an id (a serial number or MAC address), or leave all three empty.";
  }
  return null;
}
