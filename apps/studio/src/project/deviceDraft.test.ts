import { describe, expect, it } from "vitest";
import { deviceIdentity, deviceProblem } from "./deviceDraft";

describe("deviceDraft", () => {
  it("requires an id for any device field, and trims", () => {
    expect(deviceProblem({ product: "", id: "", label: "" })).toBeNull();
    expect(deviceProblem({ product: "P", id: " ", label: "" })).toMatch(/needs an id/);
    expect(deviceIdentity({ product: "", id: "", label: "" })).toBeNull();
    expect(deviceIdentity({ product: " P ", id: " SN1 ", label: "L" })).toEqual({
      id: "SN1",
      product: "P",
      label: "L",
    });
  });
});
