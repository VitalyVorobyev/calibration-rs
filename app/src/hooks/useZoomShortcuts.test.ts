import { describe, expect, it } from "vitest";
import { isEditableTarget, zoomKeyAction } from "./useZoomShortcuts";

describe("zoomKeyAction", () => {
  it("maps the zoom keys", () => {
    expect(zoomKeyAction("f")).toBe("fit");
    expect(zoomKeyAction("F")).toBe("fit");
    expect(zoomKeyAction("1")).toBe("oneToOne");
    expect(zoomKeyAction("+")).toBe("zoomIn");
    expect(zoomKeyAction("=")).toBe("zoomIn");
    expect(zoomKeyAction("-")).toBe("zoomOut");
    expect(zoomKeyAction("_")).toBe("zoomOut");
    expect(zoomKeyAction("l")).toBe("link");
  });

  it("ignores every other key", () => {
    expect(zoomKeyAction("ArrowLeft")).toBeNull();
    expect(zoomKeyAction("2")).toBeNull();
  });
});

describe("isEditableTarget", () => {
  it("is true for text fields and editable content only", () => {
    const el = (tagName: string, isContentEditable = false) =>
      ({ tagName, isContentEditable }) as unknown as EventTarget;
    expect(isEditableTarget(el("INPUT"))).toBe(true);
    expect(isEditableTarget(el("TEXTAREA"))).toBe(true);
    expect(isEditableTarget(el("SELECT"))).toBe(true);
    expect(isEditableTarget(el("DIV", true))).toBe(true);
    expect(isEditableTarget(el("DIV"))).toBe(false);
    expect(isEditableTarget(null)).toBe(false);
  });
});
