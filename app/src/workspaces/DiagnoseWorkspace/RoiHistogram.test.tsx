// @vitest-environment jsdom
/** The Diagnose ROI histogram highlights the bin the old hand-rolled
 * component did: `floor(intensity * 64 / 256)`. */
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { RoiHistogram } from "./RoiHistogram";

const BINS = 64;
const counts = Array.from({ length: BINS }, (_, i) => i + 1);

afterEach(cleanup);

describe("RoiHistogram", () => {
  it.each([0, 3.9, 4, 127.5, 200, 255])(
    "highlights bin floor(%d*64/256)",
    (intensity) => {
      const { container } = render(
        <RoiHistogram counts={counts} intensity={intensity} />,
      );
      const cursor = container.querySelector("[data-cursor-bin]");
      expect(cursor?.getAttribute("data-cursor-bin")).toBe(
        String(Math.floor((intensity * BINS) / 256)),
      );
    },
  );

  it("highlights nothing without a cursor", () => {
    const { container } = render(<RoiHistogram counts={counts} intensity={null} />);
    expect(container.querySelector("[data-cursor-bin]")).toBeNull();
  });

  it("draws an empty frame, not a crash, with no counts", () => {
    const { container } = render(<RoiHistogram counts={[]} intensity={10} />);
    expect(
      container.querySelector('[aria-label="ROI luminance histogram"]'),
    ).toBeTruthy();
  });
});
