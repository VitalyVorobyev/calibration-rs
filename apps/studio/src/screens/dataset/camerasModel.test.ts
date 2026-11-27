import { describe, expect, it } from "vitest";
import type { CameraSource } from "../../api/generated/types";
import {
  duplicateCamera,
  move,
  newCamera,
  nextCameraId,
  nextIdAfter,
} from "./camerasModel";

const cam = (id: string, pattern = `${id}/*.png`): CameraSource => ({
  id,
  images: { kind: "glob", pattern },
});

describe("camerasModel", () => {
  it("offers the next free camN, skipping taken ids", () => {
    expect(nextCameraId([cam("cam0"), cam("cam1")])).toBe("cam2");
    expect(nextCameraId([cam("left"), cam("cam1")])).toBe("cam2");
    expect(nextCameraId([cam("cam2"), cam("cam0")])).toBe("cam3");
    expect(nextCameraId([])).toBe("cam0");
  });

  it("makes a camera whose images are its own folder", () => {
    expect(newCamera("cam3")).toEqual(cam("cam3"));
  });

  it("counts an id up, keeping its width, past taken ones", () => {
    expect(nextIdAfter("cam0", new Set(["cam0", "cam1"]))).toBe("cam2");
    expect(nextIdAfter("cam09", new Set())).toBe("cam10");
    expect(nextIdAfter("cam007", new Set())).toBe("cam008");
    expect(nextIdAfter("left", new Set(["left"]))).toBe("left2");
  });

  it("duplicates with the next id, its patterns following it", () => {
    const withLaser: CameraSource = {
      ...cam("cam0"),
      laser_images: { kind: "list", paths: ["cam0/laser_1.png", "cam0/laser_2.png"] },
      roi_xywh: [0, 0, 64, 32],
    };
    const copy = duplicateCamera(withLaser, [withLaser, cam("cam1")]);
    expect(copy.id).toBe("cam2");
    expect(copy.images).toEqual({ kind: "glob", pattern: "cam2/*.png" });
    expect(copy.laser_images).toEqual({
      kind: "list",
      paths: ["cam2/laser_1.png", "cam2/laser_2.png"],
    });
    expect(copy.roi_xywh).toEqual([0, 0, 64, 32]);
    // A pattern that does not name the camera is left as it is.
    expect(duplicateCamera(cam("a", "frames/*.png"), [cam("a")]).images).toEqual({
      kind: "glob",
      pattern: "frames/*.png",
    });
  });

  it("moves an item by one place and stops at the ends", () => {
    expect(move(["a", "b", "c"], 1, -1)).toEqual(["b", "a", "c"]);
    expect(move(["a", "b", "c"], 1, 1)).toEqual(["a", "c", "b"]);
    expect(move(["a", "b"], 0, -1)).toEqual(["a", "b"]);
    expect(move(["a", "b"], 1, 1)).toEqual(["a", "b"]);
  });
});
