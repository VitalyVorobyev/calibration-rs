import { describe, expect, it } from "vitest";
import { dirnamePath, joinPath } from "./tauri";

describe("joinPath", () => {
  it("joins with / and drops duplicate separators", () => {
    expect(joinPath("/data/", "kuka_1//", "dataset.toml")).toBe(
      "/data/kuka_1/dataset.toml",
    );
    expect(joinPath("C:\\data", "rig\\a.png")).toBe("C:/data/rig/a.png");
  });

  it("lets an absolute segment replace the base, as an absolute image root does", () => {
    expect(joinPath("/p/runs/run-001", "/data/kuka_1", "01.png")).toBe(
      "/data/kuka_1/01.png",
    );
    expect(joinPath("/p", "D:\\images")).toBe("D:/images");
  });

  it("folds . and .. segments", () => {
    expect(joinPath("/p/runs/run-001", "../..", "img/01.png")).toBe("/p/img/01.png");
    expect(joinPath("/p", ".", "a")).toBe("/p/a");
    expect(joinPath("/", "..", "a")).toBe("/a");
    expect(joinPath("C:/p/q", "../../..")).toBe("C:/");
    expect(joinPath("rel", "../../a")).toBe("../a");
    expect(joinPath("/p/")).toBe("/p");
  });
});

describe("dirnamePath", () => {
  it("is the parent folder, with / separators", () => {
    expect(dirnamePath("/data/kuka_1/dataset.toml")).toBe("/data/kuka_1");
    expect(dirnamePath("C:\\d\\m.json")).toBe("C:/d");
  });
});
