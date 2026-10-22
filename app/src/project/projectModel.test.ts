import { describe, expect, it } from "vitest";
import {
  RECENT_LIMIT,
  folderName,
  forgetRecent,
  loadRecent,
  nextRunName,
  rememberRecent,
  saveRecent,
} from "./projectModel";

/** An in-memory `Storage`, optionally refusing writes. */
function memoryStorage(initial: Record<string, string> = {}, refuse = false): Storage {
  const data = new Map(Object.entries(initial));
  return {
    get length() {
      return data.size;
    },
    clear: () => data.clear(),
    getItem: (k) => data.get(k) ?? null,
    key: (i) => [...data.keys()][i] ?? null,
    removeItem: (k) => void data.delete(k),
    setItem: (k, v) => {
      if (refuse) throw new Error("QuotaExceededError");
      data.set(k, v);
    },
  };
}

describe("nextRunName", () => {
  it("starts at run-001", () => {
    expect(nextRunName([])).toBe("run-001");
  });

  it("goes one past the highest numbered run, ignoring other names", () => {
    expect(nextRunName(["run-001", "run-007", "baseline", "run-x"])).toBe("run-008");
  });

  it("does not reuse a gap", () => {
    expect(nextRunName(["run-003"])).toBe("run-004");
  });

  it("grows past three digits", () => {
    expect(nextRunName(["run-999"])).toBe("run-1000");
  });
});

describe("recent projects", () => {
  it("moves a reopened project to the front without duplicating it", () => {
    expect(rememberRecent(["/a", "/b", "/c"], "/b")).toEqual(["/b", "/a", "/c"]);
  });

  it("caps the list", () => {
    const many = Array.from({ length: RECENT_LIMIT }, (_, i) => `/p${i}`);
    const next = rememberRecent(many, "/new");
    expect(next).toHaveLength(RECENT_LIMIT);
    expect(next[0]).toBe("/new");
    expect(next).not.toContain(`/p${RECENT_LIMIT - 1}`);
  });

  it("forgets a project", () => {
    expect(forgetRecent(["/a", "/b"], "/a")).toEqual(["/b"]);
  });

  it("round-trips through storage", () => {
    const storage = memoryStorage();
    saveRecent(["/a", "/b"], storage);
    expect(loadRecent(storage)).toEqual(["/a", "/b"]);
  });

  it("reads a corrupt or foreign value as empty", () => {
    expect(loadRecent(memoryStorage({ "calib-recent-projects": "{not json" }))).toEqual(
      [],
    );
    expect(loadRecent(memoryStorage({ "calib-recent-projects": '{"a":1}' }))).toEqual([]);
    expect(loadRecent(memoryStorage({ "calib-recent-projects": '["/a", 3]' }))).toEqual([
      "/a",
    ]);
    expect(loadRecent(null)).toEqual([]);
  });

  it("ignores a store that refuses the write", () => {
    expect(() => saveRecent(["/a"], memoryStorage({}, true))).not.toThrow();
  });
});

describe("folderName", () => {
  it("takes the last segment of either separator, trailing ones ignored", () => {
    expect(folderName("/data/kuka_1")).toBe("kuka_1");
    expect(folderName("/data/kuka_1/")).toBe("kuka_1");
    expect(folderName("C:\\data\\rig")).toBe("rig");
    expect(folderName("plain")).toBe("plain");
  });
});
