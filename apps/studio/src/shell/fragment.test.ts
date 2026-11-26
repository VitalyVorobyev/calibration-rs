import { describe, expect, it } from "vitest";
import { parseStartFragment } from "./fragment";

describe("parseStartFragment", () => {
  it("reads the token and the project", () => {
    expect(parseStartFragment("#token=abc-_9&project=%2Fdata%2Frig%20demo")).toEqual({
      token: "abc-_9",
      project: "/data/rig demo",
    });
    expect(parseStartFragment("#token=t")).toEqual({ token: "t", project: null });
  });

  it("leaves routes and empty fragments alone", () => {
    expect(parseStartFragment("")).toBeNull();
    expect(parseStartFragment("#")).toBeNull();
    expect(parseStartFragment("#/diagnose")).toBeNull();
    expect(parseStartFragment("#other=1")).toBeNull();
  });
});
