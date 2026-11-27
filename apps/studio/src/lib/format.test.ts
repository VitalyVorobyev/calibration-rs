import { describe, expect, it } from "vitest";
import {
  fixed,
  formatAngle,
  formatCost,
  formatCount,
  formatDuration,
  formatLength,
  formatPixels,
  formatWithSigma,
  sigmaDecimals,
  significant,
} from "./format";

describe("sigmaDecimals", () => {
  it("keeps two significant digits of σ", () => {
    expect(sigmaDecimals(0.214)).toBe(2);
    expect(sigmaDecimals(0.0214)).toBe(3);
    expect(sigmaDecimals(1.4)).toBe(1);
    expect(sigmaDecimals(9.96)).toBe(1);
    expect(sigmaDecimals(12.3)).toBe(0);
    expect(sigmaDecimals(4200)).toBe(0);
  });
  it("is 0 for a σ that is not a positive number", () => {
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(sigmaDecimals(bad)).toBe(0);
    }
  });
});

describe("formatWithSigma", () => {
  it("writes value ± σ at the decimals of σ", () => {
    const f = formatWithSigma(1412.3712, { kind: "sigma", sigma: 0.214 }, "px");
    expect(f.value).toBe("1412.37");
    expect(f.sigma).toBe("0.21");
    expect(f.text).toBe("1412.37 ± 0.21 px");
    expect(f.title).toBeNull();
  });
  it("rounds the value to σ's decimal, large σ included", () => {
    expect(formatWithSigma(1412.37, { kind: "sigma", sigma: 12.3 }).text).toBe(
      "1412 ± 12",
    );
    expect(formatWithSigma(0.00123456, { kind: "sigma", sigma: 0.00021 }).text).toBe(
      "0.00123 ± 0.00021",
    );
  });
  it("never writes a negative zero", () => {
    expect(formatWithSigma(-0.0004, { kind: "sigma", sigma: 0.05 }).text).toBe(
      "0.000 ± 0.050",
    );
  });
  it("writes a value without σ as `value ·` and says why on hover", () => {
    const fixedValue = formatWithSigma(1412.37, { kind: "fixed" }, "px");
    expect(fixedValue.text).toBe("1412.37 · px");
    expect(fixedValue.sigma).toBeNull();
    expect(fixedValue.title).toMatch(/^fixed/);
    const gone = formatWithSigma(2, { kind: "unavailable", reason: "rank deficient" });
    expect(gone.text).toBe("2 ·");
    expect(gone.title).toBe("unavailable: rank deficient");
    expect(formatWithSigma(2, null).title).toBe("no uncertainty known");
  });
  it("treats a σ that is zero or not finite as unavailable, not as ±0", () => {
    for (const sigma of [0, Number.NaN, -3]) {
      const f = formatWithSigma(5, { kind: "sigma", sigma });
      expect(f.sigma).toBeNull();
      expect(f.text).toBe("5 ·");
    }
  });
});

describe("number writing", () => {
  it("fixed uses the minus sign and no negative zero", () => {
    expect(fixed(-1.5, 1)).toBe("−1.5");
    expect(fixed(-0.001, 1)).toBe("0.0");
    expect(fixed(Number.NaN, 2)).toBe("NaN");
    expect(fixed(Number.POSITIVE_INFINITY, 2)).toBe("∞");
  });
  it("significant drops trailing zeros and switches to an exponent at the extremes", () => {
    expect(significant(1412.37)).toBe("1412.37");
    expect(significant(0.5)).toBe("0.5");
    expect(significant(120)).toBe("120");
    expect(significant(0)).toBe("0");
    expect(significant(1.5e-9)).toBe("1.5e−9");
  });
  it("writes pixels, lengths, angles, costs, counts and durations with their unit", () => {
    expect(formatPixels(0.2314)).toBe("0.231 px");
    expect(formatLength(0.0123)).toBe("12.3 mm");
    expect(formatLength(2.5)).toBe("2.5 m");
    expect(formatAngle(Math.PI / 2)).toBe("90°");
    expect(formatCost(0.32)).toBe("3.2e−1");
    expect(formatCost(1234)).toBe("1.2e3");
    expect(formatCount(12345)).toBe("12 345");
    expect(formatDuration(42)).toBe("42 ms");
    expect(formatDuration(3200)).toBe("3.2 s");
    expect(formatDuration(125_000)).toBe("2 min 05 s");
    expect(formatDuration(-1)).toBe("–");
  });
});
