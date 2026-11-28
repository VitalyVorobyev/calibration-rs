// @vitest-environment jsdom
/** `fix_poses` as one field: the views held, written as text, writing the
 * list only when the text is valid, and saying what is wrong with it. */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PoseListField } from "./PoseListField";

afterEach(cleanup);

function setup(value: unknown, viewCount: number | null = 6, baseline: unknown = []) {
  const onChange = vi.fn();
  render(
    <PoseListField
      label="Optimize · fix poses"
      value={value}
      baseline={baseline}
      viewCount={viewCount}
      disabled={false}
      onChange={onChange}
    />,
  );
  return { onChange, input: screen.getByRole<HTMLInputElement>("textbox") };
}

describe("PoseListField", () => {
  it("shows the list as text, not as JSON", () => {
    const { input } = setup([0, 2, 3, 4]);
    expect(input.value).toBe("0, 2-4");
    expect(screen.getByText("Optimize · fix poses")).toBeTruthy();
    expect(screen.getByText("4 held")).toBeTruthy();
  });

  it("writes the list a valid text names", () => {
    const { input, onChange } = setup([]);
    expect(screen.getByText("none held")).toBeTruthy();
    fireEvent.change(input, { target: { value: "3, 0-1" } });
    expect(onChange).toHaveBeenLastCalledWith([0, 1, 3]);
    fireEvent.change(input, { target: { value: "" } });
    expect(onChange).toHaveBeenLastCalledWith([]);
  });

  it("keeps a half-typed text and says what is wrong, writing nothing", () => {
    const { input, onChange } = setup([1]);
    fireEvent.change(input, { target: { value: "1, x" } });
    expect(onChange).not.toHaveBeenCalled();
    expect(input.value).toBe("1, x");
    expect(screen.getByText(/“x” is not a view number/)).toBeTruthy();
    fireEvent.blur(input);
    expect(input.value).toBe("1");
  });

  it("warns about a view the dataset does not have, and marks a change", () => {
    setup([2, 9], 6, [2]);
    expect(screen.getByText(/dataset has 6 views; 9 is beyond them/)).toBeTruthy();
    expect(screen.getByText("changed")).toBeTruthy();
    expect(screen.getByText("2")).toBeTruthy();
  });
});
