import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { NO_SELECTION } from "../lib/selection";
import { useStore } from "../store";
import { startSelectionBridge, useSelection } from "./selection";

let stop = () => {};

function showRun() {
  useStore.setState({
    data: { diagnostics: {} } as never,
    cameraA: 0,
    selectedPose: 0,
    selectedFeature: null,
  });
}

beforeEach(() => {
  useSelection.setState({ selection: NO_SELECTION, hover: NO_SELECTION });
  useStore.setState({ data: null, cameraA: 0, selectedPose: 0, selectedFeature: null });
  stop = startSelectionBridge();
});

afterEach(() => {
  stop();
});

describe("the selection store", () => {
  it("applies the selection rules", () => {
    const s = useSelection.getState();
    s.setSelection({ camera: 0, view: 2, feature: 7 });
    s.selectView(3);
    expect(useSelection.getState().selection).toEqual({
      camera: 0,
      view: 3,
      feature: null,
    });
  });

  it("clears with escape, one level at a time, and says when there was nothing", () => {
    useSelection.getState().setSelection({ camera: 1, view: null, feature: null });
    expect(useSelection.getState().escape()).toBe(true);
    expect(useSelection.getState().selection).toEqual(NO_SELECTION);
    expect(useSelection.getState().escape()).toBe(false);
  });

  it("keeps the hover apart from the selection", () => {
    useSelection.getState().setHover({ camera: null, view: null, feature: 4 });
    expect(useSelection.getState().hover.feature).toBe(4);
    expect(useSelection.getState().selection).toEqual(NO_SELECTION);
  });
});

describe("the bridge to the export store", () => {
  it("has no selection while no run is shown", () => {
    expect(useSelection.getState().selection).toEqual(NO_SELECTION);
  });

  it("takes the viewers' choices once a run is shown, and follows their clicks", () => {
    showRun();
    expect(useSelection.getState().selection).toEqual({
      camera: 0,
      view: 0,
      feature: null,
    });
    useStore.getState().setSelectedPose(4);
    useStore.getState().setSelectedFeature(9);
    expect(useSelection.getState().selection).toEqual({
      camera: 0,
      view: 4,
      feature: 9,
    });
  });

  it("moves the viewers when the shell selects", () => {
    showRun();
    useSelection.getState().selectCamera(2);
    useSelection.getState().selectView(5);
    expect(useStore.getState().cameraA).toBe(2);
    expect(useStore.getState().selectedPose).toBe(5);
    useSelection.getState().selectFeature(3);
    expect(useStore.getState().selectedFeature).toBe(3);
    useSelection.getState().selectView(6);
    expect(useStore.getState().selectedFeature).toBeNull();
  });

  it("empties the selection when the run goes", () => {
    showRun();
    useStore.setState({ data: null });
    expect(useSelection.getState().selection).toEqual(NO_SELECTION);
  });
});
