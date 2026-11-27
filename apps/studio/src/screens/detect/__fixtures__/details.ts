/** Wire-typed detections for the Detect screen's tests, one per target
 * kind: what `detection_detail` answers, small enough to read. */
import type {
  CornerDetail,
  DetectionDetail,
  DetectionOverview,
  Feature,
} from "../../../api/generated/types";

const QUALITY = {
  local_residual_count: 2,
  local_residual_rms_px: 0.04,
  median_sigma_px: 0.03,
  noise_sigma: 1.5,
};

function feature(i: number, j: number, cov = false): Feature {
  return {
    image_xy: [100 + 20 * i, 80 + 20 * j],
    world_xyz: [0.02 * i, 0.02 * j, 0],
    ...(cov ? { cov_px: [0.0016, 0.0003, 0.0009] as [number, number, number] } : {}),
  };
}

function corner(i: number, j: number, extra: Partial<CornerDetail> = {}): CornerDetail {
  return { grid: [i, j], score: 0.9, ...extra };
}

/** A 2 × 2 corner chessboard, one candidate left out. */
export function chessboardDetail(): DetectionDetail {
  return {
    features: [
      feature(0, 0, true),
      feature(1, 0, true),
      feature(0, 1, true),
      feature(1, 1, true),
    ],
    outcome: { status: "detected" },
    rejected: [
      {
        image_xy: [300, 40],
        reason: { kind: "unlabelled", strength: 0.31 },
      },
    ],
    target: {
      kind: "chessboard",
      cell_size_px: 20,
      corners: [
        corner(0, 0, { axes_rad: [0, 1.57] }),
        corner(1, 0, { axes_rad: [0, 1.57] }),
        corner(0, 1, { axes_rad: [0, 1.57] }),
        corner(1, 1, { axes_rad: [0, 1.57] }),
      ],
      quality: QUALITY,
    },
  };
}

/** A ChArUco view: two corners, one decoded marker, one misread marker cell. */
export function charucoDetail(): DetectionDetail {
  const quad = (
    x: number,
    y: number,
  ): [[number, number], [number, number], [number, number], [number, number]] => [
    [x, y],
    [x + 20, y],
    [x + 20, y + 20],
    [x, y + 20],
  ];
  return {
    features: [feature(1, 1), feature(2, 1)],
    outcome: { status: "detected" },
    rejected: [],
    target: {
      kind: "charuco",
      corners: [corner(1, 1, { id: 5 }), corner(2, 1, { id: 6 })],
      markers: [
        {
          cell: [1, 1],
          hamming: 0,
          id: 12,
          quad_px: quad(120, 100),
          rotation: 0,
          score: 1,
        },
      ],
      quality: QUALITY,
      rejected_markers: [
        { decoded_id: 17, expected_id: 23, quad_px: quad(160, 100), score: -3 },
      ],
    },
  };
}

/** A PuzzleBoard view: two corners joined by one edge bit that disagrees
 * with the master pattern. */
export function puzzleboardDetail(dissent = 0.05): DetectionDetail {
  return {
    features: [feature(0, 0), feature(1, 0)],
    outcome: { status: "detected" },
    rejected: [],
    target: {
      kind: "puzzleboard",
      corners: [corner(10, 20, { id: 4021 }), corner(11, 20, { id: 4022 })],
      decode: {
        bit_error_rate: 0.1,
        dot_dissent_rate: dissent,
        edges_matched: 7,
        edges_observed: 8,
        logical_bit_error_rate: 0.1,
        master_origin: [10, 20],
      },
      edge_bits: [
        {
          bit: 1,
          confidence: 0.8,
          endpoints_px: [
            [100, 80],
            [120, 80],
          ],
          matches_master: false,
        },
      ],
      quality: QUALITY,
    },
  };
}

/** A ring grid view: one photometric ring with both ellipses and a
 * covariance, one plain ring, and an unanchored view. */
export function ringgridDetail(anchored = false): DetectionDetail {
  const ellipse = (cx: number, cy: number, a: number, b: number) => ({
    angle_rad: 0.2,
    center_px: [cx, cy] as [number, number],
    semi_axes_px: [a, b] as [number, number],
  });
  return {
    features: [feature(0, 0), feature(1, 0)],
    outcome: { status: "detected" },
    rejected: [
      { image_xy: [200, 200], reason: { kind: "unanchored", confidence: 0.6 } },
      { image_xy: [220, 200], reason: { kind: "outside_board" } },
    ],
    target: {
      kind: "ringgrid",
      anchored,
      origin_dots_px: [[90, 70]],
      rings: [
        {
          center: "photometric",
          center_cov_px: [0.0004, 0, 0.0004],
          confidence: 0.93,
          grid: [0, 0],
          id: 17,
          inner: ellipse(100, 80, 5, 4),
          outer: ellipse(100, 80, 9, 8),
        },
        {
          center: "outer_ellipse",
          confidence: 0.4,
          grid: [1, 0],
          outer: ellipse(120, 80, 9, 8),
        },
      ],
    },
  };
}

/** The overview of a camera × view dataset: 2 cameras, 3 views, a 20-corner
 * board. View 0002 has a partial board in cam0, cam1 has no image for 0003. */
export function overview(): DetectionOverview {
  const detected = (image: string, features: number) => ({
    image,
    state: {
      kind: "detected" as const,
      features,
      outcome: { status: "detected" as const },
    },
  });
  return {
    cameras: ["cam0", "cam1"],
    expected: 20,
    run: null,
    views: [
      { token: "0001", images: [detected("a/0001.png", 20), detected("b/0001.png", 20)] },
      {
        token: "0002",
        images: [
          { ...detected("a/0002.png", 14), left_out: "partial_chessboard" as const },
          {
            image: "b/0002.png",
            state: {
              kind: "detected" as const,
              features: 0,
              outcome: {
                status: "not_found" as const,
                reason: "No grid of corners found",
              },
            },
          },
        ],
      },
      {
        token: "0003",
        images: [{ image: "a/0003.png", state: { kind: "not_detected" as const } }, null],
      },
    ],
  };
}
