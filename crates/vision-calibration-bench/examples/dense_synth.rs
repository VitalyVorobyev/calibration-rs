//! Visual + quantitative evidence for the C5 pure-Rust dense matcher on
//! synthetic ground truth.
//!
//! Runs [`BlockMatcher`] (and the [`OracleMatcher`] reference) on a deterministic
//! slanted-plane rectified pair, scores both with the harness [`evaluate`], and
//! writes a tiled PNG so the result can be eyeballed:
//!
//! ```text
//!   left | right | GT disparity | estimated disparity | error
//! ```
//!
//! The GT and estimate panels share one color scale, so a correct matcher's
//! estimate panel should look like the GT ramp; the error panel should be near-
//! uniformly low. Invalid pixels are rendered black (disparity) / dark grey
//! (error) — never hidden.
//!
//! ```bash
//! cargo run -p vision-calibration-bench --example dense_synth --features tier-b --release
//! # → target/fixtures/dense_synth/composite.png  (+ per-panel PNGs)
//! ```
//!
//! Requires `--features tier-b` (pulls in the `image` crate for PNG output).

#[cfg(not(feature = "tier-b"))]
fn main() {
    eprintln!(
        "dense_synth requires the `tier-b` feature for PNG output:\n  \
         cargo run -p vision-calibration-bench --example dense_synth --features tier-b --release"
    );
    std::process::exit(2);
}

#[cfg(feature = "tier-b")]
fn main() {
    real::run();
}

#[cfg(feature = "tier-b")]
mod real {
    use std::path::{Path, PathBuf};

    use vision_calibration_bench::dense::{
        BlockMatcher, DenseMatchOptions, DenseMatcher, DenseMetrics, DisparityMap, GrayBuffer,
        OracleMatcher, SlantedPlane, evaluate, synthetic_rectified_pair,
    };

    const W: usize = 480;
    const H: usize = 320;
    const SEP: usize = 6; // white separator between composite panels

    pub fn run() {
        // A gently slanted plane: enough disparity sweep for a clear colormap
        // gradient, shallow enough that the fixture's forward-warp foreshortening
        // (GT bias ≈ d·dx/(1+dx)) stays well under a tenth of a pixel.
        let plane = SlantedPlane {
            d0: 8.0,
            dx: 0.012,
            dy: 0.004,
        };
        let (left, right, gt) = synthetic_rectified_pair(W, H, &plane);

        let opts = DenseMatchOptions {
            min_disparity: 0,
            num_disparities: 24,
            block_size: 9,
        };

        let est = BlockMatcher.match_disparity(&left, &right, &opts);
        let oracle = OracleMatcher { gt: gt.clone() }.match_disparity(&left, &right, &opts);

        let m_block = evaluate(&est, &gt, 1.0);
        let m_oracle = evaluate(&oracle, &gt, 1.0);

        // Shared disparity color scale from the GT valid range.
        let (lo, hi) = disp_range(&gt).unwrap_or((0.0, 1.0));

        let panels = [
            gray_to_rgb(&left),
            gray_to_rgb(&right),
            disp_to_rgb(&gt, lo, hi),
            disp_to_rgb(&est, lo, hi),
            error_to_rgb(&est, &gt),
        ];
        let out_dir = fixtures_dir();
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let names = ["left", "right", "gt", "est", "error"];
        for (panel, name) in panels.iter().zip(names) {
            save_rgb(&out_dir.join(format!("{name}.png")), W, H, panel);
        }
        let composite = tile_h(&panels, W, H, SEP);
        let comp_w = W * panels.len() + SEP * (panels.len() - 1);
        save_rgb(&out_dir.join("composite.png"), comp_w, H, &composite);

        // ---- report ----
        println!("C5 dense matcher — synthetic slanted-plane evidence");
        println!(
            "  image {W}x{H}, disparity range [{lo:.2}, {hi:.2}] px, block {}, search {}..{}",
            opts.block_size,
            opts.min_disparity,
            opts.min_disparity + opts.num_disparities
        );
        println!();
        println!(
            "  {:<11} {:>9} {:>8} {:>8} {:>9}",
            "matcher", "density", "rms_px", "mae_px", "bad%>1px"
        );
        report_row("block-zncc", &m_block);
        report_row("oracle", &m_oracle);
        println!();
        println!("  panels: left | right | GT | estimate | error");
        println!("  wrote  {}", out_dir.join("composite.png").display());

        // Quality gate (mirrors the harness integration test): a real matcher
        // must recover most pixels accurately.
        let pass = m_block.density > 0.5 && m_block.rms_px < 1.0;
        if pass {
            println!(
                "\n  PASS — density {:.1}% at RMS {:.3} px",
                m_block.density * 100.0,
                m_block.rms_px
            );
        } else {
            eprintln!(
                "\n  FAIL — density {:.3}, RMS {:.3} px (want density > 0.5, RMS < 1.0)",
                m_block.density, m_block.rms_px
            );
            std::process::exit(1);
        }
    }

    fn report_row(name: &str, m: &DenseMetrics) {
        println!(
            "  {:<11} {:>8.1}% {:>8.3} {:>8.3} {:>8.1}%",
            name,
            m.density * 100.0,
            m.rms_px,
            m.mae_px,
            m.bad_pixel_rate * 100.0
        );
    }

    // ---- color & layout helpers -------------------------------------------

    /// Jet colormap, `t` in `[0, 1]` → `[r, g, b]`.
    fn jet(t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        let r = ((1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0) * 255.0) as u8;
        let g = ((1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0) * 255.0) as u8;
        let b = ((1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0) * 255.0) as u8;
        [r, g, b]
    }

    /// Stepped error palette (teal → red), matching the app's `colorForError`
    /// language but scaled to sub-pixel disparity error.
    fn error_color(e: f32) -> [u8; 3] {
        if e < 0.25 {
            [26, 188, 156]
        } else if e < 0.5 {
            [46, 204, 113]
        } else if e < 1.0 {
            [241, 196, 15]
        } else if e < 2.0 {
            [230, 126, 34]
        } else {
            [231, 76, 60]
        }
    }

    fn disp_range(d: &DisparityMap) -> Option<(f32, f32)> {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        let mut any = false;
        for &v in &d.data {
            if v.is_finite() {
                any = true;
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        any.then_some((lo, hi))
    }

    fn gray_to_rgb(buf: &GrayBuffer) -> Vec<[u8; 3]> {
        buf.data.iter().map(|&v| [v, v, v]).collect()
    }

    fn disp_to_rgb(d: &DisparityMap, lo: f32, hi: f32) -> Vec<[u8; 3]> {
        let span = (hi - lo).max(1e-6);
        d.data
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    jet((v - lo) / span)
                } else {
                    [0, 0, 0] // invalid → black
                }
            })
            .collect()
    }

    fn error_to_rgb(est: &DisparityMap, gt: &DisparityMap) -> Vec<[u8; 3]> {
        (0..est.data.len())
            .map(|i| {
                let e = est.data[i];
                let g = gt.data[i];
                if e.is_finite() && g.is_finite() {
                    error_color((e - g).abs())
                } else {
                    [40, 40, 40] // not evaluated → dark grey
                }
            })
            .collect()
    }

    /// Tile equal-size RGB panels horizontally with white separators.
    fn tile_h(panels: &[Vec<[u8; 3]>], w: usize, h: usize, sep: usize) -> Vec<[u8; 3]> {
        let n = panels.len();
        let total_w = w * n + sep * (n - 1);
        let mut out = vec![[255u8, 255, 255]; total_w * h];
        for (p, panel) in panels.iter().enumerate() {
            let x0 = p * (w + sep);
            for y in 0..h {
                for x in 0..w {
                    out[y * total_w + x0 + x] = panel[y * w + x];
                }
            }
        }
        out
    }

    fn save_rgb(path: &Path, w: usize, h: usize, rgb: &[[u8; 3]]) {
        let flat: Vec<u8> = rgb.iter().flat_map(|p| p.iter().copied()).collect();
        let img = image::RgbImage::from_raw(w as u32, h as u32, flat).expect("rgb buffer size");
        img.save(path).expect("write png");
    }

    fn fixtures_dir() -> PathBuf {
        // CARGO_MANIFEST_DIR = <root>/crates/vision-calibration-bench
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = manifest
            .ancestors()
            .nth(2)
            .expect("workspace root above crates/<crate>");
        root.join("target/fixtures/dense_synth")
    }
}
