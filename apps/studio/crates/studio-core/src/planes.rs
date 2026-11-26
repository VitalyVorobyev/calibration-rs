//! Value planes: dense scalar fields (disparity, depth, a map's channels)
//! in the binary form `@vitavision/stage2d`'s `decodePlane` reads.
//!
//! The format, `VAM1`: a 24-byte header — the magic `VAM1` (four ASCII
//! bytes), then little-endian `u32` width, height, stride and channels, and
//! 4 reserved zero bytes — followed by `width · height · channels`
//! little-endian `f32` values, plane-major (channel, then row, then
//! column). `stride` is the integer decimation applied upstream (`1`: every
//! source pixel is there). `NaN` marks a pixel without a value.

/// The magic that opens every plane.
pub const MAGIC: &[u8; 4] = b"VAM1";
/// The header's length in bytes.
pub const HEADER_LEN: usize = 24;
/// The media type of a value plane.
pub const MEDIA_TYPE: &str = "application/x-vam1";
/// The media type of raw little-endian `f32` values.
pub const F32_MEDIA_TYPE: &str = "application/x-f32";

/// Why a plane could not be encoded.
#[derive(Debug, PartialEq, Eq)]
pub struct PlaneError(pub String);

impl std::fmt::Display for PlaneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// `channels` (each `width · height` values, row-major) as one plane.
pub fn encode(
    width: usize,
    height: usize,
    stride: u32,
    channels: &[&[f32]],
) -> Result<Vec<u8>, PlaneError> {
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| PlaneError("the plane is too large".into()))?;
    if channels.is_empty() {
        return Err(PlaneError("a plane has at least one channel".into()));
    }
    if let Some(bad) = channels.iter().position(|c| c.len() != pixels) {
        return Err(PlaneError(format!(
            "channel {bad} has {} values, not {width}×{height}",
            channels[bad].len()
        )));
    }
    let as_u32 = |value: usize, what: &str| {
        u32::try_from(value).map_err(|_| PlaneError(format!("the {what} does not fit a u32")))
    };
    let mut out = Vec::with_capacity(HEADER_LEN + pixels * channels.len() * 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&as_u32(width, "width")?.to_le_bytes());
    out.extend_from_slice(&as_u32(height, "height")?.to_le_bytes());
    out.extend_from_slice(&stride.to_le_bytes());
    out.extend_from_slice(&as_u32(channels.len(), "channel count")?.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    for channel in channels {
        for value in *channel {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(out)
}

/// `values` as raw little-endian `f32` bytes (a point cloud's positions).
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The golden plane the frontend's `decodePlane` test reads
    /// (`src/api/__fixtures__/golden.vam1`): 3 × 2, two channels, a `NaN`.
    fn golden() -> Vec<u8> {
        let first = [0.0, 1.5, -2.0, 3.25, f32::NAN, 1e6];
        let second = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        encode(3, 2, 1, &[&first, &second]).unwrap()
    }

    #[test]
    fn the_header_and_values_are_little_endian_and_plane_major() {
        let bytes = golden();
        assert_eq!(&bytes[..4], b"VAM1");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 2);
        assert_eq!(&bytes[20..24], &[0; 4]);
        assert_eq!(bytes.len(), HEADER_LEN + 12 * 4);
        let value = |i: usize| {
            let at = HEADER_LEN + 4 * i;
            f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
        };
        assert_eq!(value(1), 1.5);
        assert!(value(4).is_nan());
        assert_eq!(value(6), 10.0);
    }

    #[test]
    fn the_committed_golden_plane_is_what_the_encoder_writes() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/api/__fixtures__/golden.vam1");
        if std::env::var_os("STUDIO_BLESS").is_some() {
            std::fs::write(&path, golden()).unwrap();
        }
        assert_eq!(std::fs::read(&path).unwrap(), golden());
    }

    #[test]
    fn mismatched_channels_are_refused() {
        let err = encode(2, 2, 1, &[&[0.0; 4], &[0.0; 3]]).unwrap_err();
        assert!(err.0.contains("channel 1"), "{err}");
        assert!(encode(2, 2, 1, &[]).is_err());
    }
}
