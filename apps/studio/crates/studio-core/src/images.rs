//! Source images for the browser: what a webview can decode is passed
//! through, anything else is transcoded to PNG once, and thumbnails are
//! made on demand, both kept in an on-disk cache ([`ImageCache`]).
//!
//! The cache key is the image's metadata (canonical path, length, mtime),
//! not its content: keying a thumbnail must not read a 20 MB frame.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use image::{DynamicImage, ImageFormat, ImageReader};
use sha2::{Digest, Sha256};

use crate::api::resource::hex;
use crate::api::{ErrorKind, StudioError};

/// Bumped whenever an encoder's output changes, so old cache entries are
/// never served for new requests.
pub const ENCODER_REVISION: &str = "1";
/// The largest image decoded, in pixels.
pub const MAX_PIXELS: u64 = 200_000_000;
/// The most memory one decode may allocate.
pub const MAX_ALLOC: u64 = 1 << 30;
/// Thumbnail sizes (longest side); a request snaps up to the next one.
pub const THUMBNAIL_SIZES: [u32; 4] = [160, 320, 640, 1280];
/// The cache's size cap.
pub const CACHE_CAP: u64 = 1 << 30;
/// The cache is pruned every this many writes (and when it is opened).
const PRUNE_EVERY: u64 = 256;

/// The formats a browser decodes, which are served as the file itself.
fn passthrough(format: ImageFormat) -> Option<&'static str> {
    Some(match format {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Png => "image/png",
        ImageFormat::WebP => "image/webp",
        ImageFormat::Gif => "image/gif",
        ImageFormat::Bmp => "image/bmp",
        _ => return None,
    })
}

/// The format of the image file at `path`, from its content.
pub fn format_of(path: &Path) -> Result<Option<ImageFormat>, StudioError> {
    Ok(ImageReader::open(path)
        .and_then(ImageReader::with_guessed_format)
        .map_err(|e| io_error(path, e))?
        .format())
}

/// The media type a browser decodes the file at `path` as, if it does.
pub fn browser_media_type(path: &Path) -> Result<Option<&'static str>, StudioError> {
    Ok(format_of(path)?.and_then(passthrough))
}

/// The longest side a thumbnail request for `max` is served at.
pub fn thumbnail_size(max: u32) -> u32 {
    THUMBNAIL_SIZES
        .into_iter()
        .find(|&size| size >= max)
        .unwrap_or(THUMBNAIL_SIZES[THUMBNAIL_SIZES.len() - 1])
}

/// Decode the image at `path` within the size and memory limits.
pub fn decode(path: &Path) -> Result<DynamicImage, StudioError> {
    let (w, h) = image::image_dimensions(path)
        .map_err(|e| StudioError::new(ErrorKind::Io, format!("{}: {e}", path.display())))?;
    if u64::from(w) * u64::from(h) > MAX_PIXELS {
        return Err(StudioError::new(
            ErrorKind::TooLarge,
            format!(
                "{}: {w}×{h} is more than {MAX_PIXELS} pixels",
                path.display()
            ),
        ));
    }
    let mut reader = ImageReader::open(path)
        .and_then(ImageReader::with_guessed_format)
        .map_err(|e| io_error(path, e))?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_ALLOC);
    reader.limits(limits);
    reader
        .decode()
        .map_err(|e| StudioError::new(ErrorKind::Io, format!("decode {}: {e}", path.display())))
}

/// `image` as an 8-bit PNG: a 16-bit image is reduced as the detectors'
/// loader reduces it (`image`'s own conversion), so the view is what was
/// detected; grey stays grey and colour stays colour.
pub fn png8(image: &DynamicImage) -> Result<Vec<u8>, StudioError> {
    let eight = match image {
        DynamicImage::ImageLuma8(_) | DynamicImage::ImageRgb8(_) | DynamicImage::ImageRgba8(_) => {
            image.clone()
        }
        DynamicImage::ImageLuma16(_)
        | DynamicImage::ImageLumaA8(_)
        | DynamicImage::ImageLumaA16(_) => DynamicImage::ImageLuma8(image.to_luma8()),
        other if other.color().has_alpha() => DynamicImage::ImageRgba8(other.to_rgba8()),
        other => DynamicImage::ImageRgb8(other.to_rgb8()),
    };
    encode(&eight, ImageFormat::Png)
}

/// A JPEG (quality 80) of `image` with its longest side at most `size`.
pub fn thumbnail(image: &DynamicImage, size: u32) -> Result<Vec<u8>, StudioError> {
    let small = if image.width().max(image.height()) > size {
        image.thumbnail(size, size)
    } else {
        image.clone()
    };
    let rgb = DynamicImage::ImageRgb8(small.to_rgb8());
    let mut out = Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 80)
        .encode_image(&rgb)
        .map_err(|e| StudioError::internal(format!("encode a thumbnail: {e}")))?;
    Ok(out.into_inner())
}

/// `image` encoded as `format`.
pub fn encode(image: &DynamicImage, format: ImageFormat) -> Result<Vec<u8>, StudioError> {
    let mut out = Cursor::new(Vec::new());
    image
        .write_to(&mut out, format)
        .map_err(|e| StudioError::internal(format!("encode an image: {e}")))?;
    Ok(out.into_inner())
}

fn io_error(path: &Path, e: std::io::Error) -> StudioError {
    StudioError::new(ErrorKind::Io, format!("{}: {e}", path.display())).at(path.to_string_lossy())
}

/// The derived images of the source files: transcodes, thumbnails,
/// undistorted views. Files under `<cache-dir>/images/<aa>/<key>.<ext>`,
/// least-recently-used ones pruned past [`CACHE_CAP`]. It lives outside the
/// roots on purpose: a dataset may be read-only, and the cache is per
/// machine.
pub struct ImageCache {
    dir: PathBuf,
    cap: u64,
    writes: AtomicU64,
    prune: Mutex<()>,
}

impl ImageCache {
    /// The cache under `cache_dir/images`, pruned once now.
    pub fn new(cache_dir: &Path) -> Self {
        Self::with_cap(cache_dir, CACHE_CAP)
    }

    pub fn with_cap(cache_dir: &Path, cap: u64) -> Self {
        let cache = Self {
            dir: cache_dir.join("images"),
            cap,
            writes: AtomicU64::new(0),
            prune: Mutex::new(()),
        };
        cache.prune();
        cache
    }

    /// The key of a derived image of the file at canonical `path`: its
    /// metadata, the `variant` (what is derived and how) and the encoder
    /// revision.
    pub fn key(path: &Path, variant: &str) -> Result<String, StudioError> {
        let meta = std::fs::metadata(path).map_err(|e| io_error(path, e))?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        let mut hash = Sha256::new();
        for part in [
            path.to_string_lossy().as_bytes(),
            &meta.len().to_le_bytes(),
            &mtime.to_le_bytes(),
            variant.as_bytes(),
            ENCODER_REVISION.as_bytes(),
        ] {
            hash.update((part.len() as u64).to_le_bytes());
            hash.update(part);
        }
        Ok(hex(&hash.finalize()))
    }

    /// The cached file of `key`, made by `make` when it is missing.
    pub fn get_or_make(
        &self,
        key: &str,
        ext: &str,
        make: impl FnOnce() -> Result<Vec<u8>, StudioError>,
    ) -> Result<PathBuf, StudioError> {
        let dir = self.dir.join(&key[..2]);
        let path = dir.join(format!("{key}.{ext}"));
        if path.is_file() {
            // Mark it used, for the least-recently-used pruning.
            if let Ok(file) = std::fs::File::options().append(true).open(&path) {
                let _ = file.set_modified(std::time::SystemTime::now());
            }
            return Ok(path);
        }
        let bytes = make()?;
        let cache_error = |e: std::io::Error| {
            StudioError::internal(format!("image cache {}: {e}", dir.display()))
        };
        std::fs::create_dir_all(&dir).map_err(cache_error)?;
        // Write beside, then rename: a reader never sees half a file.
        let tmp = dir.join(format!(
            ".{key}.{}.{}.tmp",
            std::process::id(),
            self.writes.load(Ordering::Relaxed)
        ));
        std::fs::write(&tmp, &bytes).map_err(cache_error)?;
        std::fs::rename(&tmp, &path).map_err(cache_error)?;
        if self.writes.fetch_add(1, Ordering::Relaxed) % PRUNE_EVERY == PRUNE_EVERY - 1 {
            self.prune();
        }
        Ok(path)
    }

    /// Delete the least recently used files until the cache is under its
    /// cap. Failures are ignored: the cache is best effort.
    pub fn prune(&self) {
        let _guard = self.prune.lock().unwrap_or_else(|e| e.into_inner());
        let mut files = Vec::new();
        let Ok(shards) = std::fs::read_dir(&self.dir) else {
            return;
        };
        for shard in shards.flatten() {
            let Ok(entries) = std::fs::read_dir(shard.path()) else {
                continue;
            };
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata()
                    && meta.is_file()
                {
                    let used = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                    files.push((used, meta.len(), entry.path()));
                }
            }
        }
        let mut total: u64 = files.iter().map(|(_, len, _)| len).sum();
        if total <= self.cap {
            return;
        }
        files.sort();
        for (_, len, path) in files {
            if total <= self.cap {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total -= len;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient16(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageLuma16(image::ImageBuffer::from_fn(w, h, |x, y| {
            image::Luma([((x + y * w) * 257) as u16])
        }))
    }

    #[test]
    fn a_16_bit_tiff_becomes_the_8_bit_png_the_detectors_see() {
        let tmp = tempfile::tempdir().unwrap();
        let tiff = tmp.path().join("frame.tif");
        let source = gradient16(16, 8);
        source.save(&tiff).unwrap();
        assert_eq!(browser_media_type(&tiff).unwrap(), None);
        let png = png8(&decode(&tiff).unwrap()).unwrap();
        let back = image::load_from_memory(&png).unwrap();
        assert!(
            matches!(back, DynamicImage::ImageLuma8(_)),
            "{:?}",
            back.color()
        );
        // The detectors' loader: `to_luma8` of the decoded image.
        assert_eq!(back.to_luma8(), source.to_luma8());
    }

    #[test]
    fn browser_formats_pass_through() {
        let tmp = tempfile::tempdir().unwrap();
        let png = tmp.path().join("a.png");
        gradient16(4, 4).save(&png).unwrap();
        assert_eq!(browser_media_type(&png).unwrap(), Some("image/png"));
        // The content decides, not the name.
        let jpeg_named_png = tmp.path().join("b.png");
        std::fs::write(&jpeg_named_png, thumbnail(&gradient16(4, 4), 160).unwrap()).unwrap();
        assert_eq!(
            browser_media_type(&jpeg_named_png).unwrap(),
            Some("image/jpeg")
        );
    }

    #[test]
    fn thumbnails_snap_to_a_fixed_size() {
        assert_eq!(thumbnail_size(1), 160);
        assert_eq!(thumbnail_size(160), 160);
        assert_eq!(thumbnail_size(161), 320);
        assert_eq!(thumbnail_size(5000), 1280);
        let jpeg = thumbnail(&gradient16(1000, 500), 160).unwrap();
        let small = image::load_from_memory(&jpeg).unwrap();
        assert_eq!((small.width(), small.height()), (160, 80));
    }

    #[test]
    fn the_key_follows_the_metadata_and_the_variant() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("a.png");
        std::fs::write(&file, b"one").unwrap();
        let a = ImageCache::key(&file, "thumb:160").unwrap();
        assert_eq!(a, ImageCache::key(&file, "thumb:160").unwrap());
        assert_ne!(a, ImageCache::key(&file, "thumb:320").unwrap());
        std::fs::write(&file, b"three").unwrap();
        assert_ne!(a, ImageCache::key(&file, "thumb:160").unwrap());
    }

    #[test]
    fn the_cache_makes_once_and_prunes_the_least_recently_used() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = ImageCache::with_cap(tmp.path(), 10);
        let mut made = 0;
        let first = cache
            .get_or_make("aa01", "png", || {
                made += 1;
                Ok(vec![0; 6])
            })
            .unwrap();
        let again = cache
            .get_or_make("aa01", "png", || {
                made += 1;
                Ok(vec![0; 6])
            })
            .unwrap();
        assert_eq!((first.clone(), made), (again, 1));
        // Older than the next file, so it goes first.
        std::fs::File::options()
            .append(true)
            .open(&first)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1))
            .unwrap();
        let second = cache.get_or_make("bb02", "png", || Ok(vec![0; 6])).unwrap();
        cache.prune();
        assert!(!first.exists());
        assert!(second.exists());
    }

    #[test]
    fn a_huge_image_is_refused_before_it_is_decoded() {
        // A BMP header that claims 20000 × 20000 pixels, and no pixels.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("huge.bmp");
        let mut bmp = Vec::new();
        bmp.extend_from_slice(b"BM");
        bmp.extend_from_slice(&54u32.to_le_bytes()); // file size (unchecked)
        bmp.extend_from_slice(&0u32.to_le_bytes()); // reserved
        bmp.extend_from_slice(&54u32.to_le_bytes()); // pixel data offset
        bmp.extend_from_slice(&40u32.to_le_bytes()); // BITMAPINFOHEADER
        bmp.extend_from_slice(&20_000i32.to_le_bytes());
        bmp.extend_from_slice(&20_000i32.to_le_bytes());
        bmp.extend_from_slice(&1u16.to_le_bytes()); // planes
        bmp.extend_from_slice(&24u16.to_le_bytes()); // bits per pixel
        bmp.extend_from_slice(&[0; 24]); // no compression, sizes, palette
        std::fs::write(&path, bmp).unwrap();
        assert_eq!(decode(&path).unwrap_err().kind, ErrorKind::TooLarge);
    }
}
