//! GGUF v2 / v3 reader — `mmap`-backed.
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.
//! Candle's `gguf_file.rs` is the primary port reference.
//! Design notes:
//! - The file is `mmap`'d and held as an `Arc<Mmap>`. Tensor reads return a
//! byte slice that borrows from the mmap for as long as the reader lives —
//! no copy into an owned `Vec<u8>`.
//! - Per-rank tensor-range reads (`tensor_row_range`, `tensor_expert_range`)
//! are the X5 sharded-load pattern: a rank loads only its shard from disk,
//! avoiding the 2× VRAM spike of "load full → narrow on-device".
//! - Metadata values are decoded eagerly (they're small). Tensor payloads are
//! never decoded here — callers either dequantise via [`crate::dequant`] or
//! hand the raw bytes to a backend-specific upload path.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

#[cfg(feature = "dev_trace")]
fn dev_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}
#[cfg(not(feature = "dev_trace"))]
#[inline(always)]
fn dev_flag(_name: &str) -> bool {
    false
}
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;

use crate::dtype::GgmlDType;
use crate::error::{QuantError, Result};

const MAGIC_GGUF: u32 = 0x46554747; // "GGUF" little-endian

pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufVersion {
    V2,
    V3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    fn from_wire(u: u32) -> Result<Self> {
        Ok(match u {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return Err(QuantError::InvalidValueType { tag: u }),
        })
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U8(v) => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            Self::U32(v) => Some(*v),
            Self::I8(v) if *v >= 0 => Some(*v as u32),
            Self::I16(v) if *v >= 0 => Some(*v as u32),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U8(v) => Some(*v as u64),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            Self::I32(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: GgmlDType,
    /// Logical dimensions, outer-major (element 0 is the slowest-varying axis).
    /// The on-disk GGUF order is row-major reversed; we un-reverse here so
    /// `dims[0] * dims[1] * …` is the natural elem-count.
    pub dims: Vec<u64>,
    /// Byte offset of the tensor payload relative to its split's
    /// `tensor_data_offset`.
    pub rel_offset: u64,
    /// Which split file this tensor's payload lives in. `0` for single-file
    /// GGUFs; for split GGUFs, indexes into `GgufFile.splits`.
    pub split_idx: u16,
}

impl TensorInfo {
    pub fn elem_count(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn block_size(&self) -> u64 {
        self.dtype.block_size() as u64
    }

    pub fn type_size(&self) -> u64 {
        self.dtype.type_size() as u64
    }

    /// Total bytes occupied by this tensor's payload.
    pub fn size_in_bytes(&self) -> u64 {
        let elems = self.elem_count();
        elems / self.block_size() * self.type_size()
    }
}

/// One split file's mmap + its own `tensor_data_offset`. For single-file
/// GGUFs, `GgufFile.splits` has exactly one entry. For multi-part GGUFs
/// (`-NNNNN-of-MMMMM.gguf`), each split's tensor payloads live in its own
/// data section, addressed via `TensorInfo.split_idx`.
#[derive(Debug)]
struct SplitMmap {
    /// Path the mmap came from. Unused on the hot path; kept for `{:?}`
    /// debug output and future per-split error context.
    #[allow(dead_code)]
    path: PathBuf,
    mmap: Arc<Mmap>,
    /// Byte offset within this split file at which its tensor data section
    /// starts (alignment-rounded after the per-split metadata + tensor index).
    tensor_data_offset: u64,
}

/// Parsed GGUF header + tensor index. The underlying `mmap` is kept alive via
/// `Arc` so tensor byte slices returned by `tensor_raw` borrow from it.
///
/// For multi-part GGUFs (`-NNNNN-of-MMMMM.gguf` with `split.count > 1` in
/// part 1's metadata), `splits` holds one entry per part and each tensor's
/// payload is read from `splits[info.split_idx]`. The merged metadata comes
/// from part 1 only (the rest is duplicated). The merged `tensors` /
/// `tensor_order` covers every tensor across all parts.
#[derive(Debug)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: GgufVersion,
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    /// Stable declaration order of tensors (for `inspect-gguf` / iteration).
    pub tensor_order: Vec<String>,
    /// Part-0's `tensor_data_offset`. Kept on the outer struct for backward
    /// compatibility with the `inspect-gguf` printer; tensor reads dispatch
    /// through `splits[info.split_idx].tensor_data_offset` instead.
    pub tensor_data_offset: u64,
    splits: Vec<SplitMmap>,
}

impl GgufFile {
    /// Open `path`, mmap it, parse the header and tensor index.
    /// # Errors
    /// - `std::io::Error` wrapped as `QuantError::Io` if the file can't be
    /// opened or mmapped.
    /// - Propagates [`from_mmap`] errors: `BadMagic`, `UnsupportedVersion`,
    /// `TruncatedHeader`, or a metadata/tensor-index parse error.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        // Parse part 0 first so we can inspect its metadata for split info.
        let (part0, part0_md, part0_tensors, part0_order, part0_offset) =
            Self::open_one_split(path.to_path_buf(), 0)?;

        // Either consolidated (no split metadata, or split.count <= 1) or
        // the user passed a filename without the multi-part suffix → treat
        // as single file.
        let split_count = part0_md
            .get("split.count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let sibling_paths = if split_count > 1 {
            split_sibling_paths(path, split_count as usize)
        } else {
            None
        };

        let Some(sibling_paths) = sibling_paths else {
            return Ok(Self {
                path: path.to_path_buf(),
                version: part0.version,
                metadata: part0_md,
                tensors: part0_tensors,
                tensor_order: part0_order,
                tensor_data_offset: part0_offset,
                splits: vec![part0.into_split(part0_offset)],
            });
        };

        let mut splits = Vec::with_capacity(sibling_paths.len());
        let version = part0.version;
        splits.push(part0.into_split(part0_offset));

        let mut tensors = part0_tensors;
        let mut tensor_order = part0_order;

        for (i, sibling) in sibling_paths.iter().enumerate().skip(1) {
            let (parsed, _md, mut sib_tensors, sib_order, sib_offset) =
                Self::open_one_split(sibling.clone(), i as u16)?;
            for name in sib_order {
                let Some(mut info) = sib_tensors.remove(&name) else {
                    continue;
                };
                if tensors.contains_key(&name) {
                    return Err(QuantError::DuplicateTensor { name });
                }
                info.split_idx = i as u16;
                tensor_order.push(name.clone());
                tensors.insert(name, info);
            }
            splits.push(parsed.into_split(sib_offset));
        }

        Ok(Self {
            path: path.to_path_buf(),
            version,
            metadata: part0_md,
            tensors,
            tensor_order,
            tensor_data_offset: part0_offset,
            splits,
        })
    }

    /// Parse a GGUF header from an already-mmapped blob. Split from [`open`]
    /// so tests can feed in-memory fixtures. Always produces a single-split
    /// GgufFile — multi-part loading flows through [`open`] only.
    /// # Errors
    /// - `QuantError::BadMagic` if the blob doesn't start with `GGUF`.
    /// - `QuantError::UnsupportedVersion` for versions outside {2, 3}.
    /// - `QuantError::TruncatedHeader` if any metadata or tensor-index read
    /// runs past the mmap's length.
    /// - `QuantError::Io` wrapping a `byteorder` short-read error.
    pub fn from_mmap(path: PathBuf, mmap: Arc<Mmap>) -> Result<Self> {
        let (parsed, metadata, tensors, tensor_order, tensor_data_offset) =
            parse_gguf_header(path.clone(), mmap)?;
        let version = parsed.version;
        Ok(Self {
            path,
            version,
            metadata,
            tensors,
            tensor_order,
            tensor_data_offset,
            splits: vec![parsed.into_split(tensor_data_offset)],
        })
    }

    fn open_one_split(
        path: PathBuf,
        split_idx: u16,
    ) -> Result<(
        ParsedSplit,
        HashMap<String, Value>,
        HashMap<String, TensorInfo>,
        Vec<String>,
        u64,
    )> {
        use std::os::unix::io::AsRawFd;
        let file = File::open(&path)?;
        // 7: mirror llama.cpp's loader hints. SEQUENTIAL tells the
        // kernel to prefetch aggressively + evict early pages once we've
        // moved past them. Combined with per-tensor `munmap` during
        // loading, this keeps page-cache pressure bounded for GGUFs
        // larger than host RAM.
        // SAFETY: `posix_fadvise` operates on a kernel-side fd and never reads
        // or writes caller memory. `file.as_raw_fd()` is a valid open fd for
        // the lifetime of `file`. Return code is non-actionable (best-effort
        // page-cache hint), so we ignore it.
        unsafe {
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
        }
        // SAFETY: we treat the mmap as read-only for the reader's lifetime.
        // A concurrent writer modifying the file would make reads unsafe, but
        // model weights are a cold artefact — callers must not mutate them.
        let mmap = unsafe { Mmap::map(&file)? };
        let (parsed, md, mut tensors, order, offset) =
            parse_gguf_header(path, Arc::new(mmap))?;
        for info in tensors.values_mut() {
            info.split_idx = split_idx;
        }
        Ok((parsed, md, tensors, order, offset))
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(Value::as_str)
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(Value::as_u32)
    }

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(Value::as_f32)
    }

    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(Value::as_u64)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata_str("general.architecture")
    }

    /// Zero-copy slice of the full payload for `name`. Lifetime is tied to
    /// `self` (which owns the mmap).
    /// # Errors
    /// - `QuantError::TensorNotFound` if `name` is not in the tensor index.
    /// - `QuantError::TruncatedTensor` if the recorded tensor extent runs
    /// past the mmap's length.
    pub fn tensor_raw(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        self.raw_for(info, 0, info.size_in_bytes())
    }

    /// Hint the kernel that the given tensor's mmap range will not be
    /// re-read (equivalent to `madvise(MADV_DONTNEED)` on just those pages).
    /// Safe to call even if the tensor doesn't exist — it's a best-effort
    /// hint. Used by the weight loader to keep page-cache pressure low when
    /// the GGUF is larger than host RAM: without this, loading Qwen3.6-27B-
    /// UD-Q8_K_XL (33 GiB) on a 31 GiB box blows up per-tensor transfer
    /// latency 100× past the RAM watermark.
    /// Drop the mmap range for this tensor (definitively, via `munmap` —
    /// same technique llama.cpp's `unmap_fragment` uses; `madvise(DONTNEED)`
    /// is advisory and does not reliably free page cache on Linux 5.x/6.x).
    /// Only the page-aligned inner slice is unmapped; partially-used
    /// boundary pages stay mapped so neighbouring tensors still work.
    /// Safe to call even if the tensor doesn't exist — best-effort hint.
    /// Caller must have already transferred (and synced) the bytes off to
    /// GPU. After this, the tensor is no longer accessible via
    /// `tensor_raw`.
    pub fn advise_drop_tensor(&self, name: &str) {
        let Ok(info) = self.info(name) else { return };
        let split = &self.splits[info.split_idx as usize];
        let abs_start = (split.tensor_data_offset + info.rel_offset) as usize;
        let byte_len = info.size_in_bytes() as usize;
        if abs_start + byte_len > split.mmap.len() {
            return;
        }
        // Round START UP to the next page (leave partially-used boundary
        // pages alone — other tensors may share them). Round END DOWN to
        // page boundary for the same reason.
        let page_size = 4096usize;
        let aligned_start = (abs_start + page_size - 1) & !(page_size - 1);
        let aligned_end = (abs_start + byte_len) & !(page_size - 1);
        if aligned_end <= aligned_start {
            return;
        }
        let len = aligned_end - aligned_start;
        // SAFETY: munmapping a range entirely inside our own mmap. After
        // this, accessing those bytes via `tensor_raw` would SIGSEGV; the
        // loader guarantees it's done with this tensor.
        let rc = unsafe {
            let addr = split.mmap.as_ptr().add(aligned_start) as *mut libc::c_void;
            libc::munmap(addr, len)
        };
        if rc != 0 && dev_flag("FLAMBEAU_LOAD_TRACE") {
            eprintln!("  [munmap] {} failed: {}", name,
                std::io::Error::last_os_error());
        }
    }

    /// Zero-copy slice of bytes `[byte_start, byte_start + byte_len)` of
    /// tensor `name`, without dtype alignment checks. Used by the range
    /// helpers below.
    fn raw_range(&self, name: &str, byte_start: u64, byte_len: u64) -> Result<&[u8]> {
        let info = self.info(name)?;
        self.raw_for(info, byte_start, byte_len)
    }

    fn raw_for(&self, info: &TensorInfo, byte_start: u64, byte_len: u64) -> Result<&[u8]> {
        let total = info.size_in_bytes();
        let end = byte_start.saturating_add(byte_len);
        if end > total {
            return Err(QuantError::RangeOutOfBounds {
                name: info.name.clone(),
                start: byte_start,
                len: byte_len,
                total,
            });
        }
        let split = &self.splits[info.split_idx as usize];
        let abs_start = split.tensor_data_offset + info.rel_offset + byte_start;
        let abs_end = abs_start + byte_len;
        // File bounds — a truncated / inconsistent GGUF (tensor offsets
        // claim data past the mapped length) would otherwise panic inside
        // the slice indexing below. Report a structured error instead so
        // callers can skip / re-download cleanly.
        if (abs_end as usize) > split.mmap.len() {
            return Err(QuantError::RangeOutOfBounds {
                name: info.name.clone(),
                start: byte_start,
                len: byte_len,
                total: (split.mmap.len() as u64).saturating_sub(
                    split.tensor_data_offset + info.rel_offset,
                ),
            });
        }
        Ok(&split.mmap[abs_start as usize..abs_end as usize])
    }

    /// Look up a tensor's metadata by name.
    /// # Errors
    /// `QuantError::UnknownTensor` if `name` is not in the index.
    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| QuantError::UnknownTensor {
                name: name.to_string(),
            })
    }

    /// Dequantise `name`'s full payload to a fresh `Vec<f32>`.
    /// # Errors
    /// - `QuantError::UnknownTensor` if `name` is not indexed.
    /// - `QuantError::TruncatedTensor` if the mmap is short.
    /// - `QuantError::UnsupportedDtype` if the dtype lacks a dequantiser.
    pub fn dequantize_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        let raw = self.tensor_raw(name)?;
        let elem_count = info.elem_count() as usize;
        crate::dequant::dequantize_to_vec(info.dtype, raw, elem_count)
    }

    /// X5 pattern: read rows `[row_start, row_start + row_count)` of a 2D
    /// tensor. Each row is `cols` elements; `cols` must be a multiple of the
    /// dtype's block_size. Returns the raw packed bytes for this row range.
    /// # Errors
    /// - `QuantError::UnknownTensor` if `name` is not indexed.
    /// - `QuantError::RangeOutOfBounds` if the tensor isn't 2D, rows aren't
    /// aligned to the dtype's block boundary, or `row_start + row_count`
    /// exceeds the row count.
    /// - `QuantError::TruncatedTensor` if the underlying mmap is short.
    pub fn tensor_row_range_raw(
        &self,
        name: &str,
        row_start: u64,
        row_count: u64,
    ) -> Result<&[u8]> {
        let info = self.info(name)?;
        if info.dims.len() != 2 {
            return Err(QuantError::RangeOutOfBounds {
                name: name.to_string(),
                start: row_start,
                len: row_count,
                total: info.dims.len() as u64,
            });
        }
        let rows = info.dims[0];
        let cols = info.dims[1];
        let block_size = info.block_size();
        if cols % block_size != 0 {
            return Err(QuantError::RangeUnaligned {
                name: name.to_string(),
                dtype: info.dtype.name(),
                type_size: info.type_size() as usize,
                start: row_start,
                len: row_count,
            });
        }
        if row_start + row_count > rows {
            return Err(QuantError::RangeOutOfBounds {
                name: name.to_string(),
                start: row_start,
                len: row_count,
                total: rows,
            });
        }
        let bytes_per_row = cols / block_size * info.type_size();
        self.raw_range(name, row_start * bytes_per_row, row_count * bytes_per_row)
    }

    /// X5 pattern for MoE: read experts `[e_start, e_start + e_count)` of a
    /// 3D expert tensor `[num_experts, d1, d2]`. `d1 * d2` must be a multiple
    /// of the dtype's block_size.
    /// # Errors
    /// Same error space as [`tensor_row_range_raw`] — `UnknownTensor`,
    /// `RangeOutOfBounds`, or `TruncatedTensor`.
    pub fn tensor_expert_range_raw(
        &self,
        name: &str,
        e_start: u64,
        e_count: u64,
    ) -> Result<&[u8]> {
        let info = self.info(name)?;
        if info.dims.len() != 3 {
            return Err(QuantError::RangeOutOfBounds {
                name: name.to_string(),
                start: e_start,
                len: e_count,
                total: info.dims.len() as u64,
            });
        }
        let e_total = info.dims[0];
        let per_expert_elems = info.dims[1] * info.dims[2];
        let block_size = info.block_size();
        if per_expert_elems % block_size != 0 {
            return Err(QuantError::RangeUnaligned {
                name: name.to_string(),
                dtype: info.dtype.name(),
                type_size: info.type_size() as usize,
                start: e_start,
                len: e_count,
            });
        }
        if e_start + e_count > e_total {
            return Err(QuantError::RangeOutOfBounds {
                name: name.to_string(),
                start: e_start,
                len: e_count,
                total: e_total,
            });
        }
        let bytes_per_expert = per_expert_elems / block_size * info.type_size();
        self.raw_range(name, e_start * bytes_per_expert, e_count * bytes_per_expert)
    }
}

fn read_string(cur: &mut Cursor<&[u8]>) -> Result<String> {
    let len = cur.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cur.read_exact(&mut buf)?;
    // GGUF strings are supposed to be UTF-8 but a few GGUFs in the wild carry
    // trailing NULs or invalid bytes — mirror candle's tolerance and strip NUL,
    // then fall back to `from_utf8_lossy`.
    while let Some(0) = buf.last() {
        buf.pop();
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_value(cur: &mut Cursor<&[u8]>, vt: ValueType) -> Result<Value> {
    Ok(match vt {
        ValueType::U8 => Value::U8(cur.read_u8()?),
        ValueType::I8 => Value::I8(cur.read_i8()?),
        ValueType::U16 => Value::U16(cur.read_u16::<LittleEndian>()?),
        ValueType::I16 => Value::I16(cur.read_i16::<LittleEndian>()?),
        ValueType::U32 => Value::U32(cur.read_u32::<LittleEndian>()?),
        ValueType::I32 => Value::I32(cur.read_i32::<LittleEndian>()?),
        ValueType::F32 => Value::F32(cur.read_f32::<LittleEndian>()?),
        ValueType::Bool => Value::Bool(cur.read_u8()? != 0),
        ValueType::String => Value::String(read_string(cur)?),
        ValueType::U64 => Value::U64(cur.read_u64::<LittleEndian>()?),
        ValueType::I64 => Value::I64(cur.read_i64::<LittleEndian>()?),
        ValueType::F64 => Value::F64(cur.read_f64::<LittleEndian>()?),
        ValueType::Array => {
            let inner_tag = cur.read_u32::<LittleEndian>()?;
            let inner = ValueType::from_wire(inner_tag)?;
            let len = cur.read_u64::<LittleEndian>()? as usize;
            let mut vs = Vec::with_capacity(len);
            for _ in 0..len {
                vs.push(read_value(cur, inner)?);
            }
            Value::Array(vs)
        }
    })
}

/// One parsed split file. Internal helper for `GgufFile::open`. Holds the
/// mmap + path; gets converted to `SplitMmap` once the data-section offset
/// is computed.
struct ParsedSplit {
    path: PathBuf,
    version: GgufVersion,
    mmap: Arc<Mmap>,
}

impl ParsedSplit {
    fn into_split(self, tensor_data_offset: u64) -> SplitMmap {
        SplitMmap {
            path: self.path,
            mmap: self.mmap,
            tensor_data_offset,
        }
    }
}

/// Parse the GGUF header on `mmap`. Returns (ParsedSplit, metadata,
/// tensors-without-split-idx, tensor_order, tensor_data_offset).
fn parse_gguf_header(
    path: PathBuf,
    mmap: Arc<Mmap>,
) -> Result<(
    ParsedSplit,
    HashMap<String, Value>,
    HashMap<String, TensorInfo>,
    Vec<String>,
    u64,
)> {
    let mut cur = Cursor::new(&mmap[..]);

    let magic = cur.read_u32::<LittleEndian>()?;
    if magic != MAGIC_GGUF {
        return Err(QuantError::BadMagic { magic });
    }
    let version_raw = cur.read_u32::<LittleEndian>()?;
    let version = match version_raw {
        2 => GgufVersion::V2,
        3 => GgufVersion::V3,
        v => return Err(QuantError::UnsupportedVersion { version: v }),
    };
    let tensor_count = cur.read_u64::<LittleEndian>()? as usize;
    let metadata_kv_count = cur.read_u64::<LittleEndian>()? as usize;

    let mut metadata = HashMap::with_capacity(metadata_kv_count);
    for _ in 0..metadata_kv_count {
        let key = read_string(&mut cur)?;
        let tag = cur.read_u32::<LittleEndian>()?;
        let vt = ValueType::from_wire(tag)?;
        let value = read_value(&mut cur, vt)?;
        metadata.insert(key, value);
    }

    let mut tensors = HashMap::with_capacity(tensor_count);
    let mut tensor_order = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_string(&mut cur)?;
        let n_dims = cur.read_u32::<LittleEndian>()? as usize;
        let mut dims = vec![0u64; n_dims];
        cur.read_u64_into::<LittleEndian>(&mut dims)?;
        dims.reverse();
        let dtype_wire = cur.read_u32::<LittleEndian>()?;
        let dtype = GgmlDType::from_wire(dtype_wire)?;
        let rel_offset = cur.read_u64::<LittleEndian>()?;
        let info = TensorInfo {
            name: name.clone(),
            dtype,
            dims,
            rel_offset,
            split_idx: 0,
        };
        tensor_order.push(name.clone());
        tensors.insert(name, info);
    }

    let header_end = cur.position();
    let alignment = metadata
        .get("general.alignment")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_ALIGNMENT);
    let tensor_data_offset = header_end.div_ceil(alignment) * alignment;

    let parsed = ParsedSplit { path, version, mmap };
    Ok((parsed, metadata, tensors, tensor_order, tensor_data_offset))
}

/// If `path` matches the `(.*)-NNNNN-of-MMMMM.gguf` convention, return the
/// full ordered list of sibling files (parts 1..MMMMM) — including `path`
/// itself at index 0. Returns `None` if `path` is not a multi-part split or
/// if the part-1 file isn't the one passed in.
///
/// We accept `split_count` from metadata as authoritative — that's how
/// llama.cpp prefers to validate matched parts. The filename pattern is
/// only used to compute the sibling paths.
fn split_sibling_paths(path: &Path, split_count: usize) -> Option<Vec<PathBuf>> {
    use std::ffi::OsStr;
    let file_name = path.file_name().and_then(OsStr::to_str)?;
    // Pattern: ...-<NNNNN>-of-<MMMMM>.gguf
    let stem = file_name.strip_suffix(".gguf")?;
    let (prefix, rest) = stem.rsplit_once("-of-")?;
    let (head, part_str) = prefix.rsplit_once('-')?;
    let part_num: usize = part_str.parse().ok()?;
    let total_str = rest;
    let total: usize = total_str.parse().ok()?;
    if part_num != 1 || total != split_count {
        return None;
    }
    let width = part_str.len();
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut siblings = Vec::with_capacity(total);
    for i in 1..=total {
        let part = format!("{head}-{i:0width$}-of-{total_str}.gguf", width = width);
        siblings.push(dir.join(part));
    }
    Some(siblings)
}
