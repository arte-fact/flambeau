//! GGUF v2 / v3 reader — `mmap`-backed.
//!
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.
//! Candle's `gguf_file.rs` is the primary port reference.
//!
//! Design notes:
//! - The file is `mmap`'d and held as an `Arc<Mmap>`. Tensor reads return a
//!   byte slice that borrows from the mmap for as long as the reader lives —
//!   no copy into an owned `Vec<u8>`.
//! - Per-rank tensor-range reads (`tensor_row_range`, `tensor_expert_range`)
//!   are the X5 sharded-load pattern: a rank loads only its shard from disk,
//!   avoiding the 2× VRAM spike of "load full → narrow on-device".
//! - Metadata values are decoded eagerly (they're small). Tensor payloads are
//!   never decoded here — callers either dequantise via [`crate::dequant`] or
//!   hand the raw bytes to a backend-specific upload path.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
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
    /// Byte offset of the tensor payload relative to `tensor_data_offset`.
    pub rel_offset: u64,
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

/// Parsed GGUF header + tensor index. The underlying `mmap` is kept alive via
/// `Arc` so tensor byte slices returned by `tensor_raw` borrow from it.
#[derive(Debug)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: GgufVersion,
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    /// Stable declaration order of tensors (for `inspect-gguf` / iteration).
    pub tensor_order: Vec<String>,
    pub tensor_data_offset: u64,
    mmap: Arc<Mmap>,
}

impl GgufFile {
    /// Open `path`, mmap it, parse the header and tensor index.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // SAFETY: we treat the mmap as read-only for the reader's lifetime.
        // A concurrent writer modifying the file would make reads unsafe, but
        // model weights are a cold artefact — callers must not mutate them.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(path.to_path_buf(), Arc::new(mmap))
    }

    /// Parse a GGUF header from an already-mmapped blob. Split from [`open`]
    /// so tests can feed in-memory fixtures.
    pub fn from_mmap(path: PathBuf, mmap: Arc<Mmap>) -> Result<Self> {
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
            // GGUF stores dims in fastest-varying-first order; flip so
            // outer axes come first (matches candle + llama.cpp tensor views).
            dims.reverse();
            let dtype_wire = cur.read_u32::<LittleEndian>()?;
            let dtype = GgmlDType::from_wire(dtype_wire)?;
            let rel_offset = cur.read_u64::<LittleEndian>()?;
            let info = TensorInfo {
                name: name.clone(),
                dtype,
                dims,
                rel_offset,
            };
            tensor_order.push(name.clone());
            tensors.insert(name, info);
        }

        let header_end = cur.position();
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);
        let tensor_data_offset = header_end.div_ceil(alignment) * alignment;

        Ok(Self {
            path,
            version,
            metadata,
            tensors,
            tensor_order,
            tensor_data_offset,
            mmap,
        })
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
    pub fn tensor_raw(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        self.raw_for(info, 0, info.size_in_bytes())
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
        let abs_start = self.tensor_data_offset + info.rel_offset + byte_start;
        let abs_end = abs_start + byte_len;
        // File bounds — a truncated / inconsistent GGUF (tensor offsets
        // claim data past the mapped length) would otherwise panic inside
        // the slice indexing below. Report a structured error instead so
        // callers can skip / re-download cleanly.
        if (abs_end as usize) > self.mmap.len() {
            return Err(QuantError::RangeOutOfBounds {
                name: info.name.clone(),
                start: byte_start,
                len: byte_len,
                total: (self.mmap.len() as u64).saturating_sub(
                    self.tensor_data_offset + info.rel_offset,
                ),
            });
        }
        Ok(&self.mmap[abs_start as usize..abs_end as usize])
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| QuantError::UnknownTensor {
                name: name.to_string(),
            })
    }

    /// Dequantise `name`'s full payload to a fresh `Vec<f32>`.
    pub fn dequantize_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        let raw = self.tensor_raw(name)?;
        let elem_count = info.elem_count() as usize;
        crate::dequant::dequantize_to_vec(info.dtype, raw, elem_count)
    }

    /// X5 pattern: read rows `[row_start, row_start + row_count)` of a 2D
    /// tensor. Each row is `cols` elements; `cols` must be a multiple of the
    /// dtype's block_size. Returns the raw packed bytes for this row range.
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
