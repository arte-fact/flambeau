//! GGUF reader round-trip tests.
//!
//! We build a synthetic GGUF v3 blob in memory (no llama.cpp dependency at
//! test time), feed it to `GgufFile::from_mmap`, and assert that header +
//! metadata + tensor index + raw-range readers all agree with what we wrote.
//!
//! Byte-for-byte cross-checking against llama.cpp `gguf-dump` requires a
//! committed real-GGUF fixture — deferred to a separate integration test.

use std::path::PathBuf;
use std::sync::Arc;

use byteorder::{LittleEndian, WriteBytesExt};
use flambeau_quant::gguf::{GgufFile, GgufVersion, Value};
use flambeau_quant::GgmlDType;

const ALIGNMENT: u64 = 32;

/// Minimal synthetic GGUF v3 writer — enough for the reader tests. Only
/// supports the subset of metadata types actually used below.
#[derive(Default)]
struct Gguf {
    metadata: Vec<(String, MetaValue)>,
    tensors: Vec<TensorSpec>,
}

enum MetaValue {
    U32(u32),
    U64(u64),
    String(String),
    StringArray(Vec<String>),
}

struct TensorSpec {
    name: String,
    dtype: GgmlDType,
    /// ggml on-disk order: fastest-varying axis first.
    dims_wire: Vec<u64>,
    /// Raw packed bytes for the tensor.
    payload: Vec<u8>,
}

impl Gguf {
    fn meta(mut self, key: &str, v: MetaValue) -> Self {
        self.metadata.push((key.to_string(), v));
        self
    }
    fn tensor(mut self, name: &str, dtype: GgmlDType, dims_wire: &[u64], payload: Vec<u8>) -> Self {
        self.tensors.push(TensorSpec {
            name: name.to_string(),
            dtype,
            dims_wire: dims_wire.to_vec(),
            payload,
        });
        self
    }

    fn finish(self) -> Vec<u8> {
        let mut hdr: Vec<u8> = Vec::new();
        hdr.write_u32::<LittleEndian>(0x46554747).unwrap(); // "GGUF"
        hdr.write_u32::<LittleEndian>(3).unwrap(); // version
        hdr.write_u64::<LittleEndian>(self.tensors.len() as u64).unwrap();
        hdr.write_u64::<LittleEndian>(self.metadata.len() as u64).unwrap();

        for (k, v) in &self.metadata {
            write_str(&mut hdr, k);
            match v {
                MetaValue::U32(x) => {
                    hdr.write_u32::<LittleEndian>(4).unwrap(); // U32 tag
                    hdr.write_u32::<LittleEndian>(*x).unwrap();
                }
                MetaValue::U64(x) => {
                    hdr.write_u32::<LittleEndian>(10).unwrap(); // U64 tag
                    hdr.write_u64::<LittleEndian>(*x).unwrap();
                }
                MetaValue::String(s) => {
                    hdr.write_u32::<LittleEndian>(8).unwrap(); // String tag
                    write_str(&mut hdr, s);
                }
                MetaValue::StringArray(items) => {
                    hdr.write_u32::<LittleEndian>(9).unwrap(); // Array tag
                    hdr.write_u32::<LittleEndian>(8).unwrap(); // inner String
                    hdr.write_u64::<LittleEndian>(items.len() as u64).unwrap();
                    for s in items {
                        write_str(&mut hdr, s);
                    }
                }
            }
        }

        // Tensor index. Offsets are relative to tensor_data_offset and must be
        // assigned in declaration order (what flambeau's reader preserves).
        let mut cumulative: u64 = 0;
        let mut tensor_offsets: Vec<u64> = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            tensor_offsets.push(cumulative);
            let elems: u64 = t.dims_wire.iter().product();
            let bs = t.dtype.block_size() as u64;
            let ts = t.dtype.type_size() as u64;
            assert_eq!(elems % bs, 0, "elem_count must be multiple of block_size");
            let bytes = elems / bs * ts;
            assert_eq!(t.payload.len() as u64, bytes);
            cumulative += bytes;
        }
        for (t, off) in self.tensors.iter().zip(&tensor_offsets) {
            write_str(&mut hdr, &t.name);
            hdr.write_u32::<LittleEndian>(t.dims_wire.len() as u32).unwrap();
            for d in &t.dims_wire {
                hdr.write_u64::<LittleEndian>(*d).unwrap();
            }
            hdr.write_u32::<LittleEndian>(t.dtype.to_wire()).unwrap();
            hdr.write_u64::<LittleEndian>(*off).unwrap();
        }

        // Pad to alignment.
        let header_end = hdr.len() as u64;
        let tensor_data_offset = header_end.div_ceil(ALIGNMENT) * ALIGNMENT;
        hdr.resize(tensor_data_offset as usize, 0);

        // Payloads.
        for t in &self.tensors {
            hdr.extend_from_slice(&t.payload);
        }
        hdr
    }
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.extend_from_slice(s.as_bytes());
}

/// Wrap a `Vec<u8>` in a type that satisfies `Deref<Target=[u8]>` so we can
/// hand it to `GgufFile::from_mmap` which expects an `Arc<Mmap>`.
/// We don't need a real mmap — the reader only reads the slice.
fn open_bytes(bytes: Vec<u8>) -> GgufFile {
    // Write to a tempfile, mmap it. Simpler than faking `memmap2::Mmap`.
    let tmp = tempdir();
    let p = tmp.join("synthetic.gguf");
    std::fs::write(&p, &bytes).unwrap();
    let file = GgufFile::open(&p).unwrap();
    // Leak tmp for the test duration — test exits shortly, and the mmap
    // inside `file` holds a handle regardless.
    std::mem::forget(tmp);
    file
}

fn tempdir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "flambeau-gguf-test-{}-{}",
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn _unused_arc<T>(_: Arc<T>) {}

// ---- tests ----------------------------------------------------------------

#[test]
fn header_and_tensor_index_round_trip() {
    let f32_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.25).collect();
    let mut payload_f32 = vec![0u8; f32_data.len() * 4];
    for (i, v) in f32_data.iter().enumerate() {
        payload_f32[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    let bytes = Gguf::default()
        .meta("general.architecture", MetaValue::String("qwen3moe".into()))
        .meta("general.alignment", MetaValue::U32(32))
        .meta("qwen3moe.block_count", MetaValue::U32(48))
        .tensor(
            "output_norm.weight",
            GgmlDType::F32,
            &[8], // wire order: fastest-varying first (1D is trivial)
            payload_f32,
        )
        .finish();
    let gguf = open_bytes(bytes);

    assert_eq!(gguf.version, GgufVersion::V3);
    assert_eq!(gguf.architecture(), Some("qwen3moe"));
    assert_eq!(gguf.metadata_u32("qwen3moe.block_count"), Some(48));
    assert_eq!(gguf.tensor_order, vec!["output_norm.weight".to_string()]);
    let info = gguf.info("output_norm.weight").unwrap();
    assert_eq!(info.dtype, GgmlDType::F32);
    assert_eq!(info.dims, vec![8]);
    let deq = gguf.dequantize_tensor("output_norm.weight").unwrap();
    for (i, v) in deq.iter().enumerate() {
        assert_eq!(*v, i as f32 * 0.25);
    }
    assert_eq!(gguf.tensor_data_offset % 32, 0);
}

#[test]
fn two_tensors_offsets_stay_disjoint() {
    // Two F32 tensors of 4 elements each, back-to-back.
    let a: Vec<u8> = (0..4).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let b: Vec<u8> = (0..4).flat_map(|i| ((i + 100) as f32).to_le_bytes()).collect();
    let bytes = Gguf::default()
        .meta("general.architecture", MetaValue::String("test".into()))
        .tensor("a.weight", GgmlDType::F32, &[4], a)
        .tensor("b.weight", GgmlDType::F32, &[4], b)
        .finish();
    let gguf = open_bytes(bytes);
    let a = gguf.dequantize_tensor("a.weight").unwrap();
    let b = gguf.dequantize_tensor("b.weight").unwrap();
    assert_eq!(a, vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(b, vec![100.0, 101.0, 102.0, 103.0]);
}

#[test]
fn row_range_picks_correct_slice_of_2d_f32() {
    // 4 rows × 8 cols of F32 — row-major. GGUF wire order is fastest-varying
    // first so dims on disk are [cols, rows] = [8, 4].
    let cols: u64 = 8;
    let rows: u64 = 4;
    let total = (rows * cols) as usize;
    let mut payload = vec![0u8; total * 4];
    for r in 0..rows {
        for c in 0..cols {
            let i = (r * cols + c) as usize;
            let v = r as f32 * 100.0 + c as f32;
            payload[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    let bytes = Gguf::default()
        .meta("general.architecture", MetaValue::String("test".into()))
        .tensor("w", GgmlDType::F32, &[cols, rows], payload)
        .finish();
    let gguf = open_bytes(bytes);

    // After dims.reverse() the reader exposes dims = [rows, cols] = [4, 8].
    let info = gguf.info("w").unwrap();
    assert_eq!(info.dims, vec![rows, cols]);

    // Read middle two rows (rows 1..=2) — should be [100..108, 200..208].
    let raw = gguf.tensor_row_range_raw("w", 1, 2).unwrap();
    assert_eq!(raw.len(), (2 * cols * 4) as usize);
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expected: Vec<f32> = (1..=2u32)
        .flat_map(|r| (0..cols).map(move |c| r as f32 * 100.0 + c as f32))
        .collect();
    assert_eq!(floats, expected);
}

#[test]
fn expert_range_picks_correct_slice_of_3d_f32() {
    // 3 experts × 2 × 4 F32. Wire dims = [4, 2, 3] (fastest-varying first).
    // Logical dims = [3, 2, 4].
    let e: u64 = 3;
    let d1: u64 = 2;
    let d2: u64 = 4;
    let per_expert = (d1 * d2) as usize;
    let mut payload = vec![0u8; (e as usize) * per_expert * 4];
    for ei in 0..e {
        for i in 0..per_expert {
            let idx = ei as usize * per_expert + i;
            let v = ei as f32 * 1000.0 + i as f32;
            payload[idx * 4..(idx + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    let bytes = Gguf::default()
        .meta("general.architecture", MetaValue::String("test".into()))
        .tensor("experts", GgmlDType::F32, &[d2, d1, e], payload)
        .finish();
    let gguf = open_bytes(bytes);
    assert_eq!(gguf.info("experts").unwrap().dims, vec![e, d1, d2]);

    // Load expert 1 only.
    let raw = gguf.tensor_expert_range_raw("experts", 1, 1).unwrap();
    assert_eq!(raw.len(), per_expert * 4);
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expected: Vec<f32> = (0..per_expert).map(|i| 1000.0 + i as f32).collect();
    assert_eq!(floats, expected);
}

#[test]
fn rejects_non_gguf_magic() {
    let bytes = vec![0u8; 64];
    let tmp = tempdir();
    let p = tmp.join("bad.gguf");
    std::fs::write(&p, &bytes).unwrap();
    let err = GgufFile::open(&p).unwrap_err();
    let s = format!("{err}");
    assert!(s.contains("not a GGUF file"), "got: {s}");
}

#[test]
fn unknown_tensor_reports_clearly() {
    let bytes = Gguf::default()
        .meta("general.architecture", MetaValue::String("test".into()))
        .tensor("a", GgmlDType::F32, &[1], vec![0, 0, 0, 0])
        .finish();
    let gguf = open_bytes(bytes);
    let err = gguf.info("missing").unwrap_err();
    assert!(format!("{err}").contains("missing"));
}

#[test]
fn metadata_string_array_round_trips() {
    let bytes = Gguf::default()
        .meta("general.architecture", MetaValue::String("test".into()))
        .meta(
            "tokenizer.ggml.tokens",
            MetaValue::StringArray(vec!["<|a|>".into(), "<|b|>".into(), "<|c|>".into()]),
        )
        .tensor("a", GgmlDType::F32, &[1], vec![0, 0, 0, 0])
        .finish();
    let gguf = open_bytes(bytes);
    let arr = gguf
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(Value::as_array)
        .unwrap();
    let strs: Vec<&str> = arr.iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(strs, vec!["<|a|>", "<|b|>", "<|c|>"]);
}
