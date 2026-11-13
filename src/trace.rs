use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::sync::mpsc;

const GPU_CATS: &[&str] = &["kernel", "gpu_memcpy", "gpu_memset"];
const ANNOTATION_CAT: &str = "gpu_user_annotation";
const KEEP_CATS_BYTES: &[&[u8]] = &[
    b"kernel",
    b"gpu_memcpy",
    b"gpu_memset",
    b"gpu_user_annotation",
];
const CPU_OP_CAT: &str = "cpu_op";
const RUNTIME_CATS: &[&str] = &["cuda_runtime", "hip_runtime", "cuda_driver", "hip_driver"];

#[derive(Debug, Clone, Deserialize)]
pub struct RawEvent {
    pub name: Option<String>,
    pub cat: Option<String>,
    pub ph: Option<String>,
    pub ts: Option<f64>,
    pub dur: Option<f64>,
    pub pid: Option<serde_json::Value>,
    pub tid: Option<serde_json::Value>,
    pub args: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelEvent {
    pub name: String,
    pub cat: String,
    pub ts: f64,
    pub dur: f64,
    pub device: u64,
    pub stream: u64,
    pub grid: Option<String>,
    pub block: Option<String>,
    pub shared_memory: Option<u64>,
    pub registers_per_thread: Option<u64>,
    pub correlation: Option<u64>,
    /// Kineto `External id`: links the kernel to the CPU operator that launched it.
    pub external_id: Option<u64>,
    #[serde(default)]
    pub trace_id: usize,
}

impl KernelEvent {
    pub fn end_ts(&self) -> f64 {
        self.ts + self.dur
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AnnotationEvent {
    pub name: String,
    pub ts: f64,
    pub dur: f64,
    pub stream: u64,
    #[serde(default)]
    pub trace_id: usize,
}

impl AnnotationEvent {
    pub fn end_ts(&self) -> f64 {
        self.ts + self.dur
    }
}

#[derive(Debug, Clone, Default)]
pub struct Trace {
    pub kernels: Vec<KernelEvent>,
    pub annotations: Vec<AnnotationEvent>,
}

/// A CPU operator (`cat: cpu_op`). Input shapes, dtypes and FLOPs are only
/// present when the trace was captured with `record_shapes` / `with_flops`.
#[derive(Debug, Clone, Default)]
pub struct CpuOp {
    pub name: String,
    /// `Input Dims` as compact JSON, e.g. `[[16,4096],[4096,4096]]`.
    pub input_dims: Option<String>,
    /// `Input type` as compact JSON, e.g. `["c10::BFloat16","c10::BFloat16"]`.
    pub input_types: Option<String>,
    pub flops: Option<String>,
}

/// The CPU side needed to find the operator that launched a kernel. Kineto
/// links a kernel to its operator through `External id`, or through
/// `correlation` to the runtime launch event that carries the `External id`.
#[derive(Debug, Clone, Default)]
pub struct OpIndex {
    /// `External id` -> CPU operator.
    pub ops: HashMap<u64, CpuOp>,
    /// `correlation` -> `External id` of the runtime launch event.
    pub launches: HashMap<u64, u64>,
}

impl OpIndex {
    /// The CPU operator that launched `k`, if the trace links one.
    pub fn op_for(&self, k: &KernelEvent) -> Option<&CpuOp> {
        k.external_id
            .and_then(|ext| self.ops.get(&ext))
            .or_else(|| {
                let ext = self.launches.get(&k.correlation?)?;
                self.ops.get(ext)
            })
    }
}

pub fn parse_trace(path: &str) -> Result<Trace> {
    parse_trace_file(path, false).map(|(trace, _)| trace)
}

/// Like [`parse_trace`], but also keeps the CPU operators and runtime launch
/// events needed to join kernels with their launching operator. Slower and
/// larger than [`parse_trace`], so the TUI does not use it.
pub fn parse_trace_with_ops(path: &str) -> Result<(Trace, OpIndex)> {
    parse_trace_file(path, true)
}

fn parse_trace_file(path: &str, with_ops: bool) -> Result<(Trace, OpIndex)> {
    let file = File::open(path).with_context(|| format!("Cannot open trace file: {}", path))?;
    let total = file.metadata().map(|m| m.len()).unwrap_or(0);
    let progress = ProgressReader::new(file, total);

    let reader = if path.ends_with(".gz") {
        ThreadedReader::new(GzDecoder::new(progress))
    } else {
        ThreadedReader::new(progress)
    };
    let (mut trace, ops) = parse_lexer(reader, with_ops).context("Failed to parse trace")?;

    progress_finish(total);

    trace
        .kernels
        .sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal));
    trace
        .annotations
        .sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal));
    Ok((trace, ops))
}

/// Reads `inner` on a background thread in large blocks, so that reading and
/// gzip decompression overlap with lexing on the calling thread.
struct ThreadedReader {
    rx: mpsc::Receiver<io::Result<Vec<u8>>>,
    block: Vec<u8>,
    pos: usize,
}

impl ThreadedReader {
    const BLOCK: usize = 1 << 20;

    fn new<R: Read + Send + 'static>(mut inner: R) -> Self {
        let (tx, rx) = mpsc::sync_channel(4);
        std::thread::spawn(move || loop {
            let mut block = vec![0u8; Self::BLOCK];
            let mut len = 0;
            let res = loop {
                match inner.read(&mut block[len..]) {
                    Ok(0) => break Ok(()),
                    Ok(n) => {
                        len += n;
                        if len == block.len() {
                            break Ok(());
                        }
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => break Err(e),
                }
            };
            block.truncate(len);
            let eof = len < Self::BLOCK;
            // A send error means the lexer stopped early; nothing left to do.
            if len > 0 && tx.send(Ok(block)).is_err() {
                break;
            }
            if let Err(e) = res {
                let _ = tx.send(Err(e));
                break;
            }
            if eof {
                break;
            }
        });
        Self {
            rx,
            block: Vec::new(),
            pos: 0,
        }
    }
}

impl Read for ThreadedReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let avail = self.fill_buf()?;
        let n = avail.len().min(out.len());
        out[..n].copy_from_slice(&avail[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl BufRead for ThreadedReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.pos >= self.block.len() {
            self.pos = 0;
            self.block = match self.rx.recv() {
                Ok(block) => block?,
                // The reader thread is done: end of input.
                Err(_) => Vec::new(),
            };
        }
        Ok(&self.block[self.pos..])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = (self.pos + amt).min(self.block.len());
    }
}

/// Wraps the trace file and renders a byte-progress bar to stderr as bytes are
/// read. For `.gz` traces this tracks compressed bytes (the actual I/O), which
/// still advances monotonically to 100%.
struct ProgressReader<R: Read> {
    inner: R,
    read: u64,
    total: u64,
    last_render_pct: u64,
}

impl<R: Read> ProgressReader<R> {
    fn new(inner: R, total: u64) -> Self {
        Self {
            inner,
            read: 0,
            total,
            last_render_pct: u64::MAX,
        }
    }

    fn render(&mut self) {
        if self.total == 0 {
            return;
        }
        let pct = (self.read.saturating_mul(100) / self.total).min(100);
        if pct == self.last_render_pct {
            return;
        }
        self.last_render_pct = pct;
        let filled = (pct as usize * PROGRESS_WIDTH) / 100;
        let bar: String = "█".repeat(filled) + &"░".repeat(PROGRESS_WIDTH - filled);
        eprint!("\rParsing trace [{}] {:>3}%", bar, pct);
        let _ = io::Write::flush(&mut io::stderr());
    }
}

const PROGRESS_WIDTH: usize = 40;

fn progress_finish(total: u64) {
    if total > 0 {
        let bar = "█".repeat(PROGRESS_WIDTH);
        eprintln!("\rParsing trace [{}] 100%", bar);
    }
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.read += n as u64;
        self.render();
        Ok(n)
    }
}

fn parse_lexer<R: BufRead>(mut reader: R, with_ops: bool) -> Result<(Trace, OpIndex)> {
    const CHUNK: usize = 1 << 16;

    let first = peek_first_nonws(&mut reader)?;

    match first {
        b'[' => lex_array(reader, with_ops),
        b'{' => lex_object(reader, CHUNK, with_ops),
        other => anyhow::bail!(
            "Unexpected top-level JSON character '{}' (0x{:02x})",
            other as char,
            other
        ),
    }
}

fn lex_object<R: BufRead>(reader: R, chunk: usize, with_ops: bool) -> Result<(Trace, OpIndex)> {
    let te_marker = b"traceEvents";
    let mut window: Vec<u8> = Vec::with_capacity(chunk * 2);
    let mut rdr = reader;
    let mut found = false;

    'scan: loop {
        let buf = rdr.fill_buf().context("read error")?;
        if buf.is_empty() {
            break;
        }
        let n = buf.len().min(chunk);
        window.extend_from_slice(&buf[..n]);
        rdr.consume(n);

        if let Some(pos) = window.windows(te_marker.len()).position(|w| w == te_marker) {
            let after = pos + te_marker.len();
            if let Some(off) = window[after..].iter().position(|&b| b == b'[') {
                let remainder = window[after + off + 1..].to_vec();
                window = remainder;
                found = true;
                break 'scan;
            }
        }
        if window.len() > te_marker.len() + 64 {
            let keep = window.len() - te_marker.len() - 16;
            window.drain(..keep);
        }
    }

    if !found {
        return Ok((Trace::default(), OpIndex::default()));
    }

    lex_events_array(rdr, window, with_ops)
}

fn lex_array<R: BufRead>(reader: R, with_ops: bool) -> Result<(Trace, OpIndex)> {
    let window = Vec::new();
    lex_events_array(reader, window, with_ops)
}

fn lex_events_array<R: BufRead>(
    mut rdr: R,
    window: Vec<u8>,
    with_ops: bool,
) -> Result<(Trace, OpIndex)> {
    let mut lexer = EventLexer::new(with_ops);
    lexer.feed(&window);
    loop {
        let buf = rdr.fill_buf().context("read error")?;
        if buf.is_empty() {
            break;
        }
        let n = buf.len();
        lexer.feed(buf);
        rdr.consume(n);
    }
    Ok((lexer.trace, lexer.op_index))
}

/// Bytes of an event kept while its `cat` is still unknown.
const MAX_HEADER: usize = 512;

#[derive(PartialEq, Clone, Copy)]
enum LexState {
    Unknown,
    Keep,
    Skip,
}

/// Splits the `traceEvents` array into top-level events and parses those whose
/// `cat` is kept. Only quotes, backslashes and braces change the lexer state, so
/// it jumps between them with `memchr` and copies the plain bytes in between as
/// whole spans; skipped events, most of a trace, are never copied.
struct EventLexer {
    with_ops: bool,
    depth: u32,
    in_string: bool,
    escape_next: bool,
    state: LexState,
    /// Start of the current event, until its `cat` decides whether to keep it.
    header: Vec<u8>,
    /// The current event, once it is kept.
    event_buf: Vec<u8>,
    trace: Trace,
    op_index: OpIndex,
}

impl EventLexer {
    fn new(with_ops: bool) -> Self {
        Self {
            with_ops,
            depth: 0,
            in_string: false,
            escape_next: false,
            state: LexState::Unknown,
            header: Vec::with_capacity(MAX_HEADER),
            event_buf: Vec::with_capacity(1024),
            trace: Trace {
                kernels: Vec::with_capacity(65536),
                annotations: Vec::new(),
            },
            op_index: OpIndex::default(),
        }
    }

    fn feed(&mut self, mut buf: &[u8]) {
        while !buf.is_empty() {
            if self.escape_next {
                self.escape_next = false;
                self.push(&buf[..1]);
                buf = &buf[1..];
                continue;
            }
            if self.in_string {
                let Some(i) = memchr::memchr2(b'"', b'\\', buf) else {
                    self.push(buf);
                    return;
                };
                self.push(&buf[..=i]);
                if buf[i] == b'\\' {
                    self.escape_next = true;
                } else {
                    self.in_string = false;
                }
                buf = &buf[i + 1..];
                continue;
            }
            let Some(i) = memchr::memchr3(b'"', b'{', b'}', buf) else {
                self.push(buf);
                return;
            };
            self.push(&buf[..i]);
            // Decide before a brace can close the event.
            self.check_cat();
            match buf[i] {
                b'"' => {
                    self.in_string = true;
                    self.push(&buf[i..=i]);
                }
                b'{' => {
                    self.depth += 1;
                    if self.depth == 1 {
                        self.state = LexState::Unknown;
                        self.header.clear();
                        self.event_buf.clear();
                    }
                    self.push(&buf[i..=i]);
                }
                _ => {
                    self.push(&buf[i..=i]);
                    if self.depth == 1 {
                        self.finish_event();
                    } else if self.depth > 0 {
                        self.depth -= 1;
                    }
                }
            }
            buf = &buf[i + 1..];
        }
    }

    /// Appends plain bytes to the buffer of the current state.
    fn push(&mut self, bytes: &[u8]) {
        match self.state {
            LexState::Keep => self.event_buf.extend_from_slice(bytes),
            LexState::Unknown => {
                let room = MAX_HEADER.saturating_sub(self.header.len());
                self.header
                    .extend_from_slice(&bytes[..bytes.len().min(room)]);
            }
            LexState::Skip => {}
        }
    }

    /// Keeps or skips the current event once its `cat` has been read.
    fn check_cat(&mut self) {
        if self.state != LexState::Unknown || self.depth != 1 {
            return;
        }
        match cat_in_header(&self.header, self.with_ops) {
            Some(true) => {
                self.state = LexState::Keep;
                self.event_buf.extend_from_slice(&self.header);
                self.header.clear();
            }
            Some(false) => {
                self.state = LexState::Skip;
                self.header.clear();
            }
            None => {}
        }
    }

    fn finish_event(&mut self) {
        if self.state == LexState::Keep && !self.event_buf.is_empty() {
            if let Ok(raw) = serde_json::from_slice::<RawEvent>(&self.event_buf) {
                match classify(raw) {
                    Some(Parsed::Kernel(k)) => self.trace.kernels.push(k),
                    Some(Parsed::Annotation(a)) => self.trace.annotations.push(a),
                    Some(Parsed::Op(ext, op)) => {
                        self.op_index.ops.insert(ext, op);
                    }
                    Some(Parsed::Launch(correlation, ext)) => {
                        self.op_index.launches.insert(correlation, ext);
                    }
                    None => {}
                }
            }
        }
        self.state = LexState::Unknown;
        self.header.clear();
        self.event_buf.clear();
        self.depth = 0;
    }
}

/// Whether the event whose first bytes are `buf` should be kept, judged by its
/// `cat`; `None` until the `cat` value has been read. CPU operators and runtime
/// launches are kept only `with_ops`.
fn cat_in_header(buf: &[u8], with_ops: bool) -> Option<bool> {
    let marker = b"\"cat\"";
    let pos = buf.windows(marker.len()).position(|w| w == marker)?;
    let after = &buf[pos + marker.len()..];
    let mut i = 0;
    while i < after.len() && matches!(after[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    if i >= after.len() || after[i] != b':' {
        return None;
    }
    i += 1;
    while i < after.len() && matches!(after[i], b' ' | b'\t') {
        i += 1;
    }
    if i >= after.len() || after[i] != b'"' {
        return None;
    }
    i += 1;
    let val_start = i;
    while i < after.len() && after[i] != b'"' {
        i += 1;
    }
    if i >= after.len() {
        return None;
    }
    let val = &after[val_start..i];
    let is_op_cat =
        || val == CPU_OP_CAT.as_bytes() || RUNTIME_CATS.iter().any(|c| c.as_bytes() == val);
    Some(KEEP_CATS_BYTES.contains(&val) || (with_ops && is_op_cat()))
}

fn peek_first_nonws<R: BufRead>(reader: &mut R) -> Result<u8> {
    loop {
        let buf = reader.fill_buf().context("Failed to read trace")?;
        if buf.is_empty() {
            anyhow::bail!("Trace file is empty");
        }
        for (i, &b) in buf.iter().enumerate() {
            if !b.is_ascii_whitespace() {
                reader.consume(i);
                return Ok(b);
            }
        }
        let len = buf.len();
        reader.consume(len);
    }
}

enum Parsed {
    Kernel(KernelEvent),
    Annotation(AnnotationEvent),
    /// A CPU operator and its `External id`.
    Op(u64, CpuOp),
    /// A runtime launch: `correlation` and `External id`.
    Launch(u64, u64),
}

fn classify(e: RawEvent) -> Option<Parsed> {
    let ph = e.ph.as_deref().unwrap_or("");
    if ph == "M" || ph == "i" || ph == "I" {
        return None;
    }
    let cat = e.cat.as_deref().unwrap_or("");

    if cat == CPU_OP_CAT {
        let ext = args_u64(&e.args, "External id")?;
        let json = |key: &str| e.args.as_ref()?.get(key).map(|v| v.to_string());
        let flops = args_string(&e.args, "flops");
        return Some(Parsed::Op(
            ext,
            CpuOp {
                input_dims: json("Input Dims"),
                input_types: json("Input type"),
                flops,
                name: e.name.unwrap_or_default(),
            },
        ));
    }
    if RUNTIME_CATS.contains(&cat) {
        let correlation = args_u64(&e.args, "correlation")?;
        let ext = args_u64(&e.args, "External id")?;
        return Some(Parsed::Launch(correlation, ext));
    }

    let ts = e.ts?;
    let dur = e.dur.unwrap_or(0.0);
    let stream = tid_to_u64(&e.tid).unwrap_or(0);

    if cat == ANNOTATION_CAT {
        let name = e.name.unwrap_or_else(|| "(unnamed)".to_string());
        return Some(Parsed::Annotation(AnnotationEvent {
            name,
            ts,
            dur,
            stream,
            trace_id: 0,
        }));
    }

    if !GPU_CATS.contains(&cat) {
        return None;
    }
    let name = e.name.unwrap_or_else(|| "(unnamed)".to_string());
    let device = args_u64(&e.args, "device")
        .or_else(|| tid_to_u64(&e.pid))
        .unwrap_or(0);
    let grid = args_string(&e.args, "grid");
    let block = args_string(&e.args, "block");
    let shared_memory = args_u64(&e.args, "shared memory");
    let registers_per_thread = args_u64(&e.args, "registers per thread");
    let correlation = args_u64(&e.args, "correlation");
    let external_id = args_u64(&e.args, "External id");

    Some(Parsed::Kernel(KernelEvent {
        name,
        cat: cat.to_string(),
        ts,
        dur,
        device,
        stream,
        grid,
        block,
        shared_memory,
        registers_per_thread,
        correlation,
        external_id,
        trace_id: 0,
    }))
}

fn tid_to_u64(v: &Option<serde_json::Value>) -> Option<u64> {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_u64().or_else(|| n.as_f64().map(|f| f as u64)),
        Some(serde_json::Value::String(s)) => s.parse().ok(),
        _ => None,
    }
}

fn args_u64(args: &Option<HashMap<String, serde_json::Value>>, key: &str) -> Option<u64> {
    let map = args.as_ref()?;
    match map.get(key)? {
        serde_json::Value::Number(n) => n.as_u64().or_else(|| n.as_f64().map(|f| f as u64)),
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn args_string(args: &Option<HashMap<String, serde_json::Value>>, key: &str) -> Option<String> {
    let map = args.as_ref()?;
    match map.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    fn make_raw(cat: &str, ph: &str, ts: f64, dur: f64, tid: u64) -> RawEvent {
        RawEvent {
            name: Some("test_kernel".to_string()),
            cat: Some(cat.to_string()),
            ph: Some(ph.to_string()),
            ts: Some(ts),
            dur: Some(dur),
            pid: Some(serde_json::Value::Number(0.into())),
            tid: Some(serde_json::Value::Number(tid.into())),
            args: None,
        }
    }

    fn classify_kernel(e: RawEvent) -> Option<KernelEvent> {
        match classify(e) {
            Some(Parsed::Kernel(k)) => Some(k),
            _ => None,
        }
    }

    #[test]
    fn test_gpu_kernel_accepted() {
        let e = make_raw("kernel", "X", 1000.0, 50.0, 7);
        let k = classify_kernel(e).expect("Should parse GPU kernel");
        assert_eq!(k.name, "test_kernel");
        assert_eq!(k.stream, 7);
        assert!((k.ts - 1000.0).abs() < f64::EPSILON);
        assert!((k.dur - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cpu_op_rejected() {
        let e = make_raw("cpu_op", "X", 1000.0, 50.0, 0);
        assert!(classify_kernel(e).is_none());
    }

    #[test]
    fn test_metadata_rejected() {
        let e = make_raw("kernel", "M", 0.0, 0.0, 0);
        assert!(classify(e).is_none());
    }

    #[test]
    fn test_gpu_memcpy_accepted() {
        let e = make_raw("gpu_memcpy", "X", 2000.0, 10.0, 3);
        let k = classify_kernel(e).expect("Should parse gpu_memcpy");
        assert_eq!(k.cat, "gpu_memcpy");
        assert_eq!(k.stream, 3);
    }

    #[test]
    fn test_annotation_accepted() {
        let e = make_raw("gpu_user_annotation", "X", 500.0, 1000.0, 4);
        let a = match classify(e) {
            Some(Parsed::Annotation(a)) => a,
            _ => panic!("Should parse annotation"),
        };
        assert_eq!(a.stream, 4);
        assert!((a.ts - 500.0).abs() < f64::EPSILON);
        assert!((a.dur - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_object_wrapper() {
        let json = r#"{
            "traceEvents": [
                {"name":"cpu_fn","cat":"cpu_op","ph":"X","ts":1.0,"dur":5.0,"pid":1,"tid":1},
                {"name":"my_kernel","cat":"kernel","ph":"X","ts":10.0,"dur":3.0,"pid":0,"tid":7,
                 "args":{"device":0,"stream":7,"grid":"[32,1,1]","block":"[128,1,1]",
                         "shared memory":0,"registers per thread":32,"correlation":99}},
                {"name":"my_ctx","cat":"gpu_user_annotation","ph":"X","ts":8.0,"dur":20.0,"pid":0,"tid":7},
                {"name":"process_name","ph":"M","pid":0,"tid":0,"args":{"name":"GPU 0"}}
            ],
            "displayTimeUnit": "ms"
        }"#;
        let reader = BufReader::new(json.as_bytes());
        let (trace, _) = parse_lexer(reader, false).expect("Should parse object wrapper");
        assert_eq!(trace.kernels.len(), 1);
        assert_eq!(trace.kernels[0].name, "my_kernel");
        assert_eq!(trace.kernels[0].stream, 7);
        assert_eq!(trace.kernels[0].grid.as_deref(), Some("[32,1,1]"));
        assert_eq!(trace.annotations.len(), 1);
        assert_eq!(trace.annotations[0].name, "my_ctx");
        assert_eq!(trace.annotations[0].stream, 7);
    }

    #[test]
    fn test_parse_bare_array() {
        let json = r#"[
            {"name":"cpu_op","cat":"cpu_op","ph":"X","ts":1.0,"dur":5.0,"pid":1,"tid":1},
            {"name":"kern_a","cat":"kernel","ph":"X","ts":5.0,"dur":2.0,"pid":0,"tid":3},
            {"name":"kern_b","cat":"gpu_memcpy","ph":"X","ts":8.0,"dur":1.0,"pid":0,"tid":3}
        ]"#;
        let reader = BufReader::new(json.as_bytes());
        let (trace, _) = parse_lexer(reader, false).expect("Should parse bare array");
        assert_eq!(trace.kernels.len(), 2);
        assert_eq!(trace.kernels[0].name, "kern_a");
        assert_eq!(trace.kernels[1].name, "kern_b");
    }

    #[test]
    fn test_parse_gz_trace() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let json = r#"{"traceEvents":[
            {"name":"cpu_fn","cat":"cpu_op","ph":"X","ts":1.0,"dur":5.0,"pid":1,"tid":1},
            {"name":"gz_kernel","cat":"kernel","ph":"X","ts":10.0,"dur":3.0,"pid":0,"tid":7}
        ]}"#;

        let path = "/tmp/test_trace_rust.json.gz";
        {
            let f = File::create(path).unwrap();
            let mut gz = GzEncoder::new(f, Compression::default());
            gz.write_all(json.as_bytes()).unwrap();
            gz.finish().unwrap();
        }

        let trace = parse_trace(path).expect("Should parse .json.gz");
        assert_eq!(trace.kernels.len(), 1);
        assert_eq!(trace.kernels[0].name, "gz_kernel");
    }

    /// A kernel linked to its operator through `External id`, one linked only
    /// through `correlation` to its runtime launch, and one graph-replayed
    /// kernel with no operator.
    const OPS_TRACE: &str = r#"{"traceEvents":[
        {"ph":"X","cat":"cpu_op","name":"aten::mm","pid":1,"tid":1,"ts":1.0,"dur":5.0,
         "args":{"External id":7,"Input Dims":[[16, 64], [64, 32]],
                 "Input type":["c10::BFloat16", "c10::BFloat16"],"flops":32768}},
        {"ph":"X","cat":"cpu_op","name":"vllm::rms_norm","pid":1,"tid":1,"ts":2.0,"dur":5.0,
         "args":{"External id":8}},
        {"ph":"X","cat":"cuda_runtime","name":"cudaLaunchKernel","pid":1,"tid":1,"ts":3.0,"dur":1.0,
         "args":{"External id":8,"correlation":42}},
        {"ph":"X","cat":"python_function","name":"forward {\"x\": }","pid":1,"tid":1,"ts":0.5,"dur":9.0,
         "args":{"Python id":3}},
        {"ph":"X","cat":"kernel","name":"gemm","pid":0,"tid":7,"ts":10.0,"dur":3.0,
         "args":{"External id":7,"correlation":41}},
        {"ph":"X","cat":"kernel","name":"norm","pid":0,"tid":7,"ts":14.0,"dur":2.0,
         "args":{"correlation":42}},
        {"ph":"X","cat":"kernel","name":"replayed","pid":0,"tid":7,"ts":17.0,"dur":1.0,
         "args":{"correlation":99}}
    ]}"#;

    #[test]
    fn test_parse_with_ops_links_kernels_to_ops() {
        let (trace, ops) = parse_lexer(BufReader::new(OPS_TRACE.as_bytes()), true).unwrap();
        assert_eq!(trace.kernels.len(), 3);
        assert_eq!(ops.ops.len(), 2);
        assert_eq!(ops.launches.get(&42), Some(&8));

        let gemm = ops
            .op_for(&trace.kernels[0])
            .expect("linked by External id");
        assert_eq!(gemm.name, "aten::mm");
        assert_eq!(gemm.input_dims.as_deref(), Some("[[16,64],[64,32]]"));
        assert_eq!(
            gemm.input_types.as_deref(),
            Some(r#"["c10::BFloat16","c10::BFloat16"]"#)
        );
        assert_eq!(gemm.flops.as_deref(), Some("32768"));

        let norm = ops
            .op_for(&trace.kernels[1])
            .expect("linked by correlation");
        assert_eq!(norm.name, "vllm::rms_norm");
        assert_eq!(norm.input_dims, None);

        assert!(ops.op_for(&trace.kernels[2]).is_none());
    }

    #[test]
    fn test_parse_without_ops_skips_cpu_events() {
        let (trace, ops) = parse_lexer(BufReader::new(OPS_TRACE.as_bytes()), false).unwrap();
        assert_eq!(trace.kernels.len(), 3);
        assert!(ops.ops.is_empty());
        assert!(ops.launches.is_empty());
    }

    #[test]
    fn test_lexer_is_independent_of_chunk_boundaries() {
        let json = r#"[
            {"name":"skip \"{\" }","cat":"python_function","ph":"X","ts":1.0,"args":{"a":{"b":"}"}}},
            {"name":"k \\ \"q\" {x}","cat":"kernel","ph":"X","ts":5.0,"dur":2.0,"tid":3,
             "args":{"grid":"[1, 1, 1]"}},
            {"name":"m","cat":"gpu_memcpy","ph":"X","ts":8.0,"dur":1.0,"tid":3}
        ]"#;
        let whole = {
            let mut lexer = EventLexer::new(false);
            lexer.feed(json.as_bytes());
            lexer.trace
        };
        let bytewise = {
            let mut lexer = EventLexer::new(false);
            for b in json.as_bytes().chunks(1) {
                lexer.feed(b);
            }
            lexer.trace
        };
        for trace in [&whole, &bytewise] {
            let names: Vec<&str> = trace.kernels.iter().map(|k| k.name.as_str()).collect();
            assert_eq!(names, vec![r#"k \ "q" {x}"#, "m"]);
            assert_eq!(trace.kernels[0].grid.as_deref(), Some("[1, 1, 1]"));
        }
    }

    #[test]
    fn test_threaded_reader_streams_all_bytes() {
        let data: Vec<u8> = (0..(ThreadedReader::BLOCK * 2 + 123))
            .map(|i| (i % 251) as u8)
            .collect();
        let mut out = Vec::new();
        ThreadedReader::new(io::Cursor::new(data.clone()))
            .read_to_end(&mut out)
            .unwrap();
        assert_eq!(out, data);
    }
}
