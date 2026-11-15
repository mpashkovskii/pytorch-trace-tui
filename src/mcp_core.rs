//! Pure, bounded trace-analysis functions shared by the TUI and the MCP server.
//!
//! Every list-returning function enforces `limit` (rows returned) and `offset`
//! (rows skipped) so callers — especially an LLM over MCP — cannot pull an
//! unbounded dump from an 800MB / multi-million-kernel trace into their context.

use crate::app::{csv_escape, stage_for_annotation_name};
use crate::trace::{
    parse_trace_with_ops, AnnotationEvent, CpuAnnotationEvent, CpuOp, KernelEvent, OpIndex, Trace,
};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// The vLLM iteration stage a kernel executes under, derived from the covering
/// `gpu_user_annotation`. `None` means the kernel has no stage-bearing
/// annotation (e.g. covered only by an nccl annotation, or none at all).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Prefill,
    Decode,
    Mixed,
    None,
}

impl Stage {
    fn from_ann_str(s: &str) -> Stage {
        match s {
            "prefill" => Stage::Prefill,
            "decode" => Stage::Decode,
            "mixed" => Stage::Mixed,
            _ => Stage::None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
            Stage::Mixed => "mixed",
            Stage::None => "none",
        }
    }

    /// Parses an MCP filter argument. Only the three real stages are accepted;
    /// `"none"` is rejected because "no filter" is expressed as `Option::None`,
    /// not a filter that selects unannotated kernels.
    pub fn parse_arg(s: &str) -> Result<Stage, String> {
        match s {
            "prefill" => Ok(Stage::Prefill),
            "decode" => Ok(Stage::Decode),
            "mixed" => Ok(Stage::Mixed),
            other => Err(format!(
                "invalid stage {other:?}: expected one of prefill, decode, mixed"
            )),
        }
    }
}

/// Hard ceiling on rows any single tool call may return, regardless of the
/// requested `limit`. Keeps MCP responses bounded even on a caller mistake.
pub const MAX_ROWS: usize = 1000;

/// Default page size when a caller omits `limit`.
pub const DEFAULT_LIMIT: usize = 100;

fn effective_limit(limit: Option<usize>) -> usize {
    limit.unwrap_or(DEFAULT_LIMIT).min(MAX_ROWS)
}

/// Recursively scans `dir` for PyTorch profiler traces (`*.pt.trace.json.gz`)
/// and returns their paths relative to `dir`, sorted for determinism. The user
/// workflow nests traces at `logs/<run>/traces/*.pt.trace.json.gz`, so a flat
/// scan misses them; this walks the tree. Returns an empty vec (not an error)
/// when none are found. The returned relative paths are directly reusable as the
/// `path` argument of the other MCP tools.
pub fn list_traces_in_dir(dir: &str) -> Result<Vec<String>> {
    let root = Path::new(dir);
    let mut hits: Vec<PathBuf> = Vec::new();
    walk_traces(root, 0, &mut hits);
    let mut rel: Vec<String> = hits
        .iter()
        .map(|p| {
            p.strip_prefix(root)
                .unwrap_or(p)
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    rel.sort();
    Ok(rel)
}

/// Bounded depth-first walk collecting `*.pt.trace.json.gz` files. Skips
/// symlinks (so an ancestor-pointing symlink can't cycle), hidden directories
/// (names starting with `.`), and unreadable entries; caps recursion at
/// `MAX_DEPTH`.
fn walk_traces(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    const MAX_DEPTH: usize = 8;
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_symlink() {
            continue;
        }
        let path = entry.path();
        if ft.is_dir() {
            let hidden = entry
                .file_name()
                .to_str()
                .map(|n| n.starts_with('.'))
                .unwrap_or(true);
            if hidden {
                continue;
            }
            walk_traces(&path, depth + 1, out);
        } else if entry
            .file_name()
            .to_str()
            .map(|n| n.ends_with(".pt.trace.json.gz"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

/// Compact display label for a trace, disambiguating traces that share a
/// filename across runs. The user's layout is `logs/<run>/traces/<file>`, where
/// `<file>` (e.g. `dp0_..._rank7....pt.trace.json.gz`) repeats every run — so
/// the label combines the run folder (the segment before `traces/`, else the
/// immediate parent) with the rank (`rankN` parsed from the filename, else the
/// filename stem). A flat trace with no meaningful parent falls back to its stem.
pub fn trace_display_label(path: &str) -> String {
    let p = Path::new(path);
    let file = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let stem = file
        .strip_suffix(".pt.trace.json.gz")
        .or_else(|| file.strip_suffix(".json.gz"))
        .or_else(|| file.strip_suffix(".json"))
        .unwrap_or(file);

    let run = run_folder(p);
    let leaf = rank_of(stem).unwrap_or_else(|| stem.to_string());

    match run {
        Some(r) if !leaf.is_empty() => format!("{r}/{leaf}"),
        _ if !stem.is_empty() => stem.to_string(),
        _ => "trace".to_string(),
    }
}

/// The run folder for a trace path: the component immediately before `traces/`,
/// else the immediate parent directory name (unless it is `.`).
fn run_folder(p: &Path) -> Option<String> {
    let comps: Vec<&str> = p
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    if let Some(idx) = comps.iter().rposition(|&c| c == "traces") {
        if idx > 0 {
            return Some(comps[idx - 1].to_string());
        }
    }
    p.parent()
        .and_then(|par| par.file_name())
        .and_then(|n| n.to_str())
        .filter(|n| *n != ".")
        .map(|n| n.to_string())
}

/// Extracts a `rankN` token from a filename stem: the literal `rank` followed by
/// one or more digits (stopping at the first non-digit).
fn rank_of(stem: &str) -> Option<String> {
    let idx = stem.find("rank")?;
    let digits: String = stem[idx + 4..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        Some(format!("rank{digits}"))
    }
}

/// Resolves the CUDA stream for a stage query: the explicit one, else the
/// stream with the most total kernel time (the "main" stream in practice).
/// Returns `(stream, auto)` where `auto` marks that the default was used.
fn resolve_stream(trace: &Trace, stream: Option<u64>) -> Result<(u64, bool), String> {
    match stream {
        Some(s) => Ok((s, false)),
        None => default_stream(trace)
            .map(|s| (s, true))
            .ok_or_else(|| "trace has no kernels".to_string()),
    }
}

/// The stream with the most total kernel time; `None` when the trace has no
/// kernels. Ties break toward the lowest stream id for determinism.
pub fn default_stream(trace: &Trace) -> Option<u64> {
    let mut sums: BTreeMap<u64, f64> = BTreeMap::new();
    for k in &trace.kernels {
        *sums.entry(k.stream).or_default() += k.dur;
    }
    let mut best: Option<(u64, f64)> = None;
    for (stream, total) in sums {
        match best {
            Some((_, t)) if t >= total => {}
            _ => best = Some((stream, total)),
        }
    }
    best.map(|(s, _)| s)
}

/// `stream=4` when explicit, `stream=4(auto)` when auto-chosen. Every stage
/// tool output header carries this so the caller can see which stream was used.
fn stream_note(stream: u64, auto: bool) -> String {
    if auto {
        format!("stream={stream}(auto)")
    } else {
        format!("stream={stream}")
    }
}

/// How the kernels of a step were attributed to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Membership {
    /// By the CPU-side launch: the kernel's runtime launch event (`correlation`
    /// -> cuda_runtime/hip_runtime/driver) has its `ts` inside the step's
    /// CPU-side `user_annotation` (same process). Graph-replayed kernels are
    /// attributed through the graph launch's runtime event the same way. This
    /// is what vLLM actually executed in the step; the GPU-side annotation can
    /// cover only a small fraction of an eager step's kernels because launches
    /// outrun the GPU.
    Launch,
    /// Fallback: the kernel's `ts` sits inside the step's GPU-side
    /// (`gpu_user_annotation`) window on the stream. Used when the trace has no
    /// CPU-side step annotations for the stage or no runtime events.
    GpuWindow,
}

impl Membership {
    pub fn as_str(self) -> &'static str {
        match self {
            Membership::Launch => "launch",
            Membership::GpuWindow => "gpu_window",
        }
    }
}

/// Whether a kernel belongs to the step proper or was launched after the
/// step's annotation closed: vLLM launches the sampler and the LM head after
/// the step's CPU-side annotation ends, so those kernels belong to no step
/// unless they are attributed as post-step work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    /// Launched under the step's annotation (or inside its GPU window).
    Step,
    /// Launched after the nearest earlier step's annotation ended and before
    /// any further step annotation started (same pid), up to the end of the
    /// trace for the last step.
    Post,
}

impl Part {
    pub fn as_str(self) -> &'static str {
        match self {
            Part::Step => "step",
            Part::Post => "post",
        }
    }
}

/// One execution step of a stage: the kernels attributed to it on the queried
/// stream, in ts order, and the step's reported span. In `Launch` mode the span
/// is the GPU span of the step's kernels (first kernel start to last kernel
/// end); in `GpuWindow` mode it is the GPU-side annotation's window (post-step
/// kernels do not extend it).
struct Step<'k> {
    name: String,
    ts: f64,
    dur: f64,
    /// Attributed kernels with their step [`Part`].
    kernels: Vec<(&'k KernelEvent, Part)>,
}

impl Step<'_> {
    fn post_kernels(&self) -> usize {
        self.kernels
            .iter()
            .filter(|(_, p)| *p == Part::Post)
            .count()
    }
}

/// The execution steps of a stage on a stream, ordered by step start. See
/// [`Membership`] for how kernels are attributed. `GpuWindow` is the fallback
/// when the stage has no CPU-side step annotations or the trace holds no
/// runtime events.
fn build_steps<'k>(
    trace: &'k Trace,
    ops: &OpIndex,
    stage: Stage,
    stream: u64,
) -> (Membership, Vec<Step<'k>>) {
    let stage_of = |a: &&CpuAnnotationEvent| Stage::from_ann_str(stage_for_annotation_name(&a.name));
    let all_steps: Vec<&CpuAnnotationEvent> = ops
        .cpu_annotations
        .iter()
        .filter(|a| !matches!(stage_of(a), Stage::None))
        .collect();
    let cpu_steps: Vec<&CpuAnnotationEvent> = all_steps
        .iter()
        .copied()
        .filter(|a| stage_of(a) == stage)
        .collect();
    if !cpu_steps.is_empty() && !ops.launches.is_empty() {
        (
            Membership::Launch,
            steps_by_launch(trace, ops, stream, &cpu_steps, &all_steps),
        )
    } else {
        (Membership::GpuWindow, steps_by_window(trace, stage, stream))
    }
}

/// Launch-based attribution: for every kernel on `stream`, resolve its runtime
/// launch event and attribute it to the step annotation of the launch's
/// process whose window contains the launch's `ts` — even when the kernel
/// itself runs after that annotation ended (the GPU is behind the CPU in eager
/// steps). A launch after the nearest earlier step's end and before any later
/// step's start (across all stages: the prefill/decode/mixed decisions never
/// double-count a kernel) is attributed to that step as post-step work, which
/// is where vLLM's sampler and LM head land. Kernels launched before the
/// first step stay outside every step. Steps get the GPU span of their
/// kernels; a step with no kernels on the stream keeps its CPU annotation
/// start with zero duration.
fn steps_by_launch<'k>(
    trace: &'k Trace,
    ops: &OpIndex,
    stream: u64,
    cpu_steps: &[&CpuAnnotationEvent],
    all_steps: &[&CpuAnnotationEvent],
) -> Vec<Step<'k>> {
    let mut steps: Vec<Step> = cpu_steps
        .iter()
        .map(|a| Step {
            name: a.name.clone(),
            ts: a.ts,
            dur: 0.0,
            kernels: Vec::new(),
        })
        .collect();
    // Map a stage step annotation to its index in `steps`, by pointer identity
    // (`cpu_steps` is a subset of `all_steps`).
    let stage_idx: HashMap<usize, usize> = cpu_steps
        .iter()
        .enumerate()
        .map(|(i, &a)| (a as *const CpuAnnotationEvent as usize, i))
        .collect();
    // All-stage step annotations per CPU process, ts-ordered at parse time:
    // the container test must consider other stages' annotations so a launch
    // inside another stage's step is not stolen as this stage's post-step.
    let mut by_pid: HashMap<u64, Vec<&CpuAnnotationEvent>> = HashMap::new();
    for &a in all_steps {
        by_pid.entry(a.pid).or_default().push(a);
    }
    let key = |a: &CpuAnnotationEvent| a as *const CpuAnnotationEvent as usize;
    for k in trace.kernels.iter().filter(|k| k.stream == stream) {
        let Some(launch) = ops.launch_for(k) else {
            continue;
        };
        let Some(cands) = by_pid.get(&launch.pid) else {
            continue;
        };
        // Step annotations of a process do not overlap, so the nearest
        // earlier annotation decides.
        let i = cands.partition_point(|a| a.ts <= launch.ts);
        let Some(&nearest) = cands[..i].last() else {
            continue; // launched before the first step
        };
        let part = if nearest.end_ts() > launch.ts {
            Part::Step
        } else {
            Part::Post
        };
        if let Some(&ci) = stage_idx.get(&key(nearest)) {
            steps[ci].kernels.push((k, part));
        }
    }
    for s in &mut steps {
        if s.kernels.is_empty() {
            continue;
        }
        let start = s.kernels[0].0.ts;
        let end = s
            .kernels
            .iter()
            .fold(f64::MIN, |e, (k, _)| e.max(k.end_ts()));
        s.ts = start;
        s.dur = end - start;
    }
    steps
}

/// GPU-window attribution (the fallback): the stage's `gpu_user_annotation`
/// instances on the stream, each with the kernels starting inside its window.
/// The same post-step rule as [`steps_by_launch`] applies: kernels starting
/// after a window closes and before the next step window of any stage opens
/// on the stream are attributed to the closed step as post-step work.
fn steps_by_window<'k>(trace: &'k Trace, stage: Stage, stream: u64) -> Vec<Step<'k>> {
    let stage_of =
        |a: &&AnnotationEvent| Stage::from_ann_str(stage_for_annotation_name(&a.name));
    let all_windows: Vec<&AnnotationEvent> = trace
        .annotations
        .iter()
        .filter(|a| a.stream == stream)
        .filter(|a| !matches!(stage_of(a), Stage::None))
        .collect();
    let mut steps: Vec<Step> = Vec::new();
    for (i, ann) in all_windows.iter().enumerate() {
        if stage_of(ann) != stage {
            continue;
        }
        let lo = trace.kernels.partition_point(|k| k.ts < ann.ts);
        let hi = trace.kernels.partition_point(|k| k.ts < ann.end_ts());
        let mut kernels: Vec<(&KernelEvent, Part)> = trace.kernels[lo..hi]
            .iter()
            .filter(|k| k.stream == stream)
            .map(|k| (k, Part::Step))
            .collect();
        // Post-step kernels: after this window closes, before the next step
        // window of any stage opens (end of the trace for the last one).
        let next = all_windows.get(i + 1).map(|n| n.ts).unwrap_or(f64::MAX);
        let phi = trace.kernels.partition_point(|k| k.ts < next);
        kernels.extend(
            trace.kernels[hi..phi]
                .iter()
                .filter(|k| k.stream == stream)
                .map(|k| (k, Part::Post)),
        );
        steps.push(Step {
            name: ann.name.clone(),
            ts: ann.ts,
            dur: ann.dur,
            kernels,
        });
    }
    steps
}

/// The token counters of a vLLM step annotation. `ctx_tokens`/`gen_tokens`
/// are the scheduled query tokens of the context (prefill) and generation
/// (decode) side; the three detail pairs exist only in the detailed
/// annotation format (`detailed_trace_annotation`):
/// - `*_kv_tokens` (sk): sum over the batch's requests of the attended
///   sequence length — the KV volume attention reads;
/// - `*_qk` (sqsk): sum of query_len × seq_len — proportional to attention
///   FLOPs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StepTokens {
    pub ctx_tokens: u64,
    pub gen_tokens: u64,
    pub ctx_kv_tokens: Option<u64>,
    pub gen_kv_tokens: Option<u64>,
    pub ctx_qk: Option<u64>,
    pub gen_qk: Option<u64>,
}

/// Parses the token counts of a vLLM step annotation. Simple format:
/// `execute_context_<ctx_req>(<ctx_toks>)_generation_<gen_req>(<gen_toks>)`,
/// optionally prefixed with a total: `execute_<total>_context_...`. Detailed
/// format: `execute_<total>_context_<ctx_req>(sq<ctx_toks>sk<ctx_seq_len_sum>sqsq<ctx_qq>sqsk<ctx_qk>)_generation_<gen_req>(sq...)`.
/// Returns `None` when the name is not a step annotation.
pub(crate) fn step_token_counts(name: &str) -> Option<StepTokens> {
    let stripped = name.strip_prefix("execute_")?;
    let rest = match stripped.strip_prefix("context_") {
        Some(r) => r,
        None => {
            let (total, r) = stripped.split_once("_context_")?;
            if !total.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            r
        }
    };
    // rest = <ctx_req>(<ctx_toks>)_generation_<gen_req>(<gen_toks>)
    let p1 = rest.find('(')?;
    let e1 = p1 + 1 + rest[p1 + 1..].find(')')?;
    let marker_pos = rest[e1..].find("_generation_")? + e1;
    let after_gen = &rest[marker_pos + "_generation_".len()..];
    let p2 = after_gen.find('(')?;
    let e2 = p2 + 1 + after_gen[p2 + 1..].find(')')?;
    let (ctx_tokens, ctx_detail) = token_fields(&rest[p1 + 1..e1])?;
    let (gen_tokens, gen_detail) = token_fields(&after_gen[p2 + 1..e2])?;
    let (ctx_kv_tokens, ctx_qk) = ctx_detail.map_or((None, None), |(sk, qk)| (Some(sk), Some(qk)));
    let (gen_kv_tokens, gen_qk) = gen_detail.map_or((None, None), |(sk, qk)| (Some(sk), Some(qk)));
    Some(StepTokens {
        ctx_tokens,
        gen_tokens,
        ctx_kv_tokens,
        gen_kv_tokens,
        ctx_qk,
        gen_qk,
    })
}

/// Parses one step component's parenthesised counters: either a plain
/// query-token count (simple format) or the detailed
/// `sq<query_tokens>sk<seq_len_sum>sqsq<query_sq_sum>sqsk<qk>` form. Returns
/// the query-token count (sq) plus `(sk, sqsk)` when present.
fn token_fields(content: &str) -> Option<(u64, Option<(u64, u64)>)> {
    if let Ok(n) = content.parse::<u64>() {
        return Some((n, None));
    }
    let (sq, mut rest) = take_u64(content.strip_prefix("sq")?)?;
    let mut sk = None;
    let mut qk = None;
    while !rest.is_empty() {
        // sqsq (sum of query_len^2) is parsed but not reported.
        if let Some(r) = rest.strip_prefix("sqsq") {
            (_, rest) = take_u64(r)?;
        } else if let Some(r) = rest.strip_prefix("sqsk") {
            let (v, r2) = take_u64(r)?;
            qk = Some(v);
            rest = r2;
        } else if let Some(r) = rest.strip_prefix("sk") {
            let (v, r2) = take_u64(r)?;
            sk = Some(v);
            rest = r2;
        } else {
            return None;
        }
    }
    let detail = match (sk, qk) {
        (Some(sk), Some(qk)) => Some((sk, qk)),
        _ => None,
    };
    Some((sq, detail))
}

/// Reads a leading run of ASCII digits as a u64 and returns it with the
/// remainder; `None` when there are no digits.
fn take_u64(s: &str) -> Option<(u64, &str)> {
    let len = s.bytes().take_while(u8::is_ascii_digit).count();
    if len == 0 {
        return None;
    }
    Some((s[..len].parse().ok()?, &s[len..]))
}

/// How a kernel was launched, for the `launch` column of the per-kernel
/// outputs: `op` when linked to a CPU operator, `graph` when replayed from a
/// CUDA/HIP graph launch, empty when the trace links it to neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Launch {
    Op,
    Graph,
    Unknown,
}

impl Launch {
    fn as_str(self) -> &'static str {
        match self {
            Launch::Op => "op",
            Launch::Graph => "graph",
            Launch::Unknown => "",
        }
    }
}

/// Classifies a kernel's launch and returns its CPU operator when linked.
fn launch_of<'a>(ops: &'a OpIndex, k: &KernelEvent) -> (Launch, Option<&'a CpuOp>) {
    if let Some(op) = ops.op_for(k) {
        return (Launch::Op, Some(op));
    }
    if ops.launch_for(k).map(|l| l.graph).unwrap_or(false) {
        return (Launch::Graph, None);
    }
    (Launch::Unknown, None)
}

/// The `launch op input_dims input_types flops` cells of one kernel, each TSV
/// -cleaned and empty when the trace carries no such link or attribute.
fn shape_cells(ops: &OpIndex, k: &KernelEvent) -> (Launch, String, String, String, String) {
    let (launch, op) = launch_of(ops, k);
    (
        launch,
        op.map(|o| tsv_clean(&o.name)).unwrap_or_default(),
        op.and_then(|o| o.input_dims.as_deref())
            .map(tsv_clean)
            .unwrap_or_default(),
        op.and_then(|o| o.input_types.as_deref())
            .map(tsv_clean)
            .unwrap_or_default(),
        op.and_then(|o| o.flops.as_deref())
            .map(tsv_clean)
            .unwrap_or_default(),
    )
}

/// TSV of the stage's execution steps on a stream: one row per step with the
/// annotation name, start and duration of the step's GPU span (µs; in
/// `gpu_window` membership mode the GPU annotation's window instead), the
/// token counters parsed from the annotation name, and its kernel counts:
/// `kernels` launched under the step's annotation, `post_kernels` launched
/// after it closed and before the next step started (vLLM's sampler and LM
/// head). `ctx_tokens`/`gen_tokens` are scheduled query tokens; on traces
/// captured with `detailed_trace_annotation` the paired
/// `ctx_kv_tokens`/`gen_kv_tokens` (sum of attended sequence lengths) and
/// `ctx_qk`/`gen_qk` (sum of query_len × seq_len, proportional to attention
/// FLOPs) are populated too, and empty otherwise. The header line carries the
/// membership mode and the total step count; page with `offset`/`limit`.
pub fn stage_steps_tsv(
    trace: &Trace,
    ops: &OpIndex,
    stage: Stage,
    stream: Option<u64>,
    offset: usize,
    limit: Option<usize>,
) -> Result<String, String> {
    let (stream, auto) = resolve_stream(trace, stream)?;
    let (membership, steps) = build_steps(trace, ops, stage, stream);
    let cap = effective_limit(limit);
    let mut out = format!(
        "# stage_steps stage={} {} membership={} total={}\n",
        stage.as_str(),
        stream_note(stream, auto),
        membership.as_str(),
        steps.len()
    );
    out.push_str(
        "idx\tname\tts_us\tdur_us\tctx_tokens\tgen_tokens\tctx_kv_tokens\tgen_kv_tokens\tctx_qk\tgen_qk\tkernels\tpost_kernels\n",
    );
    for (idx, step) in steps.iter().enumerate().skip(offset).take(cap) {
        let tokens = step_token_counts(&step.name);
        let (ctx_toks, gen_toks) = tokens
            .map(|t| (t.ctx_tokens, t.gen_tokens))
            .unwrap_or((0, 0));
        let opt = |o: Option<u64>| o.map(|v| v.to_string()).unwrap_or_default();
        out.push_str(&format!(
            "{}\t{}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            idx,
            step.name,
            step.ts,
            step.dur,
            ctx_toks,
            gen_toks,
            opt(tokens.and_then(|t| t.ctx_kv_tokens)),
            opt(tokens.and_then(|t| t.gen_kv_tokens)),
            opt(tokens.and_then(|t| t.ctx_qk)),
            opt(tokens.and_then(|t| t.gen_qk)),
            step.kernels.len() - step.post_kernels(),
            step.post_kernels(),
        ));
    }
    Ok(out)
}

/// TSV of the ordered kernel sequence inside one execution step (`step` is the
/// `idx` returned by [`stage_steps_tsv`]): position, kernel name,
/// `part` (`step` for kernels launched under the step's annotation, `post`
/// for kernels launched after it closed and before the next step started —
/// vLLM's sampler and LM head land here), start offset from the step start,
/// duration, gap to the previous kernel's end (µs), grid, block, the launch
/// kind (`op` for a CPU-operator launch, `graph` for a CUDA/HIP graph replay,
/// empty when unlinked) and, for operator-launched kernels, the launching
/// operator, its input dims, input types and FLOPs per call (present only in
/// traces captured with `record_shapes` / `with_flops`). The first kernel's
/// gap is measured from the step start. Header lines carry the step name,
/// span, total kernel count and the count of post-step kernels; page with
/// `offset`/`limit`. Errors when `step` is out of range.
pub fn step_kernels_tsv(
    trace: &Trace,
    ops: &OpIndex,
    stage: Stage,
    stream: Option<u64>,
    step: usize,
    offset: usize,
    limit: Option<usize>,
) -> Result<String, String> {
    let (stream, auto) = resolve_stream(trace, stream)?;
    let (membership, steps) = build_steps(trace, ops, stage, stream);
    let Some(s) = steps.get(step) else {
        return Err(format!(
            "step {step} out of range: stage {} has {} step(s) on {}",
            stage.as_str(),
            steps.len(),
            stream_note(stream, auto)
        ));
    };
    let kernels = &s.kernels;
    let cap = effective_limit(limit);
    let mut out = format!(
        "# step_kernels stage={} {} membership={} step={} step_name={} step_ts_us={:.2} step_dur_us={:.2} total={} post={}\n",
        stage.as_str(),
        stream_note(stream, auto),
        membership.as_str(),
        step,
        s.name,
        s.ts,
        s.dur,
        kernels.len(),
        s.post_kernels(),
    );
    out.push_str("pos\tname\tpart\tstart_offset_us\tdur_us\tgap_us\tgrid\tblock\tlaunch\top\tinput_dims\tinput_types\tflops\n");
    let mut prev_end: Option<f64> = None;
    for (pos, (k, part)) in kernels.iter().enumerate() {
        let gap = match prev_end {
            None => k.ts - s.ts,
            Some(prev) => k.ts - prev,
        };
        prev_end = Some(k.end_ts());
        if pos < offset || pos >= offset + cap {
            continue;
        }
        let (launch, op, dims, types, flops) = shape_cells(ops, k);
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            pos,
            tsv_clean(&k.name),
            part.as_str(),
            k.ts - s.ts,
            k.dur,
            gap,
            tsv_clean(k.grid.as_deref().unwrap_or("")),
            tsv_clean(k.block.as_deref().unwrap_or("")),
            launch.as_str(),
            op,
            dims,
            types,
            flops,
        ));
    }
    Ok(out)
}

/// Kernel names contain commas (shape lists), never tabs; TSV output only needs
/// tabs/newlines scrubbed so a row stays one line.
fn tsv_clean(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

/// A repeating kernel-name sub-block detected inside a step: the `len`-kernel
/// block starting at `start` repeats `count` times, i.e. covers positions
/// `[start, start + len*count)`. Typically the per-layer block.
#[derive(Debug, Clone, Copy)]
struct RepeatBlock {
    start: usize,
    len: usize,
    count: usize,
}

/// One pattern: a group of steps sharing the exact same ordered kernel-name
/// sequence, plus the best repeating sub-block detected in that sequence.
struct Pattern {
    /// Step indices (into the stage's step list), ordered by step time.
    steps: Vec<usize>,
    /// The shared kernel-name sequence.
    names: Vec<String>,
    /// Per-step total kernel duration, aligned with `steps`.
    total_durs: Vec<f64>,
    repeat: Option<RepeatBlock>,
}

/// Groups a stage's steps on a stream by their ordered kernel-name sequence.
/// Patterns are ranked by summed kernel time across their steps (descending),
/// ties broken by the earliest step index, so the first pattern is the one that
/// dominates the stage. Also returns the membership mode and the steps, for
/// per-position shape detail.
fn build_patterns<'k>(
    trace: &'k Trace,
    ops: &OpIndex,
    stage: Stage,
    stream: u64,
) -> (Membership, Vec<Step<'k>>, Vec<Pattern>) {
    let (membership, steps) = build_steps(trace, ops, stage, stream);
    let mut index: HashMap<Vec<String>, usize> = HashMap::new();
    let mut patterns: Vec<Pattern> = Vec::new();
    for (step_idx, step) in steps.iter().enumerate() {
        let names: Vec<String> = step
            .kernels
            .iter()
            .map(|(k, _)| k.name.clone())
            .collect();
        let total_dur: f64 = step.kernels.iter().map(|(k, _)| k.dur).sum();
        match index.get(&names) {
            Some(&pi) => {
                patterns[pi].steps.push(step_idx);
                patterns[pi].total_durs.push(total_dur);
            }
            None => {
                index.insert(names.clone(), patterns.len());
                patterns.push(Pattern {
                    steps: vec![step_idx],
                    names,
                    total_durs: vec![total_dur],
                    repeat: None,
                });
            }
        }
    }
    for p in &mut patterns {
        let ids = name_ids(&p.names);
        p.repeat = detect_repeat_block(&ids);
    }
    patterns.sort_by(|a, b| {
        let ta: f64 = a.total_durs.iter().sum();
        let tb: f64 = b.total_durs.iter().sum();
        tb.total_cmp(&ta)
            .then_with(|| a.steps[0].cmp(&b.steps[0]))
    });
    (membership, steps, patterns)
}

/// Maps each distinct name to an id, so repeat detection compares u32s rather
/// than (often long, utf8-heavy) kernel-name strings.
fn name_ids(names: &[String]) -> Vec<u32> {
    let mut ids = Vec::with_capacity(names.len());
    let mut map: HashMap<&str, u32> = HashMap::new();
    for n in names {
        let next = map.len() as u32;
        ids.push(*map.entry(n.as_str()).or_insert(next));
    }
    ids
}

/// Finds the repeating sub-block covering the most positions of `ids`: for
/// every period, the runs where `ids[i] == ids[i+period]`; the longest covering
/// run with `count >= 2` wins (ties already favor the smaller period, then the
/// earlier start, thanks to the iteration order). `None` when nothing repeats.
fn detect_repeat_block(ids: &[u32]) -> Option<RepeatBlock> {
    let l = ids.len();
    let mut best: Option<RepeatBlock> = None;
    let mut best_covered = 0usize;
    for period in 1..=l / 2 {
        let mut i = 0;
        while i + 2 * period <= l {
            if ids[i] != ids[i + period] {
                i += 1;
                continue;
            }
            let start = i;
            let mut run = 0;
            while i + period < l && ids[i] == ids[i + period] {
                run += 1;
                i += 1;
            }
            i += 1;
            let count = run / period + 1;
            let covered = count * period;
            // Covered improves, ties keep the earlier (smaller period/earlier start).
            if covered >= 2 * period && covered > best_covered {
                best_covered = covered;
                best = Some(RepeatBlock {
                    start,
                    len: period,
                    count,
                });
            }
        }
    }
    best
}

/// Median of `durs` (sorted in place); even counts average the two middle
/// values. Empty input yields 0.
fn median_us(durs: &mut Vec<f64>) -> f64 {
    if durs.is_empty() {
        return 0.0;
    }
    durs.sort_by(|a, b| a.total_cmp(b));
    let n = durs.len();
    if n.is_multiple_of(2) {
        (durs[n / 2 - 1] + durs[n / 2]) / 2.0
    } else {
        durs[n / 2]
    }
}

/// TSV listing the stage's kernel-sequence patterns: one row per pattern with
/// its step count, sequence length, median per-step total duration, the
/// detected repeating block (`start..end xcount`, empty when none), and the
/// first step indices. Ranked by summed kernel time across the pattern's
/// steps. The header line carries the total pattern count; page with
/// `offset`/`limit`.
pub fn stage_patterns_tsv(
    trace: &Trace,
    ops: &OpIndex,
    stage: Stage,
    stream: Option<u64>,
    offset: usize,
    limit: Option<usize>,
) -> Result<String, String> {
    let (stream, auto) = resolve_stream(trace, stream)?;
    let (membership, _steps, patterns) = build_patterns(trace, ops, stage, stream);
    let cap = effective_limit(limit);
    let mut out = format!(
        "# stage_patterns stage={} {} membership={} total={}\n",
        stage.as_str(),
        stream_note(stream, auto),
        membership.as_str(),
        patterns.len()
    );
    out.push_str("pattern\tsteps\tkernels_per_step\tmedian_total_dur_us\trepeat\tstep_indices\n");
    for (idx, p) in patterns.iter().enumerate().skip(offset).take(cap) {
        let repeat = p
            .repeat
            .map(|r| format!("{}..{}x{}", r.start, r.start + r.len * r.count - 1, r.count))
            .unwrap_or_default();
        let shown: Vec<String> = p.steps.iter().take(8).map(|i| i.to_string()).collect();
        let mut step_indices = shown.join(",");
        if p.steps.len() > shown.len() {
            step_indices.push_str(",...");
        }
        let median_total = median_us(&mut p.total_durs.clone());
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.2}\t{}\t{}\n",
            idx,
            p.steps.len(),
            p.names.len(),
            median_total,
            repeat,
            step_indices,
        ));
    }
    Ok(out)
}

/// One rendered row of the pattern detail: a sequence position with the median
/// duration of the kernel at that position across the pattern's steps (for the
/// repeat block: across steps × repeats), plus the most common launch kind,
/// operator and input shape at that position, and whether the shape varies
/// across the samples.
struct DetailRow {
    pos: usize,
    name: String,
    part: &'static str,
    median_dur: f64,
    samples: usize,
    launch: &'static str,
    op: String,
    input_dims: String,
    input_types: String,
    flops: String,
    /// The input dims/types are not identical across the position's samples.
    shape_varies: bool,
}

/// The most common `(launch, op, dims, types, flops)` among `kernels` (ties
/// break lexicographically for determinism) plus whether more than one
/// `(dims, types)` pair occurs.
fn mode_shape(
    kernels: &[&KernelEvent],
    ops: &OpIndex,
) -> (&'static str, String, String, String, String, bool) {
    #[derive(PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct Key<'a>(Launch, &'a str, &'a str, &'a str, &'a str);
    let mut counts: HashMap<Key, usize> = HashMap::new();
    let mut shapes: std::collections::HashSet<(&str, &str)> = std::collections::HashSet::new();
    for k in kernels {
        let (launch, op) = launch_of(ops, k);
        let key = Key(
            launch,
            op.map(|o| o.name.as_str()).unwrap_or(""),
            op.and_then(|o| o.input_dims.as_deref()).unwrap_or(""),
            op.and_then(|o| o.input_types.as_deref()).unwrap_or(""),
            op.and_then(|o| o.flops.as_deref()).unwrap_or(""),
        );
        shapes.insert((key.2, key.3));
        *counts.entry(key).or_default() += 1;
    }
    let best = counts
        .into_iter()
        .max_by(|(ka, ca), (kb, cb)| ca.cmp(cb).then_with(|| kb.cmp(ka)));
    let varies = shapes.len() > 1;
    match best {
        Some((Key(launch, op, dims, types, flops), _)) => (
            launch.as_str(),
            tsv_clean(op),
            tsv_clean(dims),
            tsv_clean(types),
            tsv_clean(flops),
            varies,
        ),
        None => ("", String::new(), String::new(), String::new(), String::new(), false),
    }
}

/// TSV detail of one pattern: every position of the shared kernel sequence
/// once, with the repeating per-layer block collapsed to a single copy
/// (`part=repeat`), so one layer's kernels can be read without paging through
/// every layer. Columns: `pos` (position in the step's kernel sequence;
/// positions covered by the repeat beyond its first occurrence are omitted),
/// `name`, `part` (`head`/`repeat`/`tail`), `median_dur_us`, `samples` (number
/// of step(-repeat) instances the median is over), then the most common
/// launch/operator/input shape at that position: `launch` (`op`, `graph` or
/// empty), `op`, `input_dims`, `input_types`, `flops`, and `shape_varies`
/// (`*` when the dims/types differ across the position's samples, e.g. a
/// varying batch size between steps). A `# repeat_block` header line reports
/// the block bounds and its median total duration. Errors when `pattern` is
/// out of range.
pub fn stage_pattern_detail_tsv(
    trace: &Trace,
    ops: &OpIndex,
    stage: Stage,
    stream: Option<u64>,
    pattern: usize,
    offset: usize,
    limit: Option<usize>,
) -> Result<String, String> {
    let (stream, auto) = resolve_stream(trace, stream)?;
    let (membership, steps, patterns) = build_patterns(trace, ops, stage, stream);
    let Some(p) = patterns.get(pattern) else {
        return Err(format!(
            "pattern {pattern} out of range: stage {} has {} pattern(s) on {}",
            stage.as_str(),
            patterns.len(),
            stream_note(stream, auto)
        ));
    };

    // Per-step kernel sequences; every step of the pattern has exactly
    // `names.len()` kernels.
    let windows: Vec<&Vec<(&KernelEvent, Part)>> =
        p.steps.iter().map(|&s| &steps[s].kernels).collect();
    let n_steps = p.steps.len();
    let n_kernels = p.names.len();

    let kernel_at = |pos: usize, w: usize| -> Option<&KernelEvent> {
        windows.get(w).and_then(|ks| ks.get(pos)).map(|(k, _)| *k)
    };
    let dur_at = |pos: usize, w: usize| -> f64 { kernel_at(pos, w).map(|k| k.dur).unwrap_or(0.0) };
    let shape_at = |positions: &mut dyn Iterator<Item = (usize, usize)>| {
        let kernels: Vec<&KernelEvent> = positions.filter_map(|(pos, w)| kernel_at(pos, w)).collect();
        mode_shape(&kernels, ops)
    };

    let mut rows: Vec<DetailRow> = Vec::new();
    let mut repeat_line = String::new();
    let push_head_or_tail = |rows: &mut Vec<DetailRow>, pos: usize, part: &'static str| {
        let mut durs: Vec<f64> = (0..n_steps).map(|w| dur_at(pos, w)).collect();
        let (launch, op, dims, types, flops, shape_varies) =
            shape_at(&mut (0..n_steps).map(|w| (pos, w)));
        rows.push(DetailRow {
            pos,
            name: p.names[pos].clone(),
            part,
            median_dur: median_us(&mut durs),
            samples: n_steps,
            launch,
            op,
            input_dims: dims,
            input_types: types,
            flops,
            shape_varies,
        });
    };

    match p.repeat {
        Some(r) if r.count >= 2 => {
            let head_end = r.start.min(n_kernels);
            let tail_start = (r.start + r.len * r.count).min(n_kernels);
            for pos in 0..head_end {
                push_head_or_tail(&mut rows, pos, "head");
            }
            let mut block_totals: Vec<f64> = Vec::new();
            for w in 0..n_steps {
                for rep in 0..r.count {
                    let mut total = 0.0;
                    for b in 0..r.len {
                        let pos = r.start + rep * r.len + b;
                        if pos < n_kernels {
                            total += dur_at(pos, w);
                        }
                    }
                    block_totals.push(total);
                }
            }
            let median_block = median_us(&mut block_totals);
            repeat_line = format!(
                "# repeat_block start={} len={} count={} median_block_dur_us={:.2}\n",
                r.start, r.len, r.count, median_block
            );
            for b in 0..r.len {
                let mut durs: Vec<f64> = Vec::new();
                for w in 0..n_steps {
                    for rep in 0..r.count {
                        let pos = r.start + rep * r.len + b;
                        if pos < n_kernels {
                            durs.push(dur_at(pos, w));
                        }
                    }
                }
                let samples = durs.len();
                let (launch, op, dims, types, flops, shape_varies) = shape_at(&mut (0..n_steps)
                    .flat_map(|w| (0..r.count).map(move |rep| (r.start + rep * r.len + b, w))));
                rows.push(DetailRow {
                    pos: r.start + b,
                    name: p.names[r.start + b].clone(),
                    part: "repeat",
                    median_dur: median_us(&mut durs),
                    samples,
                    launch,
                    op,
                    input_dims: dims,
                    input_types: types,
                    flops,
                    shape_varies,
                });
            }
            for pos in tail_start..n_kernels {
                push_head_or_tail(&mut rows, pos, "tail");
            }
        }
        _ => {
            for pos in 0..n_kernels {
                push_head_or_tail(&mut rows, pos, "head");
            }
        }
    }

    let cap = effective_limit(limit);
    let mut out = format!(
        "# stage_pattern stage={} {} membership={} pattern={} steps={} kernels_per_step={} total={}\n",
        stage.as_str(),
        stream_note(stream, auto),
        membership.as_str(),
        pattern,
        n_steps,
        n_kernels,
        rows.len()
    );
    out.push_str(&repeat_line);
    out.push_str("pos\tname\tpart\tmedian_dur_us\tsamples\tlaunch\top\tinput_dims\tinput_types\tflops\tshape_varies\n");
    for row in rows.iter().skip(offset).take(cap) {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.pos,
            tsv_clean(&row.name),
            row.part,
            row.median_dur,
            row.samples,
            row.launch,
            row.op,
            row.input_dims,
            row.input_types,
            row.flops,
            if row.shape_varies { "*" } else { "" },
        ));
    }
    Ok(out)
}

/// One row of the kernel-to-operator join: a kernel name, the CPU operator that
/// launched it, and that operator's input shapes, dtypes and FLOPs, with the
/// GPU time of every call that shares them.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeRow {
    pub kernel: String,
    pub op: String,
    pub input_dims: String,
    pub input_types: String,
    pub flops: String,
    pub calls: usize,
    pub total_dur: f64,
}

impl ShapeRow {
    /// Everything but the timing: breaks ties in the ranking deterministically.
    fn key(&self) -> (&str, &str, &str, &str, &str) {
        (
            &self.kernel,
            &self.op,
            &self.input_dims,
            &self.input_types,
            &self.flops,
        )
    }
}

/// Kernel shapes of a trace, ranked by total GPU time.
#[derive(Debug, Clone)]
pub struct KernelShapes {
    pub rows: Vec<ShapeRow>,
    /// Kernels matching the name filter.
    pub kernels: usize,
    /// Kernels without a launching operator, e.g. replayed from a CUDA/HIP graph.
    pub unlinked: usize,
}

/// Joins every kernel whose name contains `name_filter` with the CPU operator
/// that launched it, and groups them by kernel name, operator and input shapes:
/// the same kernel at different shapes (decode vs prefill batch sizes) sits at
/// a different point of the roofline. Rows are sorted by total GPU time.
pub fn kernel_shapes(trace: &Trace, ops: &OpIndex, name_filter: &str) -> KernelShapes {
    type Key<'a> = (&'a str, &'a str, &'a str, &'a str, &'a str);
    let mut groups: HashMap<Key, (usize, f64)> = HashMap::new();
    let mut kernels = 0;
    let mut unlinked = 0;
    for k in trace
        .kernels
        .iter()
        .filter(|k| k.name.contains(name_filter))
    {
        kernels += 1;
        let op = ops.op_for(k);
        if op.is_none() {
            unlinked += 1;
        }
        let key = match op {
            Some(op) => (
                k.name.as_str(),
                op.name.as_str(),
                op.input_dims.as_deref().unwrap_or(""),
                op.input_types.as_deref().unwrap_or(""),
                op.flops.as_deref().unwrap_or(""),
            ),
            None => (k.name.as_str(), "", "", "", ""),
        };
        let g = groups.entry(key).or_default();
        g.0 += 1;
        g.1 += k.dur;
    }
    let mut rows: Vec<ShapeRow> = groups
        .into_iter()
        .map(
            |((kernel, op, dims, types, flops), (calls, total_dur))| ShapeRow {
                kernel: kernel.to_string(),
                op: op.to_string(),
                input_dims: dims.to_string(),
                input_types: types.to_string(),
                flops: flops.to_string(),
                calls,
                total_dur,
            },
        )
        .collect();
    rows.sort_by(|a, b| {
        b.total_dur
            .total_cmp(&a.total_dur)
            .then_with(|| a.key().cmp(&b.key()))
    });
    KernelShapes {
        rows,
        kernels,
        unlinked,
    }
}

/// CSV of kernel shape rows: `kernel, op, calls, total_us, mean_us, input_dims,
/// input_types, flops_per_call`.
pub fn kernel_shapes_csv(rows: &[ShapeRow]) -> String {
    let mut out =
        String::from("kernel,op,calls,total_us,mean_us,input_dims,input_types,flops_per_call\n");
    for r in rows {
        out.push_str(&format!(
            "{},{},{},{:.1},{:.2},{},{},{}\n",
            csv_escape(&r.kernel),
            csv_escape(&r.op),
            r.calls,
            r.total_dur,
            r.total_dur / r.calls as f64,
            csv_escape(&r.input_dims),
            csv_escape(&r.input_types),
            csv_escape(&r.flops),
        ));
    }
    out
}

/// The operator label of an aggregated stage-shapes row: graph-replayed
/// kernels are aggregated as `(graph)` (they have no CPU operator but their
/// time share must stay visible), unlinked kernels as the empty string.
const GRAPH_OP: &str = "(graph)";

/// One aggregated row of [`stage_shapes_tsv`]: a kernel name × launching
/// operator × input dims × input types, with the GPU time of every call in the
/// stage that shares them and its share of the stage's total kernel time.
#[derive(Debug, Clone, PartialEq)]
pub struct StageShapeRow {
    pub kernel: String,
    /// Operator name, `(graph)` for graph-replayed kernels, "" when unlinked.
    pub op: String,
    pub input_dims: String,
    pub input_types: String,
    /// FLOPs per call, as PyTorch counted them on the launching operator.
    pub flops: String,
    pub calls: usize,
    pub total_us: f64,
    /// This row's share of the stage's total kernel time on the stream, %.
    pub share_pct: f64,
}

impl StageShapeRow {
    fn key(&self) -> (&str, &str, &str, &str, &str) {
        (
            &self.kernel,
            &self.op,
            &self.input_dims,
            &self.input_types,
            &self.flops,
        )
    }
}

/// Aggregated kernel shapes of one stage, ranked by GPU time in the stage.
#[derive(Debug, Clone)]
pub struct StageShapes {
    pub rows: Vec<StageShapeRow>,
    /// Stage kernels matching the name filter.
    pub kernels: usize,
    /// Of those, post-step kernels (launched after the step's annotation
    /// closed; see [`Part::Post`]).
    pub post_kernels: usize,
    /// Of those, kernels with input shapes on their launching operator.
    pub with_shapes: usize,
    /// Of those, kernels replayed from a CUDA/HIP graph (no operator).
    pub graph_replayed: usize,
    /// Of those, kernels linked to neither an operator nor a graph launch.
    pub unlinked: usize,
    /// Total GPU time of the stage's kernels on the stream (before the name
    /// filter): the denominator of every row's `share_pct`.
    pub stage_kernel_us: f64,
    /// How kernels were attributed to the stage's steps.
    pub membership: Membership,
}

/// Joins the stage's kernels (optionally filtered by name) with the CPU
/// operator that launched them, and groups them by kernel name, operator and
/// input shapes: the same kernel at different shapes (decode vs prefill batch
/// sizes) sits at a different point of the roofline. Unlike [`kernel_shapes`]
/// this is scoped to a stage and keeps graph-replayed kernels visible as op
/// `(graph)`. Rows are sorted by total GPU time in the stage.
pub fn stage_shapes(
    trace: &Trace,
    ops: &OpIndex,
    stage: Stage,
    stream: Option<u64>,
    name_filter: &str,
) -> Result<StageShapes, String> {
    let (stream, _) = resolve_stream(trace, stream)?;
    let (membership, steps) = build_steps(trace, ops, stage, stream);
    // `+ 0.0` turns the -0.0 an empty float sum yields into 0.0, so an empty
    // stage prints `stage_kernel_us=0.00`, not `-0.00`.
    let stage_kernel_us: f64 = steps
        .iter()
        .flat_map(|s| s.kernels.iter())
        .map(|(k, _)| k.dur)
        .sum::<f64>()
        + 0.0;
    type Key<'a> = (&'a str, &'a str, &'a str, &'a str, &'a str);
    let mut groups: HashMap<Key, (usize, f64)> = HashMap::new();
    let mut kernels = 0;
    let mut post_kernels = 0;
    let mut with_shapes = 0;
    let mut graph_replayed = 0;
    let mut unlinked = 0;
    for s in &steps {
        for (k, part) in s.kernels.iter().filter(|(k, _)| k.name.contains(name_filter)) {
            if *part == Part::Post {
                post_kernels += 1;
            }
            kernels += 1;
            let (launch, op) = launch_of(ops, k);
            match launch {
                Launch::Graph => graph_replayed += 1,
                Launch::Unknown => unlinked += 1,
                Launch::Op => {
                    if op.and_then(|o| o.input_dims.as_deref()).is_some() {
                        with_shapes += 1;
                    }
                }
            }
            let op_label = match launch {
                Launch::Op => op.map(|o| o.name.as_str()).unwrap_or(""),
                Launch::Graph => GRAPH_OP,
                Launch::Unknown => "",
            };
            let key = (
                k.name.as_str(),
                op_label,
                op.and_then(|o| o.input_dims.as_deref()).unwrap_or(""),
                op.and_then(|o| o.input_types.as_deref()).unwrap_or(""),
                op.and_then(|o| o.flops.as_deref()).unwrap_or(""),
            );
            let g = groups.entry(key).or_default();
            g.0 += 1;
            g.1 += k.dur;
        }
    }
    let mut rows: Vec<StageShapeRow> = groups
        .into_iter()
        .map(|((kernel, op, dims, types, flops), (calls, total_us))| StageShapeRow {
            kernel: kernel.to_string(),
            op: op.to_string(),
            input_dims: dims.to_string(),
            input_types: types.to_string(),
            flops: flops.to_string(),
            calls,
            total_us,
            share_pct: if stage_kernel_us > 0.0 {
                total_us / stage_kernel_us * 100.0
            } else {
                0.0
            },
        })
        .collect();
    rows.sort_by(|a, b| {
        b.total_us
            .total_cmp(&a.total_us)
            .then_with(|| a.key().cmp(&b.key()))
    });
    Ok(StageShapes {
        rows,
        kernels,
        post_kernels,
        with_shapes,
        graph_replayed,
        unlinked,
        stage_kernel_us,
        membership,
    })
}

/// TSV of a stage's kernel shapes: one row per kernel name × launching
/// operator × input dims × input types, ranked by total GPU time in the stage,
/// with calls, total and mean duration, the share of the stage's kernel time,
/// and FLOPs per call. The `#` header reports coverage (kernels in the stage,
/// how many of them post-step, how many with shapes, graph-replayed and
/// unlinked) and `stage_kernel_us`,
/// the `share_pct` denominator; `total` is the row count, so page with
/// `offset`/`limit`.
pub fn stage_shapes_tsv(
    trace: &Trace,
    ops: &OpIndex,
    stage: Stage,
    stream: Option<u64>,
    name_filter: &str,
    offset: usize,
    limit: Option<usize>,
) -> Result<String, String> {
    let (stream, auto) = resolve_stream(trace, stream)?;
    let shapes = stage_shapes(trace, ops, stage, Some(stream), name_filter)?;
    let cap = effective_limit(limit);
    let mut out = format!(
        "# stage_shapes stage={} {} membership={} total={} kernels={} post_kernels={} with_shapes={} graph_replayed={} unlinked={} stage_kernel_us={:.2}\n",
        stage.as_str(),
        stream_note(stream, auto),
        shapes.membership.as_str(),
        shapes.rows.len(),
        shapes.kernels,
        shapes.post_kernels,
        shapes.with_shapes,
        shapes.graph_replayed,
        shapes.unlinked,
        shapes.stage_kernel_us,
    );
    out.push_str("kernel\top\tcalls\ttotal_us\tmean_us\tshare_pct\tinput_dims\tinput_types\tflops_per_call\n");
    for r in shapes.rows.iter().skip(offset).take(cap) {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}\t{}\n",
            tsv_clean(&r.kernel),
            tsv_clean(&r.op),
            r.calls,
            r.total_us,
            r.total_us / r.calls as f64,
            r.share_pct,
            tsv_clean(&r.input_dims),
            tsv_clean(&r.input_types),
            tsv_clean(&r.flops),
        ));
    }
    Ok(out)
}

/// Loads a trace together with its CPU operators, runtime launch events and
/// CPU-side annotations. All MCP stage tools need the full load: step
/// attribution follows the CPU-side launch, and the shape columns come from
/// the operators.
pub fn load_with_ops(path: &str) -> Result<(Trace, OpIndex)> {
    parse_trace_with_ops(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{AnnotationEvent, CpuAnnotationEvent, KernelEvent, LaunchEvent, OpIndex};

    fn kd(stream: u64, ts: f64, name: &str, dur: f64) -> KernelEvent {
        KernelEvent {
            name: name.to_string(),
            cat: "kernel".to_string(),
            ts,
            dur,
            device: 0,
            stream,
            grid: None,
            block: None,
            shared_memory: None,
            registers_per_thread: None,
            correlation: None,
            external_id: None,
            trace_id: 0,
        }
    }

    fn trace_with(stream: u64, n: usize) -> Trace {
        let kernels = (0..n)
            .map(|i| kd(stream, i as f64 * 10.0, &format!("k{}", i), 5.0))
            .collect();
        Trace {
            kernels,
            annotations: vec![],
        }
    }

    const DECODE_ANN: &str = "execute_context_0(0)_generation_1(1)";
    const PREFILL_ANN: &str = "execute_context_1(5)_generation_0(0)";

    fn ann(stream: u64, ts: f64, dur: f64, name: &str) -> AnnotationEvent {
        AnnotationEvent {
            name: name.into(),
            ts,
            dur,
            stream,
            trace_id: 0,
        }
    }

    #[test]
    fn test_stage_parse_arg_valid_and_invalid() {
        assert_eq!(Stage::parse_arg("prefill"), Ok(Stage::Prefill));
        assert_eq!(Stage::parse_arg("decode"), Ok(Stage::Decode));
        assert_eq!(Stage::parse_arg("mixed"), Ok(Stage::Mixed));
        assert!(Stage::parse_arg("none").is_err());
        assert!(Stage::parse_arg("bogus").is_err());
    }

    /// Two identical decode steps (`a b a b` repeated + `c` tail), one prefill
    /// step. Durations differ between the two decode steps so medians can be
    /// checked. Kernels/annotations are in ts order, as the parser emits them.
    /// Membership falls back to the GPU window: the trace has no CPU-side
    /// annotations.
    fn staged_trace() -> (Trace, OpIndex) {
        let mut first = kd(4, 102.0, "a", 2.0);
        first.grid = Some("[32,1,1]".to_string());
        first.block = Some("[128,1,1]".to_string());
        let kernels = vec![
            first,
            kd(4, 106.0, "b", 3.0),
            kd(4, 112.0, "a", 2.0),
            kd(4, 116.0, "b", 3.0),
            kd(4, 122.0, "c", 5.0),
            kd(4, 202.0, "a", 4.0),
            kd(4, 208.0, "b", 5.0),
            kd(4, 216.0, "a", 4.0),
            kd(4, 222.0, "b", 5.0),
            kd(4, 230.0, "c", 7.0),
            kd(4, 501.0, "p0", 10.0),
            kd(4, 512.0, "p1", 1.0),
        ];
        let annotations = vec![
            ann(4, 100.0, 30.0, DECODE_ANN),
            ann(4, 200.0, 40.0, DECODE_ANN),
            ann(4, 500.0, 30.0, PREFILL_ANN),
        ];
        (
            Trace {
                kernels,
                annotations,
            },
            OpIndex::default(),
        )
    }

    #[test]
    fn test_step_token_counts_parses_both_annotation_formats() {
        let simple = |ctx: u64, gen: u64| StepTokens {
            ctx_tokens: ctx,
            gen_tokens: gen,
            ctx_kv_tokens: None,
            gen_kv_tokens: None,
            ctx_qk: None,
            gen_qk: None,
        };
        assert_eq!(
            step_token_counts("execute_context_1(5)_generation_0(0)"),
            Some(simple(5, 0))
        );
        assert_eq!(
            step_token_counts("execute_context_0(0)_generation_1(1)"),
            Some(simple(0, 1))
        );
        // Simple form with a total prefix.
        assert_eq!(
            step_token_counts("execute_2_context_1(5)_generation_1(3)"),
            Some(simple(5, 3))
        );
        assert_eq!(step_token_counts("nccl:_all_gather_base"), None);
        assert_eq!(step_token_counts("execute_context_1(5)"), None);
    }

    #[test]
    fn test_step_token_counts_parses_detailed_format() {
        // Step 0 of a real vLLM `detailed_trace_annotation` trace.
        let t = step_token_counts(
            "execute_16384_context_1(sq16368sk81840sqsq267911424sqsk1339557120)\
             _generation_16(sq16sk232025sqsq16sqsk232025)",
        )
        .expect("detailed annotation parses");
        assert_eq!(t.ctx_tokens, 16368);
        assert_eq!(t.gen_tokens, 16);
        assert_eq!(t.ctx_kv_tokens, Some(81840));
        assert_eq!(t.gen_kv_tokens, Some(232025));
        assert_eq!(t.ctx_qk, Some(1339557120));
        assert_eq!(t.gen_qk, Some(232025));
        // Detailed form without a total prefix, and malformed ones.
        let t2 = step_token_counts(
            "execute_context_2(sq5sk10sqsq5sqsk10)_generation_0(sq0sk0sqsq0sqsk0)",
        )
        .expect("detailed annotation without total parses");
        assert_eq!((t2.ctx_tokens, t2.ctx_kv_tokens), (5, Some(10)));
        assert_eq!(step_token_counts("execute_2_context_1(zz)_generation_1(3)"), None);
        // Appended garbage after the detailed counters is rejected.
        assert_eq!(
            step_token_counts("execute_context_1(sq5sk10sqsq5sqsk10x)_generation_1(3)"),
            None
        );
    }

    #[test]
    fn test_stage_steps_reports_detailed_token_columns() {
        // One mixed step in the detailed format (gpu_window membership: no
        // CPU-side annotations needed to exercise the columns).
        let detailed = "execute_16384_context_1(sq16368sk81840sqsq267911424sqsk1339557120)_generation_16(sq16sk232025sqsq16sqsk232025)";
        let t = Trace {
            kernels: vec![kd(4, 102.0, "a", 2.0)],
            annotations: vec![ann(4, 100.0, 30.0, detailed)],
        };
        let out = stage_steps_tsv(&t, &OpIndex::default(), Stage::Mixed, Some(4), 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[2],
            format!("0\t{detailed}\t100.00\t30.00\t16368\t16\t81840\t232025\t1339557120\t232025\t1\t0")
        );
    }

    const STEPS_HEADER: &str =
        "idx\tname\tts_us\tdur_us\tctx_tokens\tgen_tokens\tctx_kv_tokens\tgen_kv_tokens\tctx_qk\tgen_qk\tkernels\tpost_kernels";

    #[test]
    fn test_stage_steps_lists_steps_per_stage_paginated() {
        let (t, ops) = staged_trace();
        let out = stage_steps_tsv(&t, &ops, Stage::Decode, Some(4), 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[0],
            "# stage_steps stage=decode stream=4 membership=gpu_window total=2"
        );
        assert_eq!(lines[1], STEPS_HEADER);
        assert_eq!(lines.len(), 4, "2 headers + 2 decode steps");
        // Simple annotation format: no KV/QK detail columns -> empty cells.
        assert_eq!(
            lines[2],
            "0\texecute_context_0(0)_generation_1(1)\t100.00\t30.00\t0\t1\t\t\t\t\t5\t0"
        );
        assert_eq!(
            lines[3],
            "1\texecute_context_0(0)_generation_1(1)\t200.00\t40.00\t0\t1\t\t\t\t\t5\t0"
        );
        // Paging: offset=1 returns the second step only.
        let paged = stage_steps_tsv(&t, &ops, Stage::Decode, Some(4), 1, Some(1)).unwrap();
        let paged_lines: Vec<&str> = paged.lines().collect();
        assert_eq!(paged_lines.len(), 3, "2 headers + 1 row");
        assert!(paged_lines[2].starts_with("1\t"));
        // The prefill stage sees its single step, with ctx tokens.
        let prefill = stage_steps_tsv(&t, &ops, Stage::Prefill, Some(4), 0, None).unwrap();
        let plines: Vec<&str> = prefill.lines().collect();
        assert_eq!(
            plines[0],
            "# stage_steps stage=prefill stream=4 membership=gpu_window total=1"
        );
        assert!(plines[2].contains("\t500.00\t30.00\t5\t0\t\t\t\t\t2\t0"));
        // A stage absent from the trace lists no steps.
        let mixed = stage_steps_tsv(&t, &ops, Stage::Mixed, Some(4), 0, None).unwrap();
        assert_eq!(mixed.lines().count(), 2, "headers only");
    }

    #[test]
    fn test_empty_stage_reports_zero_not_negative_zero() {
        let (t, ops) = staged_trace();
        let out = stage_shapes_tsv(&t, &ops, Stage::Mixed, Some(4), "", 0, None).unwrap();
        let header = out.lines().next().unwrap();
        assert!(
            header.contains("stage_kernel_us=0.00") && !header.contains("-0.00"),
            "an empty stage sums to -0.0 f64; it must print 0.00: {header}"
        );
    }

    #[test]
    fn test_stage_steps_past_end_is_header_only() {
        let (t, ops) = staged_trace();
        let out = stage_steps_tsv(&t, &ops, Stage::Decode, Some(4), 99, None).unwrap();
        assert_eq!(out.lines().count(), 2, "headers only when offset >= total");
    }

    #[test]
    fn test_default_stream_picks_most_kernel_time() {
        let t = Trace {
            kernels: vec![kd(4, 0.0, "x", 10.0), kd(7, 0.0, "y", 40.0)],
            annotations: vec![ann(7, -1.0, 100.0, DECODE_ANN)],
        };
        assert_eq!(default_stream(&t), Some(7));
        // Ties break toward the lowest stream id.
        let tie = Trace {
            kernels: vec![kd(9, 0.0, "a", 5.0), kd(3, 10.0, "b", 5.0)],
            annotations: vec![],
        };
        assert_eq!(default_stream(&tie), Some(3));
        assert_eq!(default_stream(&Trace::default()), None);
    }

    #[test]
    fn test_default_stream_is_marked_in_header() {
        let t = Trace {
            kernels: vec![kd(4, 0.0, "x", 10.0), kd(7, 0.0, "y", 40.0)],
            annotations: vec![ann(7, -1.0, 100.0, DECODE_ANN)],
        };
        let out = stage_steps_tsv(&t, &OpIndex::default(), Stage::Decode, None, 0, None).unwrap();
        assert!(
            out.lines().next().unwrap().contains("stream=7(auto)"),
            "auto-chosen stream is reported: {}",
            out.lines().next().unwrap()
        );
        // Explicit stream -> plain marker, no "(auto)".
        let explicit =
            stage_steps_tsv(&t, &OpIndex::default(), Stage::Decode, Some(4), 0, None).unwrap();
        let header = explicit.lines().next().unwrap();
        assert!(header.contains("stream=4 ") && !header.contains("(auto)"), "{header}");
    }

    #[test]
    fn test_step_kernels_offsets_gaps_grid_block() {
        let (t, ops) = staged_trace();
        let out = step_kernels_tsv(&t, &ops, Stage::Decode, Some(4), 0, 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[0].starts_with(
            "# step_kernels stage=decode stream=4 membership=gpu_window step=0 \
             step_name=execute_context_0(0)_generation_1(1) step_ts_us=100.00 step_dur_us=30.00 total=5 post=0"
        ));
        assert_eq!(
            lines[1],
            "pos\tname\tpart\tstart_offset_us\tdur_us\tgap_us\tgrid\tblock\tlaunch\top\tinput_dims\tinput_types\tflops"
        );
        // First kernel: offset and gap measured from the step start (ts=100).
        // Without operator info the shape columns are empty.
        assert_eq!(
            lines[2],
            "0\ta\tstep\t2.00\t2.00\t2.00\t[32,1,1]\t[128,1,1]\t\t\t\t\t"
        );
        assert_eq!(lines[3], "1\tb\tstep\t6.00\t3.00\t2.00\t\t\t\t\t\t\t");
        assert_eq!(lines[4], "2\ta\tstep\t12.00\t2.00\t3.00\t\t\t\t\t\t\t");
        assert_eq!(lines[5], "3\tb\tstep\t16.00\t3.00\t2.00\t\t\t\t\t\t\t");
        assert_eq!(lines[6], "4\tc\tstep\t22.00\t5.00\t3.00\t\t\t\t\t\t\t");
        // Step 1 exists too.
        let step1 = step_kernels_tsv(&t, &ops, Stage::Decode, Some(4), 1, 0, None).unwrap();
        assert!(step1.lines().any(|l| l.starts_with("4\tc\tstep\t30.00\t7.00")));
    }

    #[test]
    fn test_step_kernels_pagination_and_cap() {
        let (t, ops) = staged_trace();
        let out = step_kernels_tsv(&t, &ops, Stage::Decode, Some(4), 0, 2, Some(2)).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 4, "2 headers + 2 rows");
        assert!(lines[2].starts_with("2\ta\t"));
        // Limit is capped at MAX_ROWS regardless of the request.
        let mut big = trace_with(4, MAX_ROWS + 500);
        big.annotations.push(ann(4, -1.0, 1e9, DECODE_ANN));
        let capped =
            step_kernels_tsv(&big, &OpIndex::default(), Stage::Decode, None, 0, 0, Some(usize::MAX))
                .unwrap();
        assert_eq!(capped.lines().count(), MAX_ROWS + 2, "2 headers + MAX_ROWS");
    }

    #[test]
    fn test_step_kernels_out_of_range_errs() {
        let (t, ops) = staged_trace();
        let err = step_kernels_tsv(&t, &ops, Stage::Decode, Some(4), 5, 0, None).unwrap_err();
        assert!(err.contains("step 5 out of range"), "{err}");
        assert!(err.contains("2 step(s)"), "{err}");
        let err = step_kernels_tsv(&t, &ops, Stage::Mixed, Some(4), 0, 0, None).unwrap_err();
        assert!(err.contains("0 step(s)"), "{err}");
    }

    #[test]
    fn test_stage_patterns_collapse_identical_steps() {
        let (t, ops) = staged_trace();
        let out = stage_patterns_tsv(&t, &ops, Stage::Decode, Some(4), 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[0],
            "# stage_patterns stage=decode stream=4 membership=gpu_window total=1"
        );
        assert_eq!(
            lines[1],
            "pattern\tsteps\tkernels_per_step\tmedian_total_dur_us\trepeat\tstep_indices"
        );
        // total_durs 15 and 25 -> median 20; [a,b] repeats twice -> 0..3x2.
        assert_eq!(lines[2], "0\t2\t5\t20.00\t0..3x2\t0,1");
    }

    #[test]
    fn test_stage_patterns_ranked_by_time_and_indexed() {
        // Two distinct patterns: an expensive one (later steps) and a cheap one.
        let kernels = vec![
            kd(4, 100.0, "cheap", 1.0),
            kd(4, 200.0, "big", 10.0),
            kd(4, 300.0, "big", 10.0),
        ];
        let annotations = vec![
            ann(4, 99.0, 10.0, DECODE_ANN),
            ann(4, 199.0, 50.0, DECODE_ANN),
            ann(4, 299.0, 50.0, DECODE_ANN),
        ];
        let t = Trace {
            kernels,
            annotations,
        };
        let out = stage_patterns_tsv(&t, &OpIndex::default(), Stage::Decode, Some(4), 0, None)
            .unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[0],
            "# stage_patterns stage=decode stream=4 membership=gpu_window total=2"
        );
        assert_eq!(lines[2], "0\t2\t1\t10.00\t\t1,2", "more total time first");
        assert_eq!(lines[3], "1\t1\t1\t1.00\t\t0");
    }

    #[test]
    fn test_detect_repeat_block_variants() {
        let to_ids = |names: &[&str]| -> Vec<u32> {
            name_ids(&names.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        };
        let r = detect_repeat_block(&to_ids(&["a", "b", "a", "b", "a", "b", "c"])).unwrap();
        assert_eq!((r.start, r.len, r.count), (0, 2, 3));
        let r = detect_repeat_block(&to_ids(&["h", "l1", "l2", "l1", "l2", "t"])).unwrap();
        assert_eq!((r.start, r.len, r.count), (1, 2, 2));
        // No repeating sub-sequence found.
        assert!(detect_repeat_block(&to_ids(&["x", "y"])).is_none());
        assert!(detect_repeat_block(&to_ids(&["x"])).is_none());
    }

    #[test]
    fn test_stage_pattern_detail_collapses_repeat_block() {
        let (t, ops) = staged_trace();
        let out = stage_pattern_detail_tsv(&t, &ops, Stage::Decode, Some(4), 0, 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[0],
            "# stage_pattern stage=decode stream=4 membership=gpu_window pattern=0 steps=2 kernels_per_step=5 total=3"
        );
        // Per-rep block totals: step0 [5,5], step1 [9,9] -> median 7.00.
        assert_eq!(
            lines[1],
            "# repeat_block start=0 len=2 count=2 median_block_dur_us=7.00"
        );
        assert_eq!(
            lines[2],
            "pos\tname\tpart\tmedian_dur_us\tsamples\tlaunch\top\tinput_dims\tinput_types\tflops\tshape_varies"
        );
        // pos 0/1 median over steps*reps = 4 samples; tail over 2 steps.
        assert_eq!(lines[3], "0\ta\trepeat\t3.00\t4\t\t\t\t\t\t");
        assert_eq!(lines[4], "1\tb\trepeat\t4.00\t4\t\t\t\t\t\t");
        assert_eq!(lines[5], "4\tc\ttail\t6.00\t2\t\t\t\t\t\t");
    }

    #[test]
    fn test_stage_pattern_detail_without_repeat() {
        let kernels = vec![kd(4, 100.0, "x", 1.0), kd(4, 102.0, "y", 2.0)];
        let t = Trace {
            kernels,
            annotations: vec![ann(4, 99.0, 10.0, DECODE_ANN)],
        };
        let out = stage_pattern_detail_tsv(&t, &OpIndex::default(), Stage::Decode, Some(4), 0, 0, None)
            .unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "# stage_pattern stage=decode stream=4 membership=gpu_window pattern=0 steps=1 kernels_per_step=2 total=2");
        assert_eq!(
            lines[1],
            "pos\tname\tpart\tmedian_dur_us\tsamples\tlaunch\top\tinput_dims\tinput_types\tflops\tshape_varies",
            "no repeat_block header line"
        );
        assert_eq!(lines[2], "0\tx\thead\t1.00\t1\t\t\t\t\t\t");
        assert_eq!(lines[3], "1\ty\thead\t2.00\t1\t\t\t\t\t\t");
    }

    #[test]
    fn test_stage_pattern_detail_out_of_range_errs() {
        let (t, ops) = staged_trace();
        let err =
            stage_pattern_detail_tsv(&t, &ops, Stage::Decode, Some(4), 9, 0, None).unwrap_err();
        assert!(err.contains("pattern 9 out of range"), "{err}");
    }

    fn op(name: &str, dims: &str) -> crate::trace::CpuOp {
        crate::trace::CpuOp {
            name: name.into(),
            input_dims: Some(dims.into()),
            input_types: Some(r#"["c10::BFloat16"]"#.into()),
            flops: Some("100".into()),
        }
    }

    fn launch_evt(ts: f64, pid: u64, ext: Option<u64>, graph: bool) -> LaunchEvent {
        LaunchEvent {
            ts,
            pid,
            external_id: ext,
            graph,
        }
    }

    /// gemm at two shapes (decode and prefill), a norm linked through its
    /// runtime launch, and a graph-replayed kernel without an operator.
    fn shapes_trace() -> (Trace, OpIndex) {
        let link = |k: KernelEvent, ext: Option<u64>, corr: Option<u64>| KernelEvent {
            external_id: ext,
            correlation: corr,
            ..k
        };
        let kernels = vec![
            link(kd(4, 0.0, "gemm", 2.0), Some(1), None),
            link(kd(4, 10.0, "gemm", 4.0), Some(1), None),
            link(kd(4, 20.0, "gemm", 30.0), Some(2), None),
            link(kd(4, 60.0, "norm", 1.0), None, Some(50)),
            link(kd(4, 70.0, "replayed", 7.0), None, Some(51)),
        ];
        let mut ops = OpIndex::default();
        ops.ops.insert(1, op("aten::mm", "[[16,64],[64,32]]"));
        ops.ops.insert(2, op("aten::mm", "[[4096,64],[64,32]]"));
        ops.ops.insert(3, op("vllm::rms_norm", "[[16,64]]"));
        ops.launches.insert(50, launch_evt(0.0, 1, Some(3), false));
        ops.launches.insert(51, launch_evt(0.0, 1, None, true));
        (
            Trace {
                kernels,
                annotations: vec![],
            },
            ops,
        )
    }

    #[test]
    fn test_kernel_shapes_groups_by_shape_and_ranks_by_time() {
        let (t, ops) = shapes_trace();
        let s = kernel_shapes(&t, &ops, "");
        assert_eq!(s.kernels, 5);
        assert_eq!(s.unlinked, 1);
        let summary: Vec<(&str, &str, &str, usize, f64)> = s
            .rows
            .iter()
            .map(|r| {
                (
                    r.kernel.as_str(),
                    r.op.as_str(),
                    r.input_dims.as_str(),
                    r.calls,
                    r.total_dur,
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                ("gemm", "aten::mm", "[[4096,64],[64,32]]", 1, 30.0),
                ("replayed", "", "", 1, 7.0),
                ("gemm", "aten::mm", "[[16,64],[64,32]]", 2, 6.0),
                ("norm", "vllm::rms_norm", "[[16,64]]", 1, 1.0),
            ]
        );
    }

    #[test]
    fn test_kernel_shapes_name_filter() {
        let (t, ops) = shapes_trace();
        let s = kernel_shapes(&t, &ops, "gem");
        assert_eq!(s.kernels, 3);
        assert_eq!(s.unlinked, 0);
        assert!(s.rows.iter().all(|r| r.kernel == "gemm"));
    }

    #[test]
    fn test_kernel_shapes_csv_columns_and_escaping() {
        let (t, ops) = shapes_trace();
        let csv = kernel_shapes_csv(&kernel_shapes(&t, &ops, "").rows);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "kernel,op,calls,total_us,mean_us,input_dims,input_types,flops_per_call"
        );
        assert_eq!(
            lines[3],
            r#"gemm,aten::mm,2,6.0,3.00,"[[16,64],[64,32]]","[""c10::BFloat16""]",100"#
        );
        assert_eq!(lines[2], "replayed,,1,7.0,7.00,,,");
    }

    /// Launch-based step attribution: a prefill CPU annotation on pid 1 at
    /// [100,200) whose kernels run at ts 500+ (the GPU is behind the CPU, so
    /// the kernels run after the annotation ended), and a decode CPU
    /// annotation at [300,400) whose kernel is graph-replayed. One kernel
    /// launches before every step, so it is attributed to no step.
    fn launch_trace() -> (Trace, OpIndex) {
        let link = |k: KernelEvent, corr: u64| KernelEvent {
            correlation: Some(corr),
            ..k
        };
        let kernels = vec![
            link(kd(4, 500.0, "gemm", 10.0), 10),
            link(kd(4, 520.0, "norm", 3.0), 12),
            link(kd(4, 600.0, "graph_k", 20.0), 20),
            link(kd(4, 700.0, "lost", 5.0), 30),
        ];
        let mut ops = OpIndex::default();
        ops.cpu_annotations = vec![
            CpuAnnotationEvent {
                name: PREFILL_ANN.into(),
                ts: 100.0,
                dur: 100.0,
                pid: 1,
            },
            CpuAnnotationEvent {
                name: DECODE_ANN.into(),
                ts: 300.0,
                dur: 100.0,
                pid: 1,
            },
        ];
        ops.ops.insert(1, op("aten::mm", "[[16,64],[64,32]]"));
        ops.ops.insert(2, op("vllm::rms_norm", "[[16,64]]"));
        // The step kernels' launches sit inside their step's CPU annotation
        // window; the "lost" kernel launches before the first step.
        ops.launches.insert(10, launch_evt(150.0, 1, Some(1), false));
        ops.launches.insert(12, launch_evt(160.0, 1, Some(2), false));
        ops.launches.insert(20, launch_evt(350.0, 1, None, true));
        ops.launches.insert(30, launch_evt(50.0, 1, None, false));
        (
            Trace {
                kernels,
                annotations: vec![],
            },
            ops,
        )
    }

    #[test]
    fn test_stage_steps_launch_membership_tracks_kernels_running_late() {
        let (t, ops) = launch_trace();
        let out = stage_steps_tsv(&t, &ops, Stage::Prefill, Some(4), 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[0],
            "# stage_steps stage=prefill stream=4 membership=launch total=1"
        );
        assert_eq!(
            lines[2],
            "0\texecute_context_1(5)_generation_0(0)\t500.00\t23.00\t5\t0\t\t\t\t\t2\t0",
            "kernels running after the CPU annotation ended still count, \
             and the span is the GPU span of the kernels"
        );
        // The decode step holds the graph-replayed kernel only.
        let decode = stage_steps_tsv(&t, &ops, Stage::Decode, Some(4), 0, None).unwrap();
        let dlines: Vec<&str> = decode.lines().collect();
        assert_eq!(
            dlines[2],
            "0\texecute_context_0(0)_generation_1(1)\t600.00\t20.00\t0\t1\t\t\t\t\t1\t0"
        );
        // The early-launched "lost" kernel belongs to no step of any stage.
        let prefill = stage_shapes_tsv(&t, &ops, Stage::Prefill, Some(4), "", 0, None).unwrap();
        assert!(prefill.lines().next().unwrap().contains("kernels=2"), "{prefill}");
    }

    #[test]
    fn test_step_kernels_shape_columns_by_launch_kind() {
        let (t, ops) = launch_trace();
        let prefill = step_kernels_tsv(&t, &ops, Stage::Prefill, Some(4), 0, 0, None).unwrap();
        let lines: Vec<&str> = prefill.lines().collect();
        assert_eq!(
            lines[2],
            "0\tgemm\tstep\t0.00\t10.00\t0.00\t\t\top\taten::mm\t[[16,64],[64,32]]\t[\"c10::BFloat16\"]\t100"
        );
        assert_eq!(
            lines[3],
            "1\tnorm\tstep\t20.00\t3.00\t10.00\t\t\top\tvllm::rms_norm\t[[16,64]]\t[\"c10::BFloat16\"]\t100"
        );
        // Graph-replayed kernel of the decode step: launch=graph, no operator.
        let decode = step_kernels_tsv(&t, &ops, Stage::Decode, Some(4), 0, 0, None).unwrap();
        let dlines: Vec<&str> = decode.lines().collect();
        assert_eq!(dlines[2], "0\tgraph_k\tstep\t0.00\t20.00\t0.00\t\t\tgraph\t\t\t\t");
    }

    /// A sampler-like kernel launched between two steps attaches to the step
    /// that closed before it as post-step work; likewise a kernel launched
    /// after the last step. A kernel launched before the first step belongs
    /// to no step.
    fn post_launch_trace() -> (Trace, OpIndex) {
        let link = |k: KernelEvent, corr: u64| KernelEvent {
            correlation: Some(corr),
            ..k
        };
        let kernels = vec![
            link(kd(4, 900.0, "early", 1.0), 9),
            link(kd(4, 1000.0, "mk0", 10.0), 10),
            link(kd(4, 1010.0, "sampler", 2.0), 11),
            link(kd(4, 1020.0, "mk1", 8.0), 13),
            link(kd(4, 1030.0, "lm_head", 4.0), 14),
        ];
        let mut ops = OpIndex::default();
        ops.cpu_annotations = vec![
            CpuAnnotationEvent {
                name: DECODE_ANN.into(),
                ts: 100.0,
                dur: 100.0,
                pid: 1,
            },
            CpuAnnotationEvent {
                name: DECODE_ANN.into(),
                ts: 400.0,
                dur: 100.0,
                pid: 1,
            },
        ];
        ops.launches.insert(9, launch_evt(50.0, 1, None, false)); // before the first step
        ops.launches.insert(10, launch_evt(150.0, 1, None, false)); // inside step 0
        ops.launches.insert(11, launch_evt(250.0, 1, None, false)); // between the steps
        ops.launches.insert(13, launch_evt(450.0, 1, None, false)); // inside step 1
        ops.launches.insert(14, launch_evt(600.0, 1, None, false)); // after the last step
        (
            Trace {
                kernels,
                annotations: vec![],
            },
            ops,
        )
    }

    #[test]
    fn test_post_step_attribution_by_launch() {
        let (t, ops) = post_launch_trace();
        let steps = stage_steps_tsv(&t, &ops, Stage::Decode, Some(4), 0, None).unwrap();
        let lines: Vec<&str> = steps.lines().collect();
        // Spans include the post-step kernels (first kernel start to last end).
        assert_eq!(
            lines[2],
            "0\texecute_context_0(0)_generation_1(1)\t1000.00\t12.00\t0\t1\t\t\t\t\t1\t1"
        );
        assert_eq!(
            lines[3],
            "1\texecute_context_0(0)_generation_1(1)\t1020.00\t14.00\t0\t1\t\t\t\t\t1\t1"
        );
        let step0 = step_kernels_tsv(&t, &ops, Stage::Decode, Some(4), 0, 0, None).unwrap();
        let s0: Vec<&str> = step0.lines().collect();
        assert!(s0[0].contains("total=2 post=1"), "{step0}");
        assert_eq!(s0[2], "0\tmk0\tstep\t0.00\t10.00\t0.00\t\t\t\t\t\t\t");
        assert_eq!(s0[3], "1\tsampler\tpost\t10.00\t2.00\t0.00\t\t\t\t\t\t\t");
        let step1 = step_kernels_tsv(&t, &ops, Stage::Decode, Some(4), 1, 0, None).unwrap();
        let s1: Vec<&str> = step1.lines().collect();
        assert_eq!(s1[3], "1\tlm_head\tpost\t10.00\t4.00\t2.00\t\t\t\t\t\t\t");
        // stage_shapes counts the post-step kernels and reports them.
        let shapes = stage_shapes_tsv(&t, &ops, Stage::Decode, Some(4), "", 0, None).unwrap();
        let header = shapes.lines().next().unwrap();
        assert!(header.contains("kernels=4"), "{shapes}");
        assert!(header.contains("post_kernels=2"), "{shapes}");
        assert!(shapes.lines().any(|l| l.starts_with("sampler\t")));
    }

    #[test]
    fn test_post_step_attribution_by_gpu_window() {
        let kernels = vec![
            kd(4, 50.0, "early", 1.0),   // before the first window: unattributed
            kd(4, 110.0, "a", 2.0),      // inside decode window A
            kd(4, 150.0, "sampler", 3.0),// gap after A: post-step of A
            kd(4, 210.0, "p", 4.0),      // inside the prefill window
            kd(4, 250.0, "postp", 5.0),  // gap after prefill: post-step of it
            kd(4, 410.0, "b", 6.0),      // inside decode window B
            kd(4, 460.0, "lm_head", 7.0),// after the last window: post-step of B
        ];
        let annotations = vec![
            ann(4, 100.0, 30.0, DECODE_ANN), // window A
            ann(4, 200.0, 30.0, PREFILL_ANN),
            ann(4, 400.0, 30.0, DECODE_ANN), // window B
        ];
        let t = Trace { kernels, annotations };
        let ops = OpIndex::default();
        // Decode: A holds its kernel + sampler; B holds its kernel + lm_head.
        let decode = stage_steps_tsv(&t, &ops, Stage::Decode, Some(4), 0, None).unwrap();
        let dlines: Vec<&str> = decode.lines().collect();
        assert!(dlines[2].ends_with("\t1\t1"), "{decode}");
        assert!(dlines[3].ends_with("\t1\t1"), "{decode}");
        let step0 = step_kernels_tsv(&t, &ops, Stage::Decode, Some(4), 0, 0, None).unwrap();
        let s0: Vec<&str> = step0.lines().collect();
        assert_eq!(s0[2], "0\ta\tstep\t10.00\t2.00\t10.00\t\t\t\t\t\t\t");
        assert_eq!(s0[3], "1\tsampler\tpost\t50.00\t3.00\t38.00\t\t\t\t\t\t\t");
        // The gap after the prefill window belongs to prefill, not decode.
        let prefill = stage_steps_tsv(&t, &ops, Stage::Prefill, Some(4), 0, None).unwrap();
        let plines: Vec<&str> = prefill.lines().collect();
        assert!(plines[2].ends_with("\t1\t1"), "{prefill}");
        assert!(plines[2].contains("execute_context_1(5)"));
    }

    #[test]
    fn test_step_kernels_launch_stream_filter_and_fallback_window() {
        // The prefill gemm also exists on another stream; the requested stream
        // only sees its own kernels.
        let (mut t, ops) = launch_trace();
        let mut other = kd(7, 501.0, "gemm", 4.0);
        other.correlation = Some(11);
        t.kernels.push(other);
        t.kernels.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap());
        let mut ops = ops;
        ops.launches.insert(11, launch_evt(151.0, 1, Some(1), false));
        let out = step_kernels_tsv(&t, &ops, Stage::Prefill, Some(7), 0, 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3, "2 headers + only stream 7's kernel");

        // A trace without CPU annotations falls back to the GPU window.
        let t = Trace {
            kernels: vec![kd(4, 102.0, "a", 2.0)],
            annotations: vec![ann(4, 100.0, 30.0, DECODE_ANN)],
        };
        let out = stage_steps_tsv(&t, &OpIndex::default(), Stage::Decode, Some(4), 0, None).unwrap();
        assert!(
            out.lines().next().unwrap().contains("membership=gpu_window"),
            "{out}"
        );
    }

    /// Three decode steps with identical kernel names; the gemm sits at two
    /// shapes: A in the first two steps, B in the third.
    fn varying_shape_trace() -> (Trace, OpIndex) {
        let link = |k: KernelEvent, ext: u64| KernelEvent {
            external_id: Some(ext),
            ..k
        };
        let kernels = vec![
            link(kd(4, 100.0, "gemm", 2.0), 1),
            link(kd(4, 110.0, "norm", 1.0), 10),
            link(kd(4, 200.0, "gemm", 2.0), 1),
            link(kd(4, 210.0, "norm", 1.0), 10),
            link(kd(4, 300.0, "gemm", 4.0), 2),
            link(kd(4, 310.0, "norm", 1.0), 10),
        ];
        let annotations = vec![
            ann(4, 99.0, 50.0, DECODE_ANN),
            ann(4, 199.0, 50.0, DECODE_ANN),
            ann(4, 299.0, 50.0, DECODE_ANN),
        ];
        let mut nn = op("vllm::rms_norm", "[[16,64]]");
        nn.flops = None;
        let mut ops = OpIndex::default();
        ops.ops.insert(1, op("aten::mm", "[[16,64],[64,32]]"));
        ops.ops.insert(2, op("aten::mm", "[[32,64],[64,32]]"));
        ops.ops.insert(10, nn);
        (Trace { kernels, annotations }, ops)
    }

    #[test]
    fn test_stage_pattern_detail_reports_mode_shape_and_variation() {
        let (t, ops) = varying_shape_trace();
        let out = stage_pattern_detail_tsv(&t, &ops, Stage::Decode, Some(4), 0, 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[2],
            "0\tgemm\thead\t2.00\t3\top\taten::mm\t[[16,64],[64,32]]\t[\"c10::BFloat16\"]\t100\t*",
            "shape A (2 of 3 steps) is the mode; variation is marked"
        );
        assert_eq!(
            lines[3],
            "1\tnorm\thead\t1.00\t3\top\tvllm::rms_norm\t[[16,64]]\t[\"c10::BFloat16\"]\t\t",
            "same shape in every step: no * marker, empty flops"
        );
    }

    #[test]
    fn test_stage_shapes_ranks_aggregates_graph_and_reports_coverage() {
        let (t, ops) = varying_shape_trace();
        let out = stage_shapes_tsv(&t, &ops, Stage::Decode, Some(4), "", 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        // Stage kernel time: gemm 2+2+4 + norm 1+1+1 = 11. Shape A gemm: 4/11.
        assert_eq!(
            lines[0],
            "# stage_shapes stage=decode stream=4 membership=gpu_window total=3 kernels=6 post_kernels=0 with_shapes=6 graph_replayed=0 unlinked=0 stage_kernel_us=11.00"
        );
        assert_eq!(
            lines[1],
            "kernel\top\tcalls\ttotal_us\tmean_us\tshare_pct\tinput_dims\tinput_types\tflops_per_call"
        );
        assert_eq!(
            lines[2],
            "gemm\taten::mm\t2\t4.00\t2.00\t36.36\t[[16,64],[64,32]]\t[\"c10::BFloat16\"]\t100"
        );
        assert_eq!(
            lines[3],
            "gemm\taten::mm\t1\t4.00\t4.00\t36.36\t[[32,64],[64,32]]\t[\"c10::BFloat16\"]\t100"
        );
        assert_eq!(
            lines[4],
            "norm\tvllm::rms_norm\t3\t3.00\t1.00\t27.27\t[[16,64]]\t[\"c10::BFloat16\"]\t"
        );

        // Graph-replayed kernels are aggregated as op (graph).
        let (t, ops) = launch_trace();
        let out = stage_shapes_tsv(&t, &ops, Stage::Decode, Some(4), "", 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[0].contains("membership=launch"));
        assert!(lines[0].contains("kernels=1 post_kernels=0 with_shapes=0 graph_replayed=1 unlinked=0"), "{lines:?}");
        assert_eq!(lines[2], "graph_k\t(graph)\t1\t20.00\t20.00\t100.00\t\t\t");
    }

    #[test]
    fn test_stage_shapes_name_filter_and_pagination() {
        let (t, ops) = varying_shape_trace();
        // Name filter narrows the coverage counts; share_pct still divides by
        // the whole stage's kernel time.
        let out = stage_shapes_tsv(&t, &ops, Stage::Decode, Some(4), "gem", 0, None).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[0].contains("kernels=3 post_kernels=0 with_shapes=3"));
        assert!(lines[0].contains("stage_kernel_us=11.00"));
        assert_eq!(lines.len(), 4, "2 headers + 2 gemm rows + nothing else");
        // Paging: one row per page.
        let paged = stage_shapes_tsv(&t, &ops, Stage::Decode, Some(4), "", 2, Some(1)).unwrap();
        let plines: Vec<&str> = paged.lines().collect();
        assert_eq!(plines.len(), 3, "2 headers + 1 row");
        assert!(plines[2].starts_with("norm\t"));
        let capped = stage_shapes_tsv(&t, &ops, Stage::Decode, Some(4), "", 0, Some(usize::MAX));
        assert_eq!(capped.unwrap().lines().count(), 5, "2 headers + 3 rows fit under MAX_ROWS");
    }

    fn unique_tmp(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ptt-{tag}-{}-{n}", std::process::id()))
    }

    fn touch(path: &std::path::Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"").unwrap();
    }

    #[test]
    fn test_list_traces_in_dir_finds_nested() {
        let root = unique_tmp("nested");
        touch(&root.join("logs/run_a/traces/x.pt.trace.json.gz"));
        touch(&root.join("logs/run_b/traces/x.pt.trace.json.gz"));
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(
            got,
            vec![
                "logs/run_a/traces/x.pt.trace.json.gz".to_string(),
                "logs/run_b/traces/x.pt.trace.json.gz".to_string(),
            ]
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_list_traces_in_dir_finds_flat_and_nested() {
        let root = unique_tmp("flatnested");
        touch(&root.join("flat.pt.trace.json.gz"));
        touch(&root.join("logs/run/traces/deep.pt.trace.json.gz"));
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(
            got,
            vec![
                "flat.pt.trace.json.gz".to_string(),
                "logs/run/traces/deep.pt.trace.json.gz".to_string(),
            ]
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_list_traces_in_dir_empty_is_ok_not_err() {
        let root = unique_tmp("empty");
        std::fs::create_dir_all(&root).unwrap();
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert!(got.is_empty());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_list_traces_in_dir_ignores_non_traces() {
        let root = unique_tmp("ignore");
        touch(&root.join("a.json"));
        touch(&root.join("b.txt"));
        touch(&root.join("c.pt.trace.json"));
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert!(got.is_empty(), "only *.pt.trace.json.gz counts: {got:?}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_list_traces_in_dir_skips_hidden_dirs() {
        let root = unique_tmp("hidden");
        touch(&root.join(".cache/traces/x.pt.trace.json.gz"));
        touch(&root.join("visible/traces/y.pt.trace.json.gz"));
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(got, vec!["visible/traces/y.pt.trace.json.gz".to_string()]);
        std::fs::remove_dir_all(&root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn test_list_traces_in_dir_does_not_follow_symlink_cycle() {
        let root = unique_tmp("cycle");
        touch(&root.join("logs/traces/x.pt.trace.json.gz"));
        // A symlink pointing back to an ancestor would loop a naive walker.
        std::os::unix::fs::symlink(&root, root.join("logs/loop")).unwrap();
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(got, vec!["logs/traces/x.pt.trace.json.gz".to_string()]);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_label_run_and_rank() {
        assert_eq!(
            trace_display_label(
                "logs/20250115_1200-tp8/traces/dp0_pp0_tp7_dcp0_ep7_rank7.178.pt.trace.json.gz"
            ),
            "20250115_1200-tp8/rank7"
        );
    }

    #[test]
    fn test_label_run_no_rank() {
        assert_eq!(
            trace_display_label("logs/run/traces/foo.pt.trace.json.gz"),
            "run/foo"
        );
    }

    #[test]
    fn test_label_flat_stem() {
        assert_eq!(trace_display_label("flat.pt.trace.json.gz"), "flat");
    }

    #[test]
    fn test_label_no_traces_parent_uses_parent() {
        assert_eq!(
            trace_display_label("logs/run/foo_rank3.pt.trace.json.gz"),
            "run/rank3"
        );
    }

    #[test]
    fn test_label_rank_parsing_boundaries() {
        assert_eq!(
            trace_display_label("logs/r/traces/a_rank12_b.pt.trace.json.gz"),
            "r/rank12"
        );
        assert_eq!(
            trace_display_label("logs/r/traces/norankhere.pt.trace.json.gz"),
            "r/norankhere"
        );
    }

    #[test]
    fn test_label_empty_stem_fallback() {
        assert_eq!(trace_display_label(".pt.trace.json.gz"), "trace");
    }
}
