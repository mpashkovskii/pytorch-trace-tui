//! MCP (Model Context Protocol) server exposing read-only, bounded
//! trace-analysis tools over stdio, so an AI assistant can inspect PyTorch
//! profiler traces without a human driving the TUI.
//!
//! The tools are centred on vLLM execution stages (`prefill`, `decode`,
//! `mixed`): listing a stage's execution steps, reading the kernel sequence
//! inside one step, collapsing per-step kernel sequences into patterns with
//! the repeating per-layer block detected, and aggregating kernel shapes per
//! stage. Every tool delegates to the pure cores in [`crate::mcp_core`], which
//! enforce `limit`/`offset` bounds — traces are 800MB+ / millions of kernels,
//! so an unbounded dump would overflow the caller's context.

use crate::mcp_core;
use crate::mcp_core::Stage;
use anyhow::Result;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::{tool, tool_router, ServiceExt};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct TraceMcpServer;

fn stage_arg(s: &str) -> Result<Stage, String> {
    Stage::parse_arg(s)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StageStepsParam {
    /// Path to a `.pt.trace.json` or `.pt.trace.json.gz` file (e.g. one
    /// returned by `list_traces`).
    pub path: String,
    /// Which execution stage: `prefill`, `decode`, or `mixed`.
    pub stage: String,
    /// CUDA stream id. Omit to use the stream with the most kernel time; the
    /// output header reports which stream was used (`stream=N(auto)`).
    pub stream: Option<u64>,
    /// Rows to skip over the stage's step list. Defaults to 0.
    #[serde(default)]
    pub offset: usize,
    /// Max rows to return (default 100, hard cap 1000). The header's `total=`
    /// is the number of steps, so you know whether to keep paging.
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StepKernelsParam {
    /// Path to a `.pt.trace.json` or `.pt.trace.json.gz` file.
    pub path: String,
    /// Which execution stage: `prefill`, `decode`, or `mixed`.
    pub stage: String,
    /// Which step: the `idx` column value from `stage_steps` (0-based index
    /// into the stage's steps on the stream, ordered by time).
    pub step: usize,
    /// CUDA stream id. Omit to use the stream with the most kernel time; the
    /// output header reports which stream was used (`stream=N(auto)`).
    pub stream: Option<u64>,
    /// Kernels to skip before the first returned row. Defaults to 0.
    #[serde(default)]
    pub offset: usize,
    /// Max rows to return (default 100, hard cap 1000). The header's `total=`
    /// is the number of kernels in the step, so you know whether to keep
    /// paging.
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StagePatternsParam {
    /// Path to a `.pt.trace.json` or `.pt.trace.json.gz` file.
    pub path: String,
    /// Which execution stage: `prefill`, `decode`, or `mixed`.
    pub stage: String,
    /// CUDA stream id. Omit to use the stream with the most kernel time; the
    /// output header reports which stream was used (`stream=N(auto)`).
    pub stream: Option<u64>,
    /// Optional: a `pattern` index from the list mode. With it, returns the
    /// per-position detail of that pattern (per-position median durations and
    /// most common launch/operator/input shape; the repeating per-layer block
    /// collapsed to one copy). Without it, returns one row per pattern: steps
    /// whose kernel-name sequences are identical, collapsed with a count,
    /// median total duration, and the detected repeat block.
    pub pattern: Option<usize>,
    /// Rows to skip. Defaults to 0.
    #[serde(default)]
    pub offset: usize,
    /// Max rows to return (default 100, hard cap 1000). The header's `total=`
    /// is the full row count, so you know whether to keep paging.
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StageShapesParam {
    /// Path to a `.pt.trace.json` or `.pt.trace.json.gz` file.
    pub path: String,
    /// Which execution stage: `prefill`, `decode`, or `mixed`.
    pub stage: String,
    /// CUDA stream id. Omit to use the stream with the most kernel time; the
    /// output header reports which stream was used (`stream=N(auto)`).
    pub stream: Option<u64>,
    /// Only kernels whose name contains this substring. Defaults to all
    /// kernels of the stage.
    pub name: Option<String>,
    /// Rows to skip over the ranked shape rows. Defaults to 0.
    #[serde(default)]
    pub offset: usize,
    /// Max rows to return (default 100, hard cap 1000). The header's `total=`
    /// is the full row count, so you know whether to keep paging.
    pub limit: Option<usize>,
}

#[tool_router(server_handler)]
impl TraceMcpServer {
    #[tool(
        name = "list_traces",
        description = "List PyTorch profiler traces (*.pt.trace.json.gz) found recursively under the current directory as paths relative to it. Each path is directly reusable as the `path` argument of the other tools."
    )]
    fn list_traces(&self) -> Result<String, String> {
        let names = mcp_core::list_traces_in_dir(".").map_err(|e| e.to_string())?;
        Ok(names.join("\n"))
    }

    #[tool(
        name = "stage_steps",
        description = "TSV list of a stage's (prefill|decode|mixed) execution steps on a stream of a trace: one row per vLLM step annotation with its name, the GPU span of the step's kernels as start/duration (µs), token counters parsed from the annotation name, `kernels` launched under the step's annotation, and `post_kernels` launched after it closed and before the next step started (vLLM launches its sampler and LM head there). Token columns: ctx_tokens/gen_tokens are scheduled query tokens (context/prefill and generation/decode side); on traces captured with detailed_trace_annotation, ctx_kv_tokens/gen_kv_tokens (sum of attended sequence lengths, the KV volume attention reads) and ctx_qk/gen_qk (sum of query_len × seq_len, proportional to attention FLOPs) are populated too, and empty on simple annotations. Membership is by launch (`membership=launch` in the header): a kernel belongs to the step whose CPU-side annotation contains its launch call, so eager steps whose GPU annotation covers only part of their kernels still report all of them; it falls back to the GPU window (`membership=gpu_window`) when the trace has no CPU-side step annotations. First line is `# ... total=N` (number of steps), then a column header; page with offset/limit. Omit `stream` to use the stream with the most kernel time (reported as `stream=N(auto)`). Use the `idx` values as the `step` argument of `step_kernels`."
    )]
    fn stage_steps(&self, Parameters(p): Parameters<StageStepsParam>) -> Result<String, String> {
        let stage = stage_arg(&p.stage)?;
        let (trace, ops) = mcp_core::load_with_ops(&p.path).map_err(|e| e.to_string())?;
        mcp_core::stage_steps_tsv(&trace, &ops, stage, p.stream, p.offset, p.limit)
    }

    #[tool(
        name = "step_kernels",
        description = "TSV of the ordered kernel sequence inside one execution step of a stage (`step` is an `idx` from `stage_steps`). One row per kernel: pos, name, part (`step` when launched under the step's annotation, `post` when launched after it closed and before the next step started — vLLM's sampler and LM head land here), start_offset from step start, duration, gap to the previous kernel's end (µs), grid, block, launch (`op` when launched by a CPU operator, `graph` when replayed from a CUDA/HIP graph, empty when unlinked), and for operator launches the operator and its input_dims, input_types and flops per call (present only in traces captured with record_shapes / with_flops). Rows are in launch order, so consecutive gaps show idle time inside the step and repeating row runs show per-layer structure. First lines are `# ... total=N` (kernels in the step), then a column header; page with offset/limit. Omit `stream` to default to the stream with the most kernel time."
    )]
    fn step_kernels(&self, Parameters(p): Parameters<StepKernelsParam>) -> Result<String, String> {
        let stage = stage_arg(&p.stage)?;
        let (trace, ops) = mcp_core::load_with_ops(&p.path).map_err(|e| e.to_string())?;
        mcp_core::step_kernels_tsv(&trace, &ops, stage, p.stream, p.step, p.offset, p.limit)
    }

    #[tool(
        name = "stage_patterns",
        description = "Condensed view of a stage (prefill|decode|mixed) on a stream. Without `pattern`: TSV with one row per distinct kernel-name sequence across the stage's steps (the per-step sequence includes post-step kernels — those launched after the step's annotation closed and before the next step started) — steps with identical sequences collapsed into one pattern, ranked by total kernel time, with their step count, kernels per step, median total duration, the detected repeating block (`start..end xcount`, typically the per-layer block) and the first step indices. With `pattern=N`: TSV detail of that pattern — one row per sequence position with the median duration and the most common launch/operator/input shape across the pattern's steps (`shape_varies` marks positions whose shape differs between steps), and the repeat block collapsed to a single copy (part=repeat, one layer's kernels, samples = steps × repeats) so one layer can be read without paging through all of them. `#`-prefixed header lines carry the stream used, the membership mode and `total=` row counts."
    )]
    fn stage_patterns(
        &self,
        Parameters(p): Parameters<StagePatternsParam>,
    ) -> Result<String, String> {
        let stage = stage_arg(&p.stage)?;
        let (trace, ops) = mcp_core::load_with_ops(&p.path).map_err(|e| e.to_string())?;
        match p.pattern {
            None => mcp_core::stage_patterns_tsv(&trace, &ops, stage, p.stream, p.offset, p.limit),
            Some(pattern) => mcp_core::stage_pattern_detail_tsv(
                &trace, &ops, stage, p.stream, pattern, p.offset, p.limit,
            ),
        }
    }

    #[tool(
        name = "stage_shapes",
        description = "TSV of the kernel shapes of one stage (prefill|decode|mixed): one row per kernel name × launching operator × input dims × input types, ranked by total GPU time in the stage. Columns: kernel, op, calls, total_us, mean_us, share_pct (share of the stage's kernel time on the stream), input_dims, input_types, flops_per_call (present only in traces captured with record_shapes / with_flops). The same kernel at different shapes (decode vs prefill batch sizes) gets separate rows because it sits at a different point of the roofline. Kernels replayed from a CUDA/HIP graph are aggregated as op `(graph)` so their time share stays visible; their shapes come from the model config and the step's tokens instead. Kernels launched after a step's annotation closed and before the next step started (vLLM's sampler and LM head) are included as that step's post-step work. The `#` header reports coverage: kernels in the stage, how many of them post-step, how many with shapes, graph-replayed and unlinked, plus stage_kernel_us (the share_pct denominator), and total= row count for paging with offset/limit. Filter kernel names with `name`. Omit `stream` to default to the stream with the most kernel time."
    )]
    fn stage_shapes(&self, Parameters(p): Parameters<StageShapesParam>) -> Result<String, String> {
        let stage = stage_arg(&p.stage)?;
        let (trace, ops) = mcp_core::load_with_ops(&p.path).map_err(|e| e.to_string())?;
        mcp_core::stage_shapes_tsv(
            &trace,
            &ops,
            stage,
            p.stream,
            p.name.as_deref().unwrap_or(""),
            p.offset,
            p.limit,
        )
    }
}

/// Runs the MCP server over stdio until the client disconnects.
pub async fn serve() -> Result<()> {
    let service = TraceMcpServer.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
