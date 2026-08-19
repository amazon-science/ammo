// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0
/**
 * LIGHTGRID Tooltip Registry
 * Central content store for contextual help tooltips.
 * Format: { title: 'ELEMENT_NAME', body: 'Educational description...' }
 */
const LG_TOOLTIP_REGISTRY = {
    // L1 — Campaign Grid
    'l1-session-card': {
        title: 'SESSION_CARD',
        body: 'A single GPU optimization campaign. Shows the model being optimized, current speedup, and pipeline progress. Click to drill into the circuit board view.'
    },
    'l1-speedup': {
        title: 'SPEEDUP_METRIC',
        body: 'Performance improvement of the best optimized kernel compared to the original baseline. Measured as wall-clock latency reduction averaged over 100 runs.'
    },
    'l1-status-active': {
        title: 'ACTIVE_STATUS',
        body: 'This session is actively running optimizations. GPUs are allocated and Claude is working on improving the kernel. The session will continue until paused or terminated.'
    },
    'l1-status-paused': {
        title: 'PAUSED_STATUS',
        body: 'Session is paused. GPUs have been released for other sessions but all optimization state is preserved. Resume anytime to continue where you left off.'
    },
    'l1-status-creating': {
        title: 'CREATING_STATUS',
        body: 'Session is being set up. The workspace, virtual environment, and CLI tool are being configured. This typically takes 10-25 seconds.'
    },
    'l1-model-name': {
        title: 'MODEL_NAME',
        body: 'The HuggingFace model being optimized in this campaign. This is the model whose inference kernels Claude is trying to make faster.'
    },
    'l1-dtype-badge': {
        title: 'DATA_TYPE',
        body: 'The numerical precision used for model inference. Options include fp8, bf16, fp16, int8. Lower precision is faster but may reduce accuracy.'
    },
    'l1-elapsed': {
        title: 'ELAPSED_TIME',
        body: 'Total wall-clock time since this optimization campaign started. Includes all rounds of mining, debate, implementation, and validation.'
    },
    'l1-pipeline-dots': {
        title: 'PIPELINE_PROGRESS',
        body: 'Shows which stages the latest optimization round has completed. Stages flow left to right: Mining \u2192 Debate \u2192 Implement \u2192 Validate \u2192 Integrate.'
    },
    'l1-new-session': {
        title: 'NEW_SESSION',
        body: 'Launch a new optimization campaign. Pick a HuggingFace model, set tensor parallelism and data type, then Claude begins optimizing the inference kernel.'
    },
    'l1-server-gpu': {
        title: 'SERVER_GPU_INFO',
        body: 'Number of GPUs currently available on this server. Each active session uses 1-8 GPUs depending on the tensor parallelism setting.'
    },
    'l1-version-badge': {
        title: 'VERSION_BADGE',
        body: 'Current AMMO version. Click to see the changelog with recent updates and new features.'
    },
    'l1-pause-btn': {
        title: 'PAUSE_SESSION',
        body: 'Pause this campaign. Releases GPUs immediately but preserves all optimization state. The agent stops working. Resume later to continue.'
    },
    'l1-resume-btn': {
        title: 'RESUME_SESSION',
        body: 'Resume this paused campaign. Re-acquires GPUs and restarts the optimization agent from where it left off.'
    },
    'l1-terminal-btn': {
        title: 'OPEN_TERMINAL',
        body: 'Open a live terminal connected to this session. Interact directly with the Claude agent running your optimization. Only available for active sessions.'
    },
    'l1-download-btn': {
        title: 'DOWNLOAD_SESSION',
        body: 'Download all session artifacts as a ZIP archive. Includes optimized kernels, profiling data, logs, and reports.'
    },
    'l1-terminate-btn': {
        title: 'TERMINATE_SESSION',
        body: 'Permanently end this campaign. Cleans up the workspace and releases all resources. This cannot be undone.'
    },
    'l1-paused-section': {
        title: 'PAUSED_SECTION',
        body: 'Collapsed section containing all paused campaigns. Click to expand/collapse. Paused sessions do not count toward your active session limit.'
    },

    // L2 — Circuit Board
    'l2-circuit-board': {
        title: 'CIRCUIT_BOARD',
        body: 'The optimization pipeline visualized as a circuit board. Each column is a stage, each node is a round. Reads left to right \u2014 signals flow from Baseline to Integration.'
    },
    'l2-stage-baseline': {
        title: 'BASELINE_STAGE',
        body: 'Starting point. Measures original kernel performance before any optimization. All speedup calculations compare against this baseline.'
    },
    'l2-stage-mining': {
        title: 'MINING_STAGE',
        body: 'Claude analyzes the kernel source code, profiling data, and hardware specs to identify bottlenecks and optimization opportunities.'
    },
    'l2-stage-debate': {
        title: 'DEBATE_STAGE',
        body: 'Adversarial review of proposed optimizations. An advocate argues FOR, a critic argues AGAINST. Only the strongest ideas survive to implementation.'
    },
    'l2-stage-implement': {
        title: 'IMPLEMENT_STAGE',
        body: 'The optimized kernel code is written and compiled. May include CUDA C++, Triton kernels, fused operations, or multi-kernel strategies.'
    },
    'l2-stage-validate': {
        title: 'VALIDATE_STAGE',
        body: 'Two checks: correctness (does the optimized kernel produce the same output?) and performance (is it actually faster?). Both must pass.'
    },
    'l2-stage-integrate': {
        title: 'INTEGRATE_STAGE',
        body: 'Validated optimizations are merged into the session\'s best kernel. The speedup metric updates to reflect the new champion.'
    },
    'l2-round-node': {
        title: 'ROUND_NODE',
        body: 'One optimization attempt. Green border = succeeded. Pulsing = in progress. Red = failed validation. Click to drill into L3 artifact details.'
    },
    'l2-trace': {
        title: 'CIRCUIT_TRACE',
        body: 'PCB traces connecting rounds across stages. Shows how each optimization attempt flows through the pipeline from mining to integration.'
    },
    'l2-speedup-chart': {
        title: 'SPEEDUP_TREND',
        body: 'Cumulative speedup over optimization rounds. Upward trend = campaign is finding improvements. Plateaus suggest diminishing returns or hitting Amdahl\'s ceiling.'
    },
    'l2-round-card': {
        title: 'ROUND_SUMMARY',
        body: 'Summary of one optimization round: the strategy attempted, whether it passed validation, and the resulting speedup delta (+/- percentage).'
    },
    'l2-session-info': {
        title: 'SESSION_INFO',
        body: 'Metadata for this campaign: model name, configuration, total rounds completed, current best speedup, and session status.'
    },

    // L3 — Artifact Viewer
    'l3-round-badge': {
        title: 'ROUND_BADGE',
        body: 'Identifies which optimization round you are viewing. The round number and optimization strategy name.'
    },
    'l3-baseline-latency': {
        title: 'BASELINE_LATENCY',
        body: 'Original kernel execution time before this round\'s optimization. Measured in milliseconds, averaged over 100 runs with warm-up.'
    },
    'l3-top-component': {
        title: 'TOP_COMPONENT',
        body: 'The most time-consuming operation in the kernel. This is the primary optimization target \u2014 improving it yields the biggest speedup.'
    },
    'l3-fdecode-share': {
        title: 'F_DECODE_SHARE',
        body: 'Percentage of total inference time spent in the decode phase. Higher values mean decode is the bottleneck and the best target for optimization.'
    },
    'l3-amdahl-ceiling': {
        title: 'AMDAHL_CEILING',
        body: 'Theoretical maximum speedup if this component were infinitely fast (Amdahl\'s Law). Sets the upper bound of what any optimization of this component can achieve.'
    },
    'l3-pipeline-viz': {
        title: 'PIPELINE_STAGES',
        body: 'Shows which stages this round completed. Green = passed, red = failed, gray = not reached. Failed stages mean the optimization was rejected.'
    },
    'l3-track-table': {
        title: 'OPTIMIZATION_TRACKS',
        body: 'Detailed table of each optimization track in this round. Shows the approach taken, compilation result, correctness test, and performance change.'
    },
    'l3-artifact-tabs': {
        title: 'ARTIFACT_TABS',
        body: 'Switch between artifacts: source code, compilation logs, profiling data, correctness results, and debate transcripts. Raw output from each stage.'
    },
    'l3-breadcrumb': {
        title: 'NAVIGATION',
        body: 'Breadcrumb navigation. Click "All Campaigns" to return to L1 grid, or the session name to return to L2 circuit board.'
    },
};
