# AMMO Codex runtime installation

This export is installed in two parts. **Copying only the skill directory is not
runnable**: the AMMO workflow names custom agents and relies on lifecycle hooks,
hook registration, and a trusted state schema that live outside the skill root.

## Install both parts

Let `AMMO_EXPORT` be this export directory and `PROJECT` be the consuming Git
project root.

1. Copy the export's skill content into
   `$PROJECT/.codex/skills/ammo/`. The packaging-only `runtime-overlay/`
   directory does not need to be nested inside the installed skill.
2. Overlay `runtime-overlay/.codex/` onto `$PROJECT/.codex/`, preserving the
   `agents/`, `hooks/`, and `schemas/` subdirectories and `hooks.json`.
3. Start a fresh Codex project session (or reload the runtime) so the new agent
   definitions and hook registration are discovered.

For example:

```sh
mkdir -p "$PROJECT/.codex/skills/ammo" "$PROJECT/.codex"
rsync -a --exclude '/runtime-overlay/' "$AMMO_EXPORT/" "$PROJECT/.codex/skills/ammo/"
cp -a "$AMMO_EXPORT/runtime-overlay/.codex/." "$PROJECT/.codex/"
```

The example uses `rsync` only as an install-time copy tool; an equivalent copy
that omits the packaging-only overlay from the skill destination is fine.

## Runtime configuration

Use Codex **0.153.4 or later**. The AMMO image and generated session home select `gpt-6-astra` with high reasoning effort.
AMMO role TOMLs omit `model` and `model_provider`; they inherit the consuming root's selection and keep their role-specific reasoning effort.
A standalone consumer must select a model available through its own provider. Preserve its credentials and provider settings.

Merge these runtime settings into the configuration used to launch the project:

```toml
[features]
multi_agent = false
hooks = true
shell_tool = true
unified_exec = true

[features.context_management]
experimental_mode = true

[features.multi_agent_v2]
enabled = true
hide_spawn_agent_metadata = false
tool_namespace = "agents"
max_concurrent_threads_per_session = 17

[agents]
job_max_runtime_seconds = 7200
```

Codex enables experimental context management only for eligible ChatGPT accounts/providers. Other authentication paths retain ordinary compaction.
Do not force private token-budget settings or a context-window size. Both paths use the file recovery contract in `references/context-recovery.md`.

Tracked `exec_command` launches and final completion run the Bash hooks. AMMO uses `tty=false`, empty `write_stdin` polling, and owned interruption.
The non-TTY rule is a workflow instruction: the hook payload exposes the command, not the TTY setting.

## Managed server installation

The server image separately installs the complete `.codex` tree at `/opt/codex-managed-hooks` and `requirements.toml` at `/etc/codex/requirements.toml`.
Managed requirements enable `unified_exec` and require managed hooks only. A project setting cannot override a conflicting managed requirement.
Rebuild the image and verify those installed files as well as the project template and generated `CODEX_HOME/config.toml`.

Project hooks alone do not provide the server's root-owned enforcement boundary. A standalone managed installation must use a trusted owner, the complete hook dependency tree, and its actual Python/interpreter paths.
Start a fresh Codex root after changing the runtime. Keep active campaigns on their installed runtime until a separate migration is validated.

## Runtime manifest

The overlay installs:

- 11 custom-agent TOMLs under `.codex/agents/`: `ammo-auditor`,
  `ammo-champion`, `ammo-delegate`, `ammo-impl-champion`, `ammo-implementer`,
  `ammo-investigator`, `ammo-report-writer`, `ammo-report`, `ammo-researcher`,
  `ammo-resolver`, and `ammo-transcript-monitor`.
- Six Python hook modules under `.codex/hooks/`: `common.py`,
  `session_start.py`, `pre_compact.py`, `pre_tool_use_guard.py`,
  `post_tool_use_guard.py`, and `stop_gate_guard.py`.
- `.codex/hooks.json`, which registers `SessionStart`, `PreCompact`,
  `PreToolUse`, `PostToolUse`, `SubagentStart`, `SubagentStop`, and `Stop`.
- `.codex/schemas/state.schema.json`, loaded by the post-tool state guard from
  the schema directory beside the installed hooks.

There are nine canonical role documents in the skill. The extra TOMLs are
compatibility aliases: `ammo-impl-champion` reads
`agents/ammo-implementer.md`, and `ammo-report-writer` reads
`agents/ammo-report.md`. No duplicate alias Markdown files are required.

The hook-to-skill references close only when the two install parts use the
documented layout. In particular, hooks resolve
`.codex/skills/ammo/scripts/ammo_state.py`, `gpu_reservation.py`, and
`monitor_queue_ack.py`, as well as the skill's agent, orchestration, reference,
and report contracts. Moving either tree independently breaks that closure.

## Operational dependencies

- A Codex release that supports project-local skills, `.codex/agents/*.toml`,
  native multi-agent execution, and Codex lifecycle hooks. Native multi-agent
  support must be enabled in the consuming runtime using the configuration
  appropriate to that Codex release.
- A Git project root. Hook root discovery and AMMO worktree operations use
  `git` and project-relative `.codex` paths.
- `/bin/sh`, `/usr/bin/env`, and Python 3. The hook registration invokes the
  scripts through these system tools.
- The Python `jsonschema` package for full `state.schema.json` enforcement.
  Hooks can report that schema validation was skipped when it is unavailable,
  but that is degraded operation rather than semantic parity.
- The installed AMMO skill scripts and their project Python environment. AMMO
  worktree commands expect the worktree virtual environment; GPU benchmarks
  additionally require the consuming project's CUDA, vLLM, and profiling
  stack. Those workload dependencies are not bundled in this overlay.
- Writable campaign artifact/worktree locations and temporary storage. GPU
  sessions should provide their normal `CUDA_VISIBLE_DEVICES`,
  `AMMO_SESSION_ID`, `AMMO_GPU_RES_DIR`, and `AMMO_PROJECT_DIR` integration
  values; the package contains no site-specific values.

For hardened deployments, install `.codex/hooks/` and `.codex/schemas/` as
policy assets that campaign subagents cannot modify. In particular,
`state.schema.json` must remain a regular, non-symlink file beside the hooks.
