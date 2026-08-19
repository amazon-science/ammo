---
name: ammo-report
description: Generate publication-quality GPU kernel optimization reports from completed AMMO campaign artifacts. Reads profiling data, candidate evaluations, and implementation results to produce REPORT.md with charts, code listings, and lessons learned.
---

# AMMO Report Writer

Create an artifact-grounded report for engineers deploying vLLM in production. The reader should not need to know AMMO's internal vocabulary.

## Environment (BLOCKING)

Use the prebuilt `.venv`; never install packages or create another environment. If a required import fails, report it to the orchestrator. Reporting requires no GPU access.

## Instructions

Read `.codex/skills/ammo/report/SKILL.md` completely. It is the sole authority for report structure, terminology translation, chart construction, source selection, and quality checks.

The dispatch prompt supplies the artifact directory. Write `{artifact_dir}/REPORT.md`; place chart images and reproducible generation scripts under `{artifact_dir}/report_assets/`.

## Constraints

- Trace every quantitative claim to a primary campaign artifact.
- If evidence is missing, say so; never fabricate a result, method, attribution, or explanation.
- Translate all AMMO-internal terminology using `report/SKILL.md`.
- Describe only profiling, evaluation, implementation, and validation steps documented by the artifacts.
- Chart scripts must read actual artifacts rather than hardcoded values.
- Resolve conflicts in favor of authoritative raw artifacts and `state.json`, and disclose unresolved disagreement.

## Adversarial Review

Codex does not run agent-local Stop hooks from this Markdown, so preserve the review gate explicitly.

After drafting `REPORT.md`, spawn a fresh `ammo-delegate` fact-checker with `fork_turns="none"` and an artifact-only prompt containing only the report and artifact-directory paths. Require an explicit `ok: true` or `ok: false` verdict covering:

1. data traceability for every quantitative claim;
2. jargon compliance with `report/SKILL.md`;
3. shipped candidates, failures, projections, and GATED_PASS descriptions against `state.json` and validation artifacts, including invalid GATED_PASS claims; and
4. invented methodology, causal explanations, or attribution.

Treat the reviewer as independent: do not provide expected answers or ask it to confirm the report. Persist its structured verdict at `{artifact_dir}/report_assets/report_fact_check.json`, including `ok`, `findings`, and `report_sha256`, where the hash is SHA-256 of the exact reviewed `REPORT.md` bytes. If it returns `ok: false`, or the report changes after review, fix the report and dispatch a fresh reviewer. Do not return until the current report has a matching `ok:true` fact-check artifact.

The reviewer exchange and `report_fact_check.json` are the audit record; do not invent a state field or additional sidecar.

## Artifact Metadata

`REPORT.md` is the terminal report artifact. AMMO does not use `.metrics.json` sidecars.
