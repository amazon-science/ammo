// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0
/**
 * campaign-app.js — LIGHTGRID Campaign Dashboard Alpine.js component
 *
 * Registers Alpine.data('campaignApp', ...) alongside the existing sessionApp.
 * Manages: campaign data, 3-level hash routing, polling, terminal overlay state.
 *
 * Hash routes:
 *   #campaigns              → L1 overview grid
 *   #campaigns/{id}         → L2 circuit board
 *   #campaigns/{id}/{round}/{node} → L3 artifact viewer
 *
 * Existing routes (#session/{id}, #create, etc.) remain handled by sessionApp.
 */

/* ═══ Tooltip System — showTooltip / hideTooltip ═══ */
let _lgTipEl = null;

function _lgShowTooltip(anchor, entry) {
    if (!_lgTipEl) {
        _lgTipEl = document.createElement('div');
        _lgTipEl.className = 'lg-tip-tooltip';
        _lgTipEl.innerHTML = '<div class="lg-tip-title"></div><div class="lg-tip-body"></div>';
        document.body.appendChild(_lgTipEl);
        _lgTipEl.addEventListener('mouseleave', () => {
            setTimeout(() => {
                if (!_lgTipEl.matches(':hover')) _lgHideTooltip();
            }, 150);
        });
    }
    _lgTipEl.querySelector('.lg-tip-title').textContent = entry.title;
    _lgTipEl.querySelector('.lg-tip-body').textContent = entry.body;

    // Reset for measurement
    _lgTipEl.classList.remove('lg-tip-show');
    _lgTipEl.style.display = 'block';
    _lgTipEl.style.left = '-9999px';
    _lgTipEl.style.top = '0';

    const rect = anchor.getBoundingClientRect();
    const tipW = _lgTipEl.offsetWidth;
    const tipH = _lgTipEl.offsetHeight;

    // Default: position right of anchor
    let left = rect.right + 8;
    let top = rect.top - 4;

    // If off right edge, flip to left of anchor
    if (left + tipW > window.innerWidth - 16) {
        left = rect.left - tipW - 8;
    }
    // If off bottom, shift up
    if (top + tipH > window.innerHeight - 16) {
        top = window.innerHeight - tipH - 16;
    }
    // If off top, shift down
    if (top < 16) top = 16;
    // If off left edge, fall back to below anchor
    if (left < 16) {
        left = Math.max(16, rect.left);
        top = rect.bottom + 8;
    }

    _lgTipEl.style.left = left + 'px';
    _lgTipEl.style.top = top + 'px';
    requestAnimationFrame(() => _lgTipEl.classList.add('lg-tip-show'));
}

function _lgHideTooltip() {
    if (_lgTipEl) _lgTipEl.classList.remove('lg-tip-show');
}

/* ═══ Shared FE helpers (B2/B3/B5/B6) — exported to window.LG_HELPERS
 *     so circuit-board.js (loaded after this file) can use the same code.
 *
 * These replace duplicated inline patterns across campaign-app.js and
 * circuit-board.js. Defined at module scope so they're hoisted before
 * alpine:init fires and before any component method uses them.
 */

// v2 round accessors — mirror circuit-board.js helpers. All per-round/per-stage
// state lives under campaign.rounds[current_round-1]. Past-round access uses
// round.{parallel_tracks.tracks, integration, debate} directly (same shape).
function _currentRoundObj(state) {
    const campaign = (state && state.campaign) || {};
    const rounds = campaign.rounds || [];
    const idx = (campaign.current_round || 1) - 1;
    return (idx >= 0 && idx < rounds.length) ? (rounds[idx] || {}) : {};
}
function currentTracks(state) {
    const pt = _currentRoundObj(state).parallel_tracks || {};
    if (pt.tracks) return pt.tracks;
    return (state && state.parallel_tracks) || {};
}
function currentIntegration(state) {
    return _currentRoundObj(state).integration || {};
}
function currentDebate(state) {
    return _currentRoundObj(state).debate || {};
}
function currentStage(state) {
    return ((state && state.campaign) || {}).current_stage || '';
}

// v1→v2 stage-name → round sub-object key mapping.
const STAGE_KEY_MAP = {
    '1_baseline': 'baseline',
    '2_bottleneck_mining': 'bottleneck_mining',
    '3_debate': 'debate',
    '4_5_parallel_tracks': 'parallel_tracks',
    '6_integration': 'integration',
    '7_campaign_eval': 'campaign_eval',
};

// B2. Round lookup — handles archives that stamp `round_id` OR legacy `round`.
function findRound(rounds, roundId) {
    if (!rounds) return null;
    return rounds.find(r => (r.round_id ?? r.round) === roundId) || null;
}
// B2. Entry-to-round match — unstamped entries (no `round`) fall back to R1.
function matchRound(entry, roundId) {
    return entry.round === roundId || (entry.round == null && roundId === 1);
}

// Canonical workload-bucket identity used by state latency/verdict maps.
// Homogeneous matrices use `bs{BS}`; heterogeneous matrices use the complete
// `il{IL}_ol{OL}_bs{BS}` tuple. Bare numeric keys remain readable for archived
// campaigns, but `tag` always preserves the exact map key so two heterogeneous
// rows with the same BS can never alias each other.
function parseBucketTag(rawTag) {
    if (rawTag == null) return null;
    const tag = String(rawTag);
    let match = /^il(\d+)_ol(\d+)_bs(\d+)$/.exec(tag);
    if (match) {
        const inputLen = Number(match[1]);
        const outputLen = Number(match[2]);
        const batchSize = Number(match[3]);
        if (![inputLen, outputLen, batchSize].every(value => Number.isSafeInteger(value) && value > 0)) return null;
        return {
            tag, inputLen, outputLen, batchSize,
            heterogeneous: true,
            legacyNumeric: false,
            label: `IL${inputLen} \u00b7 OL${outputLen} \u00b7 BS${batchSize}`,
            compactLabel: `IL${inputLen}/OL${outputLen}/BS${batchSize}`,
        };
    }
    match = /^bs(\d+)$/.exec(tag);
    if (!match) match = /^(\d+)$/.exec(tag);
    if (!match) return null;
    const batchSize = Number(match[1]);
    if (!Number.isSafeInteger(batchSize) || batchSize <= 0) return null;
    return {
        tag, inputLen: null, outputLen: null, batchSize,
        heterogeneous: false,
        legacyNumeric: /^\d+$/.test(tag),
        label: `BS${batchSize}`,
        compactLabel: `BS${batchSize}`,
    };
}

function bucketRecords(bucketMap) {
    if (!bucketMap || typeof bucketMap !== 'object') return [];
    return Object.keys(bucketMap)
        .map(parseBucketTag)
        .filter(Boolean)
        .sort((a, b) => {
            const aIl = a.inputLen ?? -1;
            const bIl = b.inputLen ?? -1;
            if (aIl !== bIl) return aIl - bIl;
            const aOl = a.outputLen ?? -1;
            const bOl = b.outputLen ?? -1;
            if (aOl !== bOl) return aOl - bOl;
            if (a.batchSize !== b.batchSize) return a.batchSize - b.batchSize;
            return a.tag < b.tag ? -1 : (a.tag > b.tag ? 1 : 0);
        });
}

// Bug 6: file-type badge for artifact tabs. Returns uppercased extension
// (without the dot) or '' for extensionless files. Examples:
//   'a/b/foo.md'              → 'MD'
//   'rounds/1/diff.patch'     → 'PATCH'
//   'state.json'              → 'JSON'
//   'README'                  → ''
function extBadge(path) {
    const m = (path || '').match(/\.([a-z0-9]+)$/i);
    return m ? m[1].toUpperCase() : '';
}

/** Build an artifact URL without allowing path metacharacters to alter it.
 * Each path segment is encoded independently so directory separators remain
 * readable/routable while `#`, `?`, `%`, spaces, and unicode stay data. */
function buildArtifactUrl(sessionId, path) {
    const sid = encodeURIComponent(String(sessionId || ''));
    const encodedPath = String(path || '')
        .split('/')
        .map(segment => encodeURIComponent(segment))
        .join('/');
    return sid ? `/api/campaigns/${sid}/artifacts/${encodedPath}` : encodedPath;
}

// C3. Materialize catalog.entries once per catalog version (revised 3× —
//     DA-round3 H3). Cache key is (last_updated, last_scan_file_count) —
//     aggregator stamps both at write time (campaign_data_service.py:232-233).
//     Returns the SAME array reference on cache hit so downstream memos
//     (_trackOverviewData and other L2 consumers) can rely on identity-based
//     invalidation.
//
//     Defensive fallback: if the caller's catalog object identity changes
//     WITHOUT a (last_updated, last_scan_file_count) bump (schema violation
//     from the BE side), we log once and fall through to re-materialize, so
//     the FE is never stuck showing stale entries.
// Collapse `a/b/../c` → `a/c` and `./` segments. Mirrors what
// `campaign_data_service.read_artifact` does server-side (via `os.path.realpath`
// + a startswith guard), but we normalize client-side too so paths stay clean
// in tooltips, debug logs, and any consumer doing a strict string compare.
function normalizeArtifactPath(p) {
    if (typeof p !== 'string' || !p) return p;
    const parts = p.split('/');
    const out = [];
    for (const seg of parts) {
        if (seg === '..') {
            if (out.length && out[out.length - 1] !== '..') out.pop();
            else out.push(seg);
        } else if (seg !== '.') {
            out.push(seg);
        }
    }
    return out.join('/');
}

// Resolve a `refs[].path` stored inside a `*.metrics.json` sidecar into the
// real on-disk path the /api/campaigns/{id}/artifacts endpoint can serve.
//
// Two emitter conventions are in the wild:
//   • Legacy sidecar-dir-relative (`./foo.json`, `../foo.json`, `../../x.md`):
//     resolve against the sidecar's own directory, then normalize.
//   • New artifact-dir-relative (bare path, no `./` / `../` prefix, e.g.
//     `e2e_latency/e2e_latency_results.json`): pass through as-is. This is
//     what `run_vllm_bench_latency_sweep.py` now emits via `_rel(p)`.
//
// Keeping both behaviours side-by-side means old catalogs keep resolving and
// new sweep-emitted sidecars stop producing nested wrong paths like
// `e2e_latency/json/e2e_latency/e2e_latency_results.json`.
function resolveRefPath(sidecarKey, refPath) {
    if (typeof refPath !== 'string' || !refPath) return refPath;
    if (typeof sidecarKey !== 'string' || !sidecarKey.endsWith('.metrics.json')) return refPath;
    if (refPath.startsWith('./') || refPath.startsWith('../')) {
        const baseDir = sidecarKey.substring(0, sidecarKey.lastIndexOf('/') + 1);
        return normalizeArtifactPath(baseDir + refPath);
    }
    return refPath;
}

const _CATALOG_ENTRIES_MEMO = { catalog: null, key: null, entries: null, warnedOnce: false };

// Build a stable cache key for the materialized entries memo.
//
// Sidecar removal (2026-05-27): the catalog input is now the tree-endpoint
// response (`{root, files: [path, ...]}`). The memo key tracks file count;
// adding/removing a file invalidates the cache.
function buildCatalogEntriesKey(tree) {
    if (!tree) return 'null';
    if (Array.isArray(tree.files)) return `tree|${tree.files.length}`;
    if (Array.isArray(tree)) return `arr|${tree.length}`;
    if (typeof tree === 'object') return `flat|${Object.keys(tree).length}`;
    return 'null';
}

// B3 → C3. Normalize catalog.entries into an array, preserving path from the
// key. Memoized by catalog version: consecutive calls with the same catalog
// return the same array reference.
//
// Path-derived round override: when the resolved on-disk path contains a
// `round_N/` or `campaign_round_N/` directory segment, we use N as the round
// regardless of what the indexer stamped. The on-disk path is ground truth;
// the stamped `round` field is a snapshot of `campaign.current_round` at
// index time and can drift when the orchestrator indexes older artifacts.
// Only overrides when the extracted round is a positive integer; entries
// without a round segment in the path keep their stamped value (or null).
//
// V2 plural form: `rounds/{N}/...` is the new layout. Capture group [1] is
// for legacy `(campaign_)?round_N`, [2] is for v2 `rounds/N`. Use
// `m[1] ?? m[2]` so both forms map to a positive integer.
const _ROUND_SEG_RE = /(?:^|\/)(?:(?:campaign_)?round_(\d+)|rounds\/(\d+))(?:\/|$)/;
function _deriveRoundFromPath(path) {
    if (typeof path !== 'string' || !path) return null;
    const m = path.match(_ROUND_SEG_RE);
    if (!m) return null;
    const n = parseInt(m[1] ?? m[2], 10);
    return Number.isFinite(n) && n > 0 ? n : null;
}

// Task 1a (artifact-layout-v2-frontend). Parse a v2 artifact path into
// `{round, stage, track_id, kind, op_id, champion_id, stance}`. Single
// source of truth replacing scattered `e.stage === ...` filters and
// `e.labels.*` lookups that previously came from sidecars.
//
// V2 layout: artifacts live under `rounds/{N}/<stage-dir>/...`. We derive
// stage from the segment immediately following `rounds/{N}/`. For legacy
// flat paths (no `rounds/` prefix), we fall back to historical heuristics
// so v1 campaigns keep rendering correctly during migration.
//
// Sidecar removal (2026-05-27): `kind`, `op_id`, `champion_id`, `stance`
// used to come from `.metrics.json` sidecars. The sidecar layer was
// deleted; these labels are now derived purely from the artifact path
// per `references/artifact-layout.md` conventions. Path → kind table:
//   debate/proposals/<champ>_proposal.md            → kind=proposal
//   debate/round_*/<champ>_argument*.md             → kind=debate_rationale, stance=argument
//   debate/round_*/<champ>_critique_*.md            → kind=debate_rationale, stance=critique
//   debate/round_*/<champ>_rebuttal*.md             → kind=debate_rationale, stance=rebuttal
//   debate/summary.md                                → kind=debate_summary
//   tracks/{op}/validator_tests/gate_5_1a_results.json → kind=gate_result
//   tracks/{op}/validator_tests/gate_5_2_results.json  → kind=gate_result
//   tracks/{op}/validation_results.md                → kind=validation_results
//   tracks/{op}/diff.patch | shipped_code.py        → kind=diff or kind=source_code
//   sweeps/*/e2e_latency_results*.json              → kind=e2e_latency
//   sweeps/*/json/golden_refs.json                  → kind=golden_refs
//   profiling/nsys/*.nsys-rep                       → kind=nsys_trace
//   mining/bottleneck_analysis.md                   → kind=bottleneck_analysis
//   audits/stage_*.md                                → kind=audit_verdict
//   integration/diff*.patch                          → kind=diff (no op_id)
//   REPORT.md and report/sections/*.md               → kind=report_section
//
// Stage carve-outs:
//   - `tracks/{op_id}/validator_tests/*` → 'validation'
//   - `_archive/` → stage=null (caller filters these out from views).
//   - `debate/round_{D}/` is a debate sub-round, NOT a campaign round.
//     Stage stays 'debate', round comes from the OUTER `rounds/{N}/`.
function _deriveDebateLabels(filename) {
    // filename = last segment of a debate path. Returns {kind, championId, stance}.
    // Champion id pattern matches `champion-1`, `champion_proj`, `oproj`, etc.
    // up to but not including the next stance suffix or extension.
    const out = { kind: null, championId: null, stance: null };
    if (!filename) return out;
    const base = filename.replace(/\.[^.]+$/, ''); // strip extension
    if (base === 'summary') {
        out.kind = 'debate_summary';
        return out;
    }
    // Stance suffix detection. The filename pattern is
    //   <champion_id>_<stance>[_<extra>][.<ext>]
    // where <stance> ∈ {argument, critique, rebuttal, proposal}. The
    // extra suffix (e.g. critique_round_2) gets folded into the stance.
    let stanceMatch = base.match(/_(argument|critique|rebuttal|proposal)(?:_.*)?$/);
    if (stanceMatch) {
        const stance = stanceMatch[1];
        out.stance = stance;
        out.championId = base.slice(0, base.length - stanceMatch[0].length) || null;
        out.kind = stance === 'proposal' ? 'proposal' : 'debate_rationale';
        return out;
    }
    // No recognised stance — default to a generic debate artifact.
    out.kind = 'debate_other';
    out.championId = base;
    return out;
}

function parseArtifactPath(path) {
    const empty = { round: null, stage: null, track_id: null, kind: null, op_id: null, champion_id: null, stance: null };
    if (typeof path !== 'string' || !path) return empty;

    const fileName = path.split('/').pop() || '';

    // V2 prefix: `rounds/{N}/<rest>`
    const v2 = path.match(/^rounds\/(\d+)\/(.+)$/);
    if (v2) {
        const round = parseInt(v2[1], 10);
        const rest = v2[2];
        const out = { round, stage: null, track_id: null, kind: null, op_id: null, champion_id: null, stance: null };

        // Excluded from views entirely.
        if (rest.startsWith('_archive/')) {
            return out;
        }

        // tracks/{op_id}/...: validator_tests/ carve-out gives 'validation';
        // anything else under the track is 'implementation'.
        let m = rest.match(/^tracks\/([^/]+)(?:\/(.*))?$/);
        if (m) {
            const trackId = m[1];
            const sub = m[2] || '';
            out.track_id = trackId;
            out.op_id = trackId;
            out.stage = sub.startsWith('validator_tests/') ? 'validation' : 'implementation';
            if (/gate_5_1a_results\.json$|gate_5_2_results\.json$/.test(sub)) {
                out.kind = 'gate_result';
            } else if (sub.startsWith('validator_tests/') && /\.(py|md)$/.test(sub)) {
                out.kind = 'validator_script';
            } else if (/\bvalidation_results\.md$/.test(sub)) {
                out.kind = 'validation_results';
            } else if (/\.(patch|diff)$/.test(sub) || sub.endsWith('diff.patch')) {
                out.kind = 'diff';
            } else if (sub.startsWith('monitor_audits/')) {
                out.kind = 'monitor_audit';
            } else if (/shipped_code\.|kernel\.|\.cu$|\.cuh$|\.py$/.test(sub)) {
                out.kind = 'source_code';
            }
            return out;
        }

        // sweeps/opt/{op_id}/... → implementation, track_id from op_id
        m = rest.match(/^sweeps\/opt\/([^/]+)\/(.*)$/);
        if (m) {
            out.track_id = m[1];
            out.op_id = m[1];
            out.stage = 'implementation';
            const sub = m[2] || '';
            if (/e2e_latency_results.*\.json$/.test(sub)) out.kind = 'e2e_latency';
            else if (/golden_refs\.json$/.test(sub)) out.kind = 'golden_refs';
            else if (/\.nsys-rep$/.test(sub)) out.kind = 'nsys_trace';
            return out;
        }

        // Other sweeps subdirs.
        if (rest.startsWith('sweeps/baseline/') || rest.startsWith('sweeps/golden_capture/')) {
            out.stage = 'baseline';
            if (/e2e_latency_results.*\.json$/.test(rest)) out.kind = 'e2e_latency';
            else if (/golden_refs\.json$/.test(rest)) out.kind = 'golden_refs';
            else if (/\.nsys-rep$/.test(rest)) out.kind = 'nsys_trace';
            return out;
        }
        if (rest.startsWith('sweeps/integration/')) {
            out.stage = 'integration';
            if (/e2e_latency_results.*\.json$/.test(rest)) out.kind = 'e2e_latency';
            else if (/golden_refs\.json$/.test(rest)) out.kind = 'golden_refs';
            else if (/\.nsys-rep$/.test(rest)) out.kind = 'nsys_trace';
            return out;
        }

        // Stage directories.
        if (rest.startsWith('profiling/')) {
            out.stage = 'baseline';
            if (/\.nsys-rep$/.test(rest)) out.kind = 'nsys_trace';
            else if (/\.ncu-rep$/.test(rest)) out.kind = 'ncu_trace';
            return out;
        }
        if (rest.startsWith('mining/')) {
            out.stage = 'mining';
            if (/bottleneck_analysis\.md$/.test(rest)) out.kind = 'bottleneck_analysis';
            return out;
        }
        // debate/... — covers debate/proposals/, debate/round_{D}/, debate/summary.md
        // The campaign round is the outer rounds/{N}/; the inner round_{D}/
        // (when present) is a debate sub-round, NOT a campaign round.
        if (rest.startsWith('debate/') || rest === 'debate') {
            out.stage = 'debate';
            if (rest === 'debate/summary.md' || rest.endsWith('/summary.md')) {
                out.kind = 'debate_summary';
                return out;
            }
            // proposals/<champion>_proposal.md → kind=proposal
            const propMatch = rest.match(/^debate\/proposals\/(.+)$/);
            if (propMatch) {
                const labels = _deriveDebateLabels(fileName);
                out.kind = labels.kind || 'proposal';
                out.champion_id = labels.championId;
                out.stance = labels.stance || 'proposal';
                if (out.champion_id) out.op_id = out.champion_id;
                return out;
            }
            // debate/round_{D}/<champion>_<stance>*.md → kind=debate_rationale
            const rndMatch = rest.match(/^debate\/round_\d+\/(.+)$/);
            if (rndMatch) {
                const labels = _deriveDebateLabels(fileName);
                out.kind = labels.kind || 'debate_rationale';
                out.champion_id = labels.championId;
                out.stance = labels.stance;
                if (out.champion_id) out.op_id = out.champion_id;
                return out;
            }
            // monitor_audits/, micro_experiments/, etc.
            if (rest.startsWith('debate/monitor_audits/')) { out.kind = 'monitor_audit'; return out; }
            if (rest.startsWith('debate/micro_experiments/')) { out.kind = 'micro_experiment'; return out; }
            // Bare debate/<file> — treat as misc debate artifact.
            return out;
        }

        // audits/stage_NN[...].
        m = rest.match(/^audits\/stage_(\d+)/);
        if (m) {
            const stageNum = m[1];
            const first = stageNum[0];
            if (first === '1') out.stage = 'baseline';
            else if (first === '4' || first === '5') out.stage = 'implementation';
            else if (first === '6' || first === '7') out.stage = 'integration';
            out.kind = 'audit_verdict';
            return out;
        }

        // integration/diff*.patch → kind=diff with no op_id.
        if (rest.startsWith('integration/')) {
            out.stage = 'integration';
            if (/\.(patch|diff)$/.test(rest)) out.kind = 'diff';
            return out;
        }

        // Top-level REPORT or report sections.
        if (rest === 'REPORT.md' || rest.startsWith('report/')) {
            out.stage = 'integration';
            out.kind = 'report_section';
            return out;
        }

        // round-scoped constraints.
        if (rest === 'constraints.md' || rest.endsWith('/constraints.md')) {
            out.stage = 'baseline';
            return out;
        }

        return out;
    }

    // Legacy flat paths (no `rounds/` prefix). Fall back to historical
    // heuristics keyed off the path prefix.
    const round = _deriveRoundFromPath(path);
    const out = { round, stage: null, track_id: null, kind: null, op_id: null, champion_id: null, stance: null };

    // tracks/{op_id}/... — same validator_tests carve-out.
    let m = path.match(/^tracks\/([^/]+)(?:\/(.*))?$/);
    if (m) {
        const trackId = m[1];
        const sub = m[2] || '';
        out.track_id = trackId;
        out.op_id = trackId;
        out.stage = sub.startsWith('validator_tests/') ? 'validation' : 'implementation';
        if (/gate_5_1a_results\.json$|gate_5_2_results\.json$/.test(sub)) out.kind = 'gate_result';
        else if (/\bvalidation_results\.md$/.test(sub)) out.kind = 'validation_results';
        return out;
    }

    if (path.startsWith('bottleneck_analysis')) { out.stage = 'mining'; out.kind = 'bottleneck_analysis'; return out; }
    if (path === 'REPORT.md') { out.stage = 'integration'; out.kind = 'report_section'; return out; }
    if (path.startsWith('debate/') || path === 'debate') {
        out.stage = 'debate';
        if (path.endsWith('/summary.md') || path === 'debate/summary.md') { out.kind = 'debate_summary'; return out; }
        const labels = _deriveDebateLabels(fileName);
        if (labels.kind) {
            out.kind = labels.kind;
            out.champion_id = labels.championId;
            out.stance = labels.stance;
        }
        return out;
    }
    if (path.startsWith('e2e_latency')) { out.stage = 'baseline'; out.kind = 'e2e_latency'; return out; }
    if (path.startsWith('ncu/')) { out.stage = 'baseline'; out.kind = 'ncu_trace'; return out; }
    if (path.startsWith('nsys/')) { out.stage = 'baseline'; out.kind = 'nsys_trace'; return out; }
    if (path === 'constraints.md' || path.endsWith('/constraints.md')) { out.stage = 'baseline'; return out; }
    if (path.startsWith('monitor_log_')) { out.stage = 'debate'; out.kind = 'monitor_audit'; return out; }

    return out;
}

// Task 1c. Port of CampaignDataService._build_pipeline_progress (Python).
// Maps campaign.current_stage to a 6-element pipeline progress array used by
// L1 cards. Terminal detection uses campaign.status (campaign_complete /
// campaign_exhausted), not current_stage.
const _PIPELINE_STAGES = ['baseline', 'mining', 'debate', 'implementation', 'validation', 'integration'];
const _PIPELINE_STAGE_ORDER = {
    '1_baseline': 0,
    '2_bottleneck_mining': 1,
    '3_debate': 2,
    '4_5_parallel_tracks': 3,
    '6_integration': 5,
    '7_campaign_eval': 6,
    '7b_report': 7,
};
function _buildPipelineProgress(campaign) {
    const c = campaign || {};
    const currentStageStr = c.current_stage || '';
    const currentIdx = (currentStageStr in _PIPELINE_STAGE_ORDER)
        ? _PIPELINE_STAGE_ORDER[currentStageStr]
        : -1;
    const status = c.status || '';
    const terminal = status === 'campaign_complete' ||
        status === 'campaign_exhausted' ||
        currentStageStr === 'campaign_complete';
    const progress = [];
    for (let i = 0; i < _PIPELINE_STAGES.length; i++) {
        const name = _PIPELINE_STAGES[i];
        let s;
        if (terminal || currentIdx >= 6) {
            s = 'completed';
        } else if (i < currentIdx) {
            s = 'completed';
        } else if (i === currentIdx || (i === 4 && currentIdx === 3)) {
            // i==4 is 'validation'; auto-activates alongside implementation
            // when current_stage is 4_5_parallel_tracks.
            s = 'active';
        } else {
            s = 'pending';
        }
        progress.push({ stage: name, status: s });
    }
    return progress;
}

// Task 1d. Standalone version of `_countAllTrackStatuses` (instance method)
// that accepts state as an arg. Works with the L1 trimmed projection
// (only {status, verdict, kernel_speedup, classification, fail_reason} per
// track). Internally calls `trackStatus()` (defined later) which reads
// verdict/status + the shippedSet.
function _countTrackStatuses(state) {
    const empty = { shipped: 0, failed: 0, active: 0 };
    if (!state || typeof state !== 'object') return empty;
    const campaign = state.campaign || {};
    const shippedList = campaign.shipped_optimizations || [];
    const shippedSet = new Set();
    for (const entry of shippedList) {
        if (!entry) continue;
        if (typeof entry === 'string') shippedSet.add(entry);
        else if (typeof entry === 'object') {
            const id = entry.op_id || entry.opId;
            if (id) shippedSet.add(String(id));
        }
    }
    const rounds = campaign.rounds || [];
    const currentRoundId = campaign.current_round || 1;
    const counted = new Set();
    let shipped = 0, failed = 0, active = 0;
    const bucketOf = (s) => {
        if (s === 'shipped' || s === 'gated' || s === 'validated') return 'shipped';
        if (s === 'failed' || s === 'blocked') return 'failed';
        return 'active';
    };
    for (const rnd of rounds) {
        if (!rnd || typeof rnd !== 'object') continue;
        for (const opId of (rnd.shipped || [])) {
            if (!counted.has(opId)) { shipped++; counted.add(opId); }
        }
        const tracks = (rnd.parallel_tracks && rnd.parallel_tracks.tracks) || {};
        const isCurrent = rnd.round_id === currentRoundId;
        for (const [opId, track] of Object.entries(tracks)) {
            if (counted.has(opId)) continue;
            const ts = trackStatus({ ...(track || {}), op_id: opId }, shippedSet);
            const b = bucketOf(ts);
            if (b === 'shipped') { shipped++; counted.add(opId); }
            else if (b === 'failed') { failed++; counted.add(opId); }
            else if (isCurrent) { active++; counted.add(opId); }
            // past-round non-terminal tracks are NOT counted
        }
    }
    return { shipped, failed, active };
}

// Detect whether the artifact source belongs to the v2 layout. The signal
// is purely structural: at least one path under `rounds/<digit>/`. Used by
// feature gates that should only fire under the new layout. Accepts:
//   - tree response { root, files: [path, ...] } (current)
//   - legacy sidecar dict (path → entry)
//   - legacy { entries: { ... } } wrap
// Returns false for empty / null inputs so the legacy rendering path stays
// the default.
function _isV2Layout(source) {
    if (!source || typeof source !== 'object') return false;
    let paths;
    if (Array.isArray(source.files)) {
        paths = source.files;
    } else if (Array.isArray(source)) {
        paths = source;
    } else {
        const dict = (source.entries && typeof source.entries === 'object')
            ? source.entries
            : source;
        if (!dict || typeof dict !== 'object') return false;
        paths = Object.keys(dict);
    }
    for (const k of paths) {
        if (typeof k === 'string' && /^rounds\/\d+\//.test(k)) return true;
    }
    return false;
}

// Task 5. Port of server's `_normalize_speedup_field`. Mutates the campaign
// object in place: copies `cumulative_speedup_vs_round1` (v3 field name) →
// `cumulative_e2e_speedup` (legacy field that the FE reads) when the legacy
// field is absent or at its 1.0 sentinel value. Idempotent.
function _normalizeCumulativeSpeedup(campaign) {
    if (!campaign || typeof campaign !== 'object') return;
    const v3 = campaign.cumulative_speedup_vs_round1;
    if (v3 == null || v3 === 1.0) return;
    const current = campaign.cumulative_e2e_speedup;
    if (current == null || current === 1.0) {
        campaign.cumulative_e2e_speedup = v3;
    }
}

// Sidecar removal (2026-05-27): `_catalogEntries` now consumes the
// `/api/campaigns/{id}/tree` response (`{root, files: [path, ...]}`) and
// synthesizes entries entirely from path conventions via
// `parseArtifactPath`. The shape preserved for downstream consumers is:
//   { path, round, _parsed, _stage, labels: {kind, op_id, champion_id, stance} }
// `labels` is a derived view (NOT server-emitted) so existing filters
// like `e.labels.kind === 'debate_rationale'` keep working unchanged.
//
// Backward-compat: callers may still pass the legacy sidecar dict; it is
// silently coerced to a list of paths and fed through the same pipeline.
// Empty or null input → empty array.
function _catalogEntries(tree) {
    if (!tree) return [];
    // Tree endpoint response: { root, files: [path, ...] }
    let paths;
    if (Array.isArray(tree.files)) {
        paths = tree.files;
    } else if (Array.isArray(tree)) {
        paths = tree;
    } else if (typeof tree === 'object') {
        // Legacy sidecar dict — keys are paths.
        paths = Object.keys(tree);
    } else {
        return [];
    }
    if (!paths.length) return [];

    // Memoise on the array reference so successive Alpine `x-effect` reads
    // return the same materialised list.
    if (_CATALOG_ENTRIES_MEMO.catalog === tree && _CATALOG_ENTRIES_MEMO.entries) {
        return _CATALOG_ENTRIES_MEMO.entries;
    }

    const hiddenArtifactSegments = new Set(['cache', 'triton_cache', 'torch_compile_cache']);
    const arr = paths
        // Drop noise/internal paths early.
        .filter(p => typeof p === 'string' && p && !p.startsWith('_archive/') &&
                     !p.includes('/__pycache__/') && !p.endsWith('.metrics.json') &&
                     !p.split('/').some(segment => hiddenArtifactSegments.has(segment)))
        .map(path => {
            const parsed = parseArtifactPath(path);
            const derivedRound = _deriveRoundFromPath(path) ?? parsed.round ?? 1;
            // Build a labels view that mirrors the old sidecar contract so
            // call sites keying off `entry.labels.kind` keep working.
            const labels = {
                kind: parsed.kind || null,
                op_id: parsed.op_id || null,
                champion_id: parsed.champion_id || null,
                stance: parsed.stance || null,
            };
            return {
                path,
                round: derivedRound,
                _parsed: parsed,
                _stage: parsed.stage || null,
                labels,
                track_id: parsed.track_id || null,
            };
        });
    _CATALOG_ENTRIES_MEMO.catalog = tree;
    _CATALOG_ENTRIES_MEMO.key = `n${arr.length}`;
    _CATALOG_ENTRIES_MEMO.entries = arr;
    return arr;
}

// C1. Cache key for _trackOverviewData() — see plan section C1.
// Invalidates on catalog update, node change, round change, and any
// parallel_tracks[*] value change (status/verdict/kernel_speedup).
// When state._etag or state._version is provided by BE, uses that as a
// cheap short-circuit; otherwise falls back to a compact track signature.
//
// Sidecar removal: tree endpoint returns `{root, files: [...]}`. Sig =
// file count. Adding/removing a file invalidates downstream memos.
function _catalogSig(tree) {
    if (!tree) return '';
    if (Array.isArray(tree.files)) return `n${tree.files.length}`;
    if (Array.isArray(tree)) return `n${tree.length}`;
    if (typeof tree === 'object') return `n${Object.keys(tree).length}`;
    return '';
}
function buildTrackOverviewKey(catalog, currentNode, currentRound, state) {
    const catU = _catalogSig(catalog);
    const node = currentNode == null ? '' : String(currentNode);
    const rd = currentRound == null ? '' : String(currentRound);
    if (!state) return `${catU}:${node}:${rd}:nostate`;
    if (state._etag) return `${catU}:${node}:${rd}:e:${state._etag}`;
    if (state._version) return `${catU}:${node}:${rd}:v:${state._version}`;
    const pt = currentTracks(state);
    const sig = Object.entries(pt)
        .map(([k, t]) => {
            const tt = t || {};
            return `${k}#${tt.status ?? ''}#${tt.verdict ?? ''}#${tt.kernel_speedup ?? ''}#${tt.kernel_speedup_cold ?? ''}#${tt.kernel_speedup_warm ?? ''}`;
        })
        .join('|');
    return `${catU}:${node}:${rd}:s:${currentStage(state)}:pt:${sig}`;
}

// C4. Schwartzian transform over champion-keyed entries — pre-compute the
// parsed numeric key ONCE per entry instead of twice per comparator call.
// Accepts array of [championId, value] tuples (as produced by Map.entries()).
// Matches the historical default (['999']) when no digit is found, so the
// ordering of champion-N entries is unchanged vs. the inline pattern at
// campaign-app.js:2871-2877.
function sortChampionEntries(entries) {
    return (entries || [])
        .map(([k, v]) => [parseInt((String(k).match(/\d+/) || ['999'])[0], 10), k, v])
        .sort((a, b) => a[0] - b[0])
        .map(t => [t[1], t[2]]);
}

// B5. Canonical 5-way verdict ladder for a single track.
// Precedence (top wins, first true verdict determines result):
//   1. GATED     — verdict is GATED_PASS / GATED-PASS (kernel-only pass — amber
//                  stays visible even when op later lands in shipped_optimizations)
//   2. SHIPPED   — op_id present in shippedSet
//   3. FAILED    — verdict/status is FAIL / FAILED
//   4. VALIDATED — verdict is PASS / PASSED (waiting to ship)
//   5. IN_PROGRESS — track has any non-terminal status
//   6. PENDING   — track exists but has no status/verdict field at all
//   7. UNKNOWN   — fallback (track missing)
// Callers pass the normalized shippedSet built from B1's shipped_optimizations
// helper. Returns a lowercase token compatible with CB.STATUS_COLOR.
function trackStatus(track, shippedSet) {
    if (!track) return 'unknown';
    const opId = track.op_id || track.opId || track.name || null;
    const verdict = String(track.verdict || track.status || '').toUpperCase();
    const isShipped = !!(shippedSet && opId && (
        shippedSet.has(opId) ||
        shippedSet.has(String(opId).toUpperCase()) ||
        shippedSet.has(String(opId).toLowerCase())
    ));
    if (isShipped) return 'shipped';
    if (verdict === 'GATED_PASS' || verdict === 'GATED-PASS') return 'validated';
    if (verdict === 'GPU_BLOCKED') return 'blocked';
    if (verdict === 'GATING_REQUIRED') return 'gating';
    if (verdict === 'FAIL' || verdict === 'FAILED') return 'failed';
    if (verdict === 'PASS' || verdict === 'PASSED') return 'validated';
    if (verdict) return 'in_progress';
    return 'pending';
}

// B6. Case-insensitive map lookup — tries as-is, lower, upper in order.
function lookupOp(map, op) {
    if (!map || op == null) return undefined;
    const s = String(op);
    return map[s] ?? map[s.toLowerCase()] ?? map[s.toUpperCase()];
}

if (typeof window !== 'undefined') {
    window.LG_HELPERS = {
        findRound, matchRound, _catalogEntries, trackStatus, lookupOp,
        parseBucketTag, bucketRecords,
        // Artifact badges and safe URLs.
        extBadge, buildArtifactUrl,
        // C1-C4 cache-key builders + Schwartzian sort (exposed for unit tests
        // and for potential reuse by circuit-board.js).
        buildTrackOverviewKey, buildCatalogEntriesKey,
        sortChampionEntries,
        // Shared ref-path resolver — circuit-board.js `_sidecarPath` uses this
        // too so both catalog paths stay consistent.
        resolveRefPath,
        // v2 round accessors — also exposed for Alpine x-text expressions.
        currentTracks, currentIntegration, currentDebate, currentStage,
        STAGE_KEY_MAP,
        // Artifact Layout V2 (Task 1a-1d, 5, plus _isV2Layout detector).
        parseArtifactPath, _deriveRoundFromPath,
        _buildPipelineProgress, _countTrackStatuses,
        _normalizeCumulativeSpeedup, _isV2Layout,
    };
}

document.addEventListener('alpine:init', () => {

    /* ═══ x-tooltip directive ═══ */
    Alpine.directive('tooltip', (el, { expression }, { cleanup }) => {
        const key = expression;
        const entry = (typeof LG_TOOLTIP_REGISTRY !== 'undefined') ? LG_TOOLTIP_REGISTRY[key] : null;
        if (!entry) return;

        const icon = document.createElement('span');
        icon.className = 'lg-tip-icon';
        icon.textContent = '\u24d8';
        icon.setAttribute('data-tip-key', key);

        // Ensure parent is positioned for absolute icon
        const computed = getComputedStyle(el);
        if (computed.position === 'static') {
            el.style.position = 'relative';
        }
        el.appendChild(icon);

        // Show icon on element hover
        const onEnter = () => icon.classList.add('lg-tip-visible');
        const onLeave = (e) => {
            if (!e.relatedTarget || !el.contains(e.relatedTarget)) {
                setTimeout(() => {
                    if (!icon.matches(':hover') && !document.querySelector('.lg-tip-tooltip:hover')) {
                        icon.classList.remove('lg-tip-visible');
                        _lgHideTooltip();
                    }
                }, 150);
            }
        };
        el.addEventListener('mouseenter', onEnter);
        el.addEventListener('mouseleave', onLeave);

        // Show tooltip on icon hover
        const iconEnter = () => _lgShowTooltip(icon, entry);
        const iconLeave = () => {
            setTimeout(() => {
                if (!document.querySelector('.lg-tip-tooltip:hover')) {
                    _lgHideTooltip();
                }
            }, 150);
        };
        icon.addEventListener('mouseenter', iconEnter);
        icon.addEventListener('mouseleave', iconLeave);

        cleanup(() => {
            el.removeEventListener('mouseenter', onEnter);
            el.removeEventListener('mouseleave', onLeave);
            icon.removeEventListener('mouseenter', iconEnter);
            icon.removeEventListener('mouseleave', iconLeave);
            icon.remove();
        });
    });

    // Expected catalog schema version. Sidecars carry envelope + optional
    // payload; label-only entries (no metrics) are ignored by metric chips.
    const _CATALOG_SCHEMA_VERSION = 1;

    // B4. Unified fail-reason accessor. BE walker (campaign_data_service.py
    // _normalize_fail_reasons) copies failure_reason/reason → fail_reason on
    // every state read, so the canonical field is the first lookup. Remaining
    // fallbacks guard against archived states written before the walker landed.
    function _failReason(impl, rootTrack) {
        return (impl && impl.fail_reason)
            ?? (rootTrack && rootTrack.fail_reason)
            ?? (impl && impl.failure_reason)
            ?? (rootTrack && rootTrack.failure_reason)
            ?? null;
    }

    // B1. Normalize campaign.shipped_optimizations into a Set<string> of op_ids.
    // Accepts both legacy list[str] and canonical list[dict] shapes
    // (dicts may use `op_id` or `opId`). Casing is preserved verbatim — live
    // data has both UPPER-DASH (Nemotron) and lowercase-snake (DeepSeek)
    // and case-folding would break one or the other. Case-insensitive
    // fallback lookups belong to lookupOp (B6), not here.
    function _normalizeShippedOps(campaign) {
        const list = (campaign && campaign.shipped_optimizations) || [];
        const set = new Set();
        for (const entry of list) {
            if (!entry) continue;
            if (typeof entry === 'string') {
                set.add(entry);
            } else if (typeof entry === 'object') {
                const id = entry.op_id || entry.opId;
                if (id) set.add(String(id));
            }
        }
        return set;
    }

    // Augment window.LG_HELPERS (B2/B3/B5/B6 registered at module scope) with
    // the B1 + B4 helpers defined inside alpine:init above.
    if (window.LG_HELPERS) {
        window.LG_HELPERS._failReason = _failReason;
        window.LG_HELPERS._normalizeShippedOps = _normalizeShippedOps;
    }

    // MoE architecture allow-list used by cmDetectMoe() to classify custom
    // HuggingFace models. Matched against HF API `tags` entries shaped like
    // `model_type:<t>` (once backend spec §2.5 lands) or fallback regex on
    // the model id. Keep aligned with vLLM's MoE-capable architectures.
    const MOE_MODEL_TYPES = [
        'mixtral', 'qwen2_moe', 'qwen3_moe', 'qwen3_5_moe',
        'deepseek_v2', 'deepseek_v3', 'deepseek_v4',
        'dbrx', 'jamba', 'arctic',
        'olmoe', 'granitemoe', 'llama4', 'phimoe',
    ];
    const MOE_NAME_REGEX = /(moe|mixtral|mixture|expert|\d+B-A\d+B|deepseek[\-_]?v[2-9])/i;

    Alpine.data('campaignApp', () => ({
        // ── State ──────────────────────────────────────────────────────
        currentLevel: 0,           // 0=hidden, 1=L1, 2=L2, 3=L3
        currentSessionId: null,
        currentRound: null,
        currentNode: null,

        campaignOverviews: [],     // L1: array of CampaignOverview (from /api/campaigns)
        allSessions: [],           // L1: all sessions from /sessions
        allCards: [],              // L1: merged + sorted cards (computed on data update)
        campaignState: null,       // L2/L3: full state.json
        // L2/L3 file tree from /api/campaigns/{id}/tree → {root, files: [path, ...]}.
        // Sidecar removal (2026-05-27): replaces the old `.metrics.json` aggregation;
        // metrics now come directly from state.json. Variable name kept for callsite
        // stability — treat its value as a tree response, not a sidecar dict.
        artifactCatalog: null,
        artifactContent: null,     // L3: artifact text content
        artifactMime: null,

        loading: false,
        loadError: null,

        // ── Server/GPU info (header + footer) ───────────────────────
        serverInfo: null,          // from /health

        _pollInterval: null,
        _hashListener: null,
        _keyListener: null,

        // ── Create Session modal state ────────────────────────────────
        createModalOpen: false,
        createModalEntering: false,  // true during open animation
        createModalLeaving: false,   // true during close animation
        cmCreating: false,           // POST in flight
        cmError: null,               // error message string
        cmModelQuery: '',
        cmShowDropdown: false,
        cmHfModels: [],
        cmHfLoading: false,
        cmHfTimeout: null,
        cmGpuInfo: { type: 'unknown', allowed_dtypes: ['bf16', 'fp16'] },
        cmGpuInfoLoaded: false,      // flipped true after first /health response
        cmHfConfigLoading: false,    // true while /api/hf-model-config in flight
        cmGatedHint: false,          // true → UI shows "gated — set TP/DP manually" hint
        cmForm: {
            cliTool: 'claude',
            tp: 1,
            dp: 1,                  // data parallel size (MoE models only)
            ep: false,              // enable expert parallelism (MoE models only)
            gpuCount: 1,            // total GPUs reserved (>= tp*dp, <= server capacity)
            dtype: 'bf16',
            additionalFlags: '',
            branch: 'main',
            sourceMode: 'default',  // 'default' | 'custom'
            forkUrl: '',            // custom fork git URL (github.com HTTPS)
            forkToken: '',          // optional access token for private forks
            // ── Workload config ──
            batchSizes: [8],                       // default [8], power-of-2 only
            maxModelLen: 'auto',                   // 'auto' or numeric string
            islOslPairs: [{ isl: 64, osl: 512 }], // at least 1 pair
            maxNumSeqs: 8,                         // auto-linked to max(batchSizes)
            maxNumSeqsLinked: true,                // true = auto-follows max(BS)
            showCustomBsInput: false,              // + custom input visibility
            customBsInput: '',                     // + custom input value
            customBsError: '',                     // po2 validation error
        },
        cmDockerCommit: null,      // Full 40-char hash from /workspace/vllm/.docker_commit
        cmVllmVersion: null,       // Release tag string (e.g. "v0.20.0") from /workspace/vllm/.docker_version
        cmForkUrlError: '',        // client-side fork URL validation message
        cmTpOptions: [1, 2, 4, 8],
        cmDpOptions: [1, 2, 4, 8],
        cmIsMoe: false,            // true → DP/EP pills enabled; set by preset/HF handlers

        // ── Guided tour state ─────────────────────────────────────────
        lgTourCompleted: localStorage.getItem('ammo_lg_tour_completed') === 'true',
        lgL1DeepCompleted: localStorage.getItem('ammo_lg_l1_deep_completed') === 'true',
        lgL2TourCompleted: localStorage.getItem('ammo_lg_l2_tour_completed') === 'true',
        lgL3TourCompleted: localStorage.getItem('ammo_lg_l3_tour_completed') === 'true',

        // ── Theme switcher state ──────────────────────────────────────
        showWelcome: false,
        showThemeConfirm: false,
        themeTarget: null,           // 'classic' or 'lightgrid' — what we're switching TO

        // ── Session action state ──────────────────────────────────────
        loadingActions: {},         // { sessionId: 'pausing'|'resuming'|'terminating' }
        downloadingSession: null,   // sessionId currently downloading
        confirmTerminate: null,     // sessionId awaiting terminate confirmation

        // ── Terminal overlay state ─────────────────────────────────────
        termMode: 'closed',         // 'closed' | 'half' | 'full'
        termSessionId: null,        // session whose ttyd is shown
        termAnimating: false,       // true during open animation
        termCopyMode: false,        // copy mode banner visible

        // ── Auth / login state ────────────────────────────────────────
        showLoginModal: false,
        loginKeyInput: '',
        loginError: '',
        loginChecking: false,

        get apiKey() {
            return localStorage.getItem('ammo_api_key') || '';
        },
        set apiKey(val) {
            if (val) {
                localStorage.setItem('ammo_api_key', val);
                document.cookie = `ammo_api_key=${val};max-age=${365*24*60*60};path=/;SameSite=Strict`;
            } else {
                localStorage.removeItem('ammo_api_key');
                document.cookie = 'ammo_api_key=;max-age=0;path=/';
            }
        },
        get clientId() {
            // Read the same key sessionApp uses — do NOT generate a new ID here.
            return localStorage.getItem('ammo_client_id') || '';
        },

        // ── Sidecar (markdown artifact) overlay state ──────────────────
        // Lightweight modal used to preview sidecar .md artifacts
        // (debate rationales, summaries) that don't have an op-scoped
        // L3 destination.
        sidecarOverlay: {
            open: false,
            leaving: false,
            loading: false,
            path: '',
            title: '',
            renderedHtml: '',
            errorMsg: '',
        },

        // ── Lifecycle ──────────────────────────────────────────────────
        async init() {
            this._hashListener = () => this._onHashChange();
            window.addEventListener('hashchange', this._hashListener);

            // Keep cmForm.gpuCount >= TP×DP whenever tp/dp change reactively
            // (e.g. preset selection, HF model load). Explicit setters
            // (cmSetTp/cmSetDp) clamp synchronously; $watch catches any other
            // mutation path.
            this.$watch('cmForm.tp', () => this.cmClampGpuCount());
            this.$watch('cmForm.dp', () => this.cmClampGpuCount());

            // Global keyboard shortcuts for terminal overlay + create modal + report
            this._keyListener = (e) => {
                // Esc → close theme confirm dialog
                if (e.key === 'Escape' && this.showThemeConfirm) {
                    e.preventDefault();
                    this.cancelThemeSwitch();
                    return;
                }
                // Esc → close welcome popup (default to lightgrid)
                if (e.key === 'Escape' && this.showWelcome) {
                    e.preventDefault();
                    this.dismissWelcome('close');
                    return;
                }
                // Esc → close changelog overlay
                if (e.key === 'Escape' && this.showChangelog) {
                    e.preventDefault();
                    this.showChangelog = false;
                    return;
                }
                // Esc → close sidecar overlay
                if (e.key === 'Escape' && this.sidecarOverlay.open) {
                    e.preventDefault();
                    this.closeSidecar();
                    return;
                }
                // Esc → close login modal
                if (e.key === 'Escape' && this.showLoginModal) {
                    return; // don't close login modal on Esc — user must authenticate
                }
                // Esc → close create modal first, then terminal overlay
                if (e.key === 'Escape' && this.createModalOpen) {
                    e.preventDefault();
                    e.stopPropagation();
                    this.closeCreateModal();
                    return;
                }
                // Esc → minimize (half → closed, full → half)
                if (e.key === 'Escape' && this.termMode !== 'closed') {
                    e.preventDefault();
                    this.termMode === 'full' ? this.termSetMode('half') : this.termClose();
                }
                // Ctrl+` → toggle half/closed
                if (e.key === '`' && e.ctrlKey) {
                    e.preventDefault();
                    this.termToggle(this.termSessionId || this.currentSessionId);
                }
            };
            window.addEventListener('keydown', this._keyListener);

            // JSON-tree delegated click — toggles `.jt-children.hidden` and the
            // rotated-triangle state on `.jt-toggle`. A single document-level
            // listener is used (not per-node) because a large state.json tree
            // has thousands of nodes and Alpine re-renders re-introduce leaks
            // of per-node listeners. `.jt-expandable` is a unique L3-viewer
            // selector, so this never fires on other subtrees.
            //
            // NOTE: the listener is bound to a *local* const so the exact
            // reference reaches `document.addEventListener`. This also makes
            // the handler trivial to introspect without going through the
            // Alpine reactive Proxy. `this._jsonTreeClickListener` is still
            // exposed for teardown hooks that search by name.
            const jsonTreeListener = (ev) => {
                const line = ev.target && ev.target.closest
                    ? ev.target.closest('.jt-expandable')
                    : null;
                if (!line) return;
                // Parent wrapper holds the .jt-children + .jt-close siblings.
                const node = line.parentElement;
                if (!node) return;
                const children = node.querySelector(':scope > .jt-children');
                const closeLine = node.querySelector(':scope > .jt-close');
                const toggle = line.querySelector(':scope > .jt-toggle');
                if (!children) return;
                const hidden = children.classList.toggle('hidden');
                if (closeLine) closeLine.style.display = hidden ? 'none' : '';
                if (toggle) toggle.classList.toggle('collapsed', hidden);
                // Stop propagation so nested expandable lines don't ALSO
                // toggle the outer one on the same click.
                ev.stopPropagation();
            };
            this._jsonTreeClickListener = jsonTreeListener;
            document.addEventListener('click', jsonTreeListener);

            // Auth check — show login modal if server requires auth
            const authRequired = await this._checkAuthRequired();
            if (authRequired) return; // wait for login; submitLogin() calls _postAuthInit()

            this._postAuthInit();
        },

        /** Post-auth initialization: hash routing + data loading. */
        _postAuthInit() {
            const theme = localStorage.getItem('ammo_ui_theme');

            // First visit — no preference saved — show welcome popup
            if (!theme) {
                this.showWelcome = true;
                return;
            }

            // Classic mode — don't activate LIGHTGRID
            if (theme === 'classic') {
                // Listen for switch-to-lightgrid requests from sessionApp header
                this._listenForThemeSwitch();
                return;
            }

            // LIGHTGRID mode — activate normally
            this._listenForThemeSwitch();
            // Fire-and-forget fetch of /health so cmGpuInfo, cmDockerCommit,
            // and cmVllmVersion are populated before session cards render.
            // Without this, cards initially show vllm@<7-char-hash> and only
            // switch to vllm@<version> after the user opens the create modal.
            this.loadGpuAndVllmInfo();
            if (!window.location.hash || window.location.hash === '#') {
                window.location.hash = 'campaigns';
                // hashchange event will fire and call _onHashChange
                // Tour auto-start is now controlled by the theme switcher
                return;
            }
            this._onHashChange();
        },

        /** Listen for cross-component theme switch events (from sessionApp header button). */
        _listenForThemeSwitch() {
            if (this._themeSwitchListener) return;
            this._themeSwitchListener = (e) => {
                this.themeTarget = e.detail;
                this.showThemeConfirm = true;
            };
            document.addEventListener('ammo-theme-switch', this._themeSwitchListener);
        },

        destroy() {
            if (this._hashListener) {
                window.removeEventListener('hashchange', this._hashListener);
            }
            if (this._keyListener) {
                window.removeEventListener('keydown', this._keyListener);
            }
            if (this._themeSwitchListener) {
                document.removeEventListener('ammo-theme-switch', this._themeSwitchListener);
            }
            this.stopPolling();
        },

        // ── Create Session Modal ──────────────────────────────────────

        /**
         * Fetch /health once and populate:
         *   - cmGpuInfo, cmForm.dtype (for the create modal)
         *   - cmDockerCommit, cmVllmVersion (for source labels on every card)
         *
         * Guarded by `cmGpuInfoLoaded` so it's idempotent.
         * Called from init() so session cards render the release-version label
         * on first paint (instead of showing vllm@<7-char-hash> until the user
         * opens the create modal), and also from openCreateModal() as a
         * fall-back in case init() was not awaited before the user clicks.
         */
        async loadGpuAndVllmInfo() {
            if (this.cmGpuInfoLoaded) return;
            try {
                const resp = await this.apiFetch('/health');
                if (!resp.ok) return;
                const data = await resp.json();
                if (data.gpu) {
                    this.cmGpuInfo = data.gpu;
                    if (data.gpu.allowed_dtypes?.length > 0) {
                        this.cmForm.dtype = data.gpu.allowed_dtypes[0];
                    }
                }
                if (data.vllm?.docker_commit) {
                    this.cmDockerCommit = data.vllm.docker_commit;
                }
                // Release-wheel images expose vllm.version (e.g. "v0.20.0").
                // Legacy nightly images return null → cmVllmVersion stays null
                // and the UI falls back to the truncated commit hash.
                if (data.vllm?.version) {
                    this.cmVllmVersion = data.vllm.version;
                }
                this.cmGpuInfoLoaded = true;
            } catch (e) {
                console.error('[campaignApp] loadGpuAndVllmInfo failed:', e);
            }
        },

        /** Open the create modal; load GPU/vLLM info if not cached. */
        async openCreateModal() {
            this.createModalOpen = true;
            this.createModalEntering = true;
            this.cmError = null;
            this.cmCreating = false;
            this.cmGatedHint = false;
            setTimeout(() => { this.createModalEntering = false; }, 1200);
            // Fallback fetch — typically a no-op because init() already called
            // loadGpuAndVllmInfo() at app bootstrap (the cmGpuInfoLoaded guard
            // makes it idempotent), but safe to re-call if init() was skipped
            // or the fetch failed.
            await this.loadGpuAndVllmInfo();
        },

        /** Close the create modal with exit animation. */
        closeCreateModal() {
            this.createModalLeaving = true;
            this.cmShowDropdown = false;
            setTimeout(() => {
                this.createModalOpen = false;
                this.createModalLeaving = false;
                this.cmError = null;
            }, 250);
        },

        /** Debounced HuggingFace model search. */
        cmDebouncedSearch() {
            clearTimeout(this.cmHfTimeout);
            if (!this.cmModelQuery.trim()) {
                this.cmHfModels = [];
                return;
            }
            this.cmHfTimeout = setTimeout(() => this.cmSearchHf(), 300);
        },

        /** Execute HuggingFace model search. */
        async cmSearchHf() {
            const q = this.cmModelQuery.trim();
            if (!q) return;
            this.cmHfLoading = true;
            try {
                const resp = await this.apiFetch(`/api/hf-models?q=${encodeURIComponent(q)}&limit=20`);
                const data = await resp.json();
                this.cmHfModels = data.models || [];
            } catch (e) {
                this.cmHfModels = [];
            } finally {
                this.cmHfLoading = false;
            }
        },

        /**
         * Detect whether a model is MoE — name-based fallback only.
         *
         * Authoritative MoE signal now comes from the /api/hf-model-config
         * endpoint (`is_moe` in the response).  This function is retained as
         * a best-effort fallback for the brief window between HF select and
         * config resolve, and for gated models where the config fetch yields
         * no signal.
         */
        cmDetectMoe(model) {
            if (!model) return false;
            // HF path — check tags for model_type prefix OR bare architecture name.
            if (Array.isArray(model.tags)) {
                for (const t of model.tags) {
                    if (typeof t !== 'string') continue;
                    const m = t.match(/^model_type:(.+)$/i);
                    if (m && MOE_MODEL_TYPES.includes(m[1].toLowerCase())) return true;
                    if (MOE_MODEL_TYPES.includes(t.toLowerCase())) return true;
                }
            }
            // …fall back to name-based regex on the model id.
            const id = typeof model.id === 'string' ? model.id : '';
            return MOE_NAME_REGEX.test(id);
        },

        /** Force DP/EP controls back to their disabled-state defaults. */
        cmResetParallelism() {
            this.cmForm.dp = 1;
            this.cmForm.ep = false;
        },

        /**
         * Select a HuggingFace model and auto-fill TP/DP/dtype via the new
         * /api/hf-model-config endpoint.
         *
         * Graceful degradation:
         *   - `reason === "gated"` → flip `cmGatedHint`; leave TP/DP alone.
         *   - `reason === "network_error"` / fetch error → keep manual values.
         *   - `reason === "config_missing_fields"` → same as network error.
         *
         * dtype is only applied if the suggested value is in the host GPU's
         * `allowed_dtypes` list (prevents e.g. fp8 on A100).
         */
        async cmSelectHfModel(model) {
            this.cmModelQuery = model.id;
            this.cmShowDropdown = false;
            this.cmHfConfigLoading = true;
            this.cmGatedHint = false;

            try {
                const resp = await this.apiFetch(
                    `/api/hf-model-config/${encodeURIComponent(model.id)}`
                );
                if (!resp.ok) {
                    // Fall back to name-based MoE detection.
                    this.cmIsMoe = this.cmDetectMoe(model);
                    if (!this.cmIsMoe) this.cmResetParallelism();
                    return;
                }
                const cfg = await resp.json();
                if (cfg.reason === 'gated') {
                    this.cmGatedHint = true;
                    // Preserve user's current TP/DP values; fall back to
                    // name-based MoE detection for the DP/EP enablement.
                    this.cmIsMoe = this.cmDetectMoe(model);
                    if (!this.cmIsMoe) this.cmResetParallelism();
                    return;
                }
                if (cfg.suggested_tp != null) this.cmForm.tp = cfg.suggested_tp;
                if (cfg.suggested_dp != null) this.cmForm.dp = cfg.suggested_dp;
                if (
                    cfg.suggested_dtype &&
                    this.cmGpuInfo.allowed_dtypes?.includes(cfg.suggested_dtype)
                ) {
                    this.cmForm.dtype = cfg.suggested_dtype;
                }
                this.cmIsMoe = cfg.is_moe === true;
                if (!this.cmIsMoe) {
                    // Force reset on dense to prevent stale MoE parallelism
                    // state from bleeding into a model that cannot use it.
                    this.cmResetParallelism();
                }
            } catch (e) {
                console.error('[campaignApp] cmSelectHfModel config fetch:', e);
                this.cmIsMoe = this.cmDetectMoe(model);
                if (!this.cmIsMoe) this.cmResetParallelism();
            } finally {
                this.cmHfConfigLoading = false;
                // Reset pool size to the new TP×DP floor.
                this.cmForm.gpuCount = this.cmGpuMin();
            }
        },

        /** Format download count for HF results. */
        cmFormatDownloads(n) {
            if (!n) return '';
            if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M dl`;
            if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K dl`;
            return `${n} dl`;
        },

        /** Toggle a batch size chip on/off. At least 1 must remain. */
        cmToggleBatchSize(bs) {
            const idx = this.cmForm.batchSizes.indexOf(bs);
            if (idx >= 0) {
                if (this.cmForm.batchSizes.length <= 1) return; // keep at least 1
                this.cmForm.batchSizes.splice(idx, 1);
            } else {
                this.cmForm.batchSizes.push(bs);
                this.cmForm.batchSizes.sort((a, b) => a - b);
            }
            this.cmSyncMaxNumSeqs();
        },

        /** Check if a value is a power of 2. */
        cmIsPo2(n) {
            return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
        },

        /** Submit + custom batch size input. */
        cmAddCustomBatchSize() {
            const val = parseInt(this.cmForm.customBsInput, 10);
            if (!this.cmIsPo2(val)) {
                this.cmForm.customBsError = 'Must be a power of 2 (1, 2, 4, 8, 16, ...)';
                return;
            }
            this.cmForm.customBsError = '';
            if (!this.cmForm.batchSizes.includes(val)) {
                this.cmForm.batchSizes.push(val);
                this.cmForm.batchSizes.sort((a, b) => a - b);
            }
            this.cmForm.customBsInput = '';
            this.cmForm.showCustomBsInput = false;
            this.cmSyncMaxNumSeqs();
        },

        /** Cancel + custom input. */
        cmCancelCustomBs() {
            this.cmForm.showCustomBsInput = false;
            this.cmForm.customBsInput = '';
            this.cmForm.customBsError = '';
        },

        /** Auto-sync maxNumSeqs to max(batchSizes) when linked. */
        cmSyncMaxNumSeqs() {
            if (this.cmForm.maxNumSeqsLinked && this.cmForm.batchSizes.length > 0) {
                this.cmForm.maxNumSeqs = Math.max(...this.cmForm.batchSizes);
            }
        },

        /** Handle manual edit of maxNumSeqs — breaks or restores auto-link. */
        cmOnMaxNumSeqsInput(val) {
            const trimmed = String(val).trim();
            if (trimmed === '') {
                this.cmForm.maxNumSeqsLinked = true;
                this.cmSyncMaxNumSeqs();
            } else {
                this.cmForm.maxNumSeqsLinked = false;
                this.cmForm.maxNumSeqs = parseInt(trimmed, 10) || 0;
            }
        },

        /** Add a new ISL/OSL pair with default values. */
        cmAddIslOslPair() {
            this.cmForm.islOslPairs.push({ isl: 64, osl: 512 });
        },

        /** Remove an ISL/OSL pair by index (blocked if only 1 remains). */
        cmRemoveIslOslPair(index) {
            if (this.cmForm.islOslPairs.length <= 1) return;
            this.cmForm.islOslPairs.splice(index, 1);
        },

        /** Should the time warning be shown? */
        cmShowTimeWarning() {
            return this.cmForm.batchSizes.length > 1 || this.cmForm.islOslPairs.length > 1;
        },

        /** Minimum GPUs required for one model replica = TP × DP. */
        cmGpuMin() {
            return (this.cmForm.tp || 1) * (this.cmForm.dp || 1);
        },

        /** Maximum GPUs allowed = server capacity (see cmServerGpuCount). */
        cmGpuMax() {
            return this.cmServerGpuCount();
        },

        /**
         * Legacy template-alias retained for any remaining `cmGpuTotal()`
         * references (parallelism badge, prompt text, etc.). After the
         * gpu_count decouple, this is the TP×DP floor — the stepper value
         * lives on `cmForm.gpuCount`.
         */
        cmGpuTotal() {
            return this.cmGpuMin();
        },

        /** Increment gpuCount, capped at server capacity. */
        cmIncGpuCount() {
            if (this.cmForm.gpuCount < this.cmGpuMax()) this.cmForm.gpuCount++;
        },

        /** Decrement gpuCount, floored at TP×DP min. */
        cmDecGpuCount() {
            if (this.cmForm.gpuCount > this.cmGpuMin()) this.cmForm.gpuCount--;
        },

        /** Clamp gpuCount up to the TP×DP min after tp/dp changes. */
        cmClampGpuCount() {
            const min = this.cmGpuMin();
            if (this.cmForm.gpuCount < min) this.cmForm.gpuCount = min;
        },

        /** Explicit TP setter that re-clamps gpuCount synchronously. */
        cmSetTp(n) {
            this.cmForm.tp = n;
            this.cmClampGpuCount();
        },

        /** Explicit DP setter that re-clamps gpuCount synchronously. */
        cmSetDp(n) {
            this.cmForm.dp = n;
            this.cmClampGpuCount();
        },

        /**
         * Server GPU capacity sourced from `/health` (cmGpuInfo.total_gpus).
         * Defaults to Number.MAX_SAFE_INTEGER when the server does not expose
         * a capacity (spec §1.7 + DA HIGH#1). This keeps the over-capacity
         * banner silent unless capacity is known; server-side 503 handling in
         * `app.py` remains the authoritative fallback.
         */
        cmServerGpuCount() {
            const n = this.cmGpuInfo?.total_gpus;
            return (typeof n === 'number' && n > 0) ? n : Number.MAX_SAFE_INTEGER;
        },

        /** True when gpuCount exceeds known server capacity (badge turns red). */
        cmGpuOverCapacity() {
            return this.cmForm.gpuCount > this.cmServerGpuCount();
        },

        /** Build the initial prompt string with all workload config flags. */
        cmGeneratePrompt() {
            const model = this.cmModelQuery.trim();
            if (!model) return 'Select a model to see the generated prompt';

            const bs = [...this.cmForm.batchSizes].sort((a, b) => a - b).join(' ');
            const mml = this.cmForm.maxModelLen || 'auto';
            const mns = this.cmForm.maxNumSeqs;
            const pairs = this.cmForm.islOslPairs;

            let prompt = `Use $ammo for model_id=${model} TP=${this.cmForm.tp} dtype=${this.cmForm.dtype}`;
            prompt += ` --batch-sizes ${bs}`;
            prompt += ` --max-model-len=${mml}`;
            prompt += ` --max-num-seqs=${mns}`;

            if (pairs.length === 1) {
                prompt += ` --input-len=${pairs[0].isl} --output-len=${pairs[0].osl}`;
            } else {
                const pairStr = pairs.map(p => `${p.isl}:${p.osl}`).join(',');
                prompt += ` --isl-osl=${pairStr}`;
            }

            // Parallelism flags (MoE only — cmForm.dp/ep are pinned to
            // defaults for dense models by cmResetParallelism).
            if (this.cmForm.dp > 1) {
                prompt += ` --data-parallel-size ${this.cmForm.dp}`;
            }
            if (this.cmForm.ep) {
                prompt += ` --enable-expert-parallel`;
            }

            if (this.cmForm.additionalFlags) {
                prompt += ` ${this.cmForm.additionalFlags}`;
            }
            return prompt;
        },

        /** Can we create? */
        cmCanCreate() {
            return this.cmModelQuery.trim()
                && this.cmForm.tp > 0
                && this.cmForm.dp > 0
                && !this.cmGpuOverCapacity()
                && this.cmForm.gpuCount >= this.cmGpuMin()
                && !this.cmCreating;
        },

        /** Validate and submit session creation to this server. */
        /** Client-side github.com HTTPS fork URL check. Returns '' if ok/empty. */
        cmValidateForkUrl(url) {
            if (!url) return '';
            const re = /^https:\/\/github\.com\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+(\.git)?\/?$/;
            return re.test(url.trim()) ? '' : 'Must be https://github.com/<owner>/<repo>';
        },

        /** Elapsed build time label, e.g. " 4m". card.created_at is unix seconds. */
        cmBuildElapsed(card) {
            if (!card || !card.created_at) return '';
            const mins = Math.floor((Date.now() / 1000 - card.created_at) / 60);
            return mins > 0 ? ` ${mins}m` : '';
        },

        /** Show the build error (last lines) for a FAILED fork session. */
        cmShowBuildError(card) {
            this.cmError = card.build_error || 'Build failed (no log captured).';
            // Reuse the LIGHTGRID toast surface; fall back to alert.
            if (typeof this.showToast === 'function') {
                this.showToast(this.cmError, 8000);
            } else {
                window.alert(this.cmError);
            }
        },

        async cmCreateSession() {
            if (!this.cmCanCreate()) return;
            this.cmCreating = true;
            this.cmError = null;

            // Resolve branch from source mode
            let branch = 'main';
            if (this.cmForm.sourceMode === 'default' && this.cmDockerCommit) {
                branch = this.cmDockerCommit;
            } else if (this.cmForm.sourceMode === 'custom') {
                branch = this.cmForm.branch || 'main';
            }

            const tp = this.cmForm.tp;
            const dp = this.cmForm.dp || 1;
            const payload = {
                repo_name: 'vllm',
                cli_tool: this.cmForm.cliTool,
                branch: branch,
                gpu_count: this.cmForm.gpuCount,
                tp_size: tp,
                dp_size: dp,
                initial_prompt: this.cmGeneratePrompt(),
                inactivity_timeout_mins: 720,
                model_name: this.cmModelQuery.trim(),
                dtype: this.cmForm.dtype,
            };

            // Custom fork (only when a URL is entered under Custom source).
            if (this.cmForm.sourceMode === 'custom' && this.cmForm.forkUrl) {
                const err = this.cmValidateForkUrl(this.cmForm.forkUrl);
                if (err) { this.cmError = err; this.cmCreating = false; return; }
                payload.vllm_fork_url = this.cmForm.forkUrl.trim();
                if (this.cmForm.forkToken) {
                    payload.vllm_fork_token = this.cmForm.forkToken;
                }
            }

            try {
                const resp = await this.apiFetch('/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });

                if (resp.ok) {
                    this.closeCreateModal();
                    // Refresh in background — polling will also pick it up
                    this.loadCampaigns();
                    return;
                }

                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.detail || errData.message || `HTTP ${resp.status}`);
            } catch (e) {
                this.cmError = e.message || 'Failed to create session';
            } finally {
                this.cmCreating = false;
            }
        },

        // ── Theme Switcher ─────────────────────────────────────────────

        /** Handle welcome popup dismissal.
         *  @param {'explore'|'classic'|'close'} choice */
        dismissWelcome(choice) {
            this.showWelcome = false;
            this._listenForThemeSwitch();
            if (choice === 'explore') {
                localStorage.setItem('ammo_ui_theme', 'lightgrid');
                window.location.hash = 'campaigns';
                this._onHashChange();
                setTimeout(() => this.startQuickTour(), 300);
            } else if (choice === 'classic') {
                localStorage.setItem('ammo_ui_theme', 'classic');
                this.playGlitchTransition(() => {
                    // sessionApp is already rendered underneath — just stay hidden
                });
            } else {
                // Close (X) — default to lightgrid, no tour
                localStorage.setItem('ammo_ui_theme', 'lightgrid');
                window.location.hash = 'campaigns';
                this._onHashChange();
            }
        },

        /** Show the theme switch confirmation dialog (from LIGHTGRID header). */
        showConfirmDialog(target) {
            this.themeTarget = target;
            this.showThemeConfirm = true;
        },

        /** Cancel the theme switch. */
        cancelThemeSwitch() {
            this.showThemeConfirm = false;
            this.themeTarget = null;
        },

        /** Confirm and execute the theme switch. */
        confirmThemeSwitch() {
            const target = this.themeTarget;
            this.showThemeConfirm = false;
            this.themeTarget = null;
            localStorage.setItem('ammo_ui_theme', target);

            this.playGlitchTransition(() => {
                if (target === 'classic') {
                    this.currentLevel = 0;
                    document.body.classList.remove('lg-active');
                    window.location.hash = '';
                } else {
                    window.location.hash = 'campaigns';
                    this._onHashChange();
                }
            });
        },

        /** Play the CRT glitch transition overlay. Content swaps at 250ms, overlay clears at 500ms.
         *  @param {Function} [callback] — called at 250ms (the swap point) */
        playGlitchTransition(callback) {
            const overlay = document.getElementById('lg-theme-glitch-overlay');
            if (!overlay) { callback?.(); return; }
            overlay.classList.add('active');
            setTimeout(() => { callback?.(); }, 250);
            setTimeout(() => { overlay.classList.remove('active'); }, 500);
        },

        // ── Guided Tour System ──────────────────────────────────────────
        //
        // 4 independent tours, each with its own localStorage completion key:
        //   1. Quick Intro   — startQuickTour()       → ammo_lg_tour_completed
        //   2. L1 Deep Dive  — startL1DeepTour()      → ammo_lg_l1_deep_completed
        //   3. L2 Circuit    — startL2Tour()           → ammo_lg_l2_tour_completed
        //   4. L3 Artifacts  — startL3Tour()           → ammo_lg_l3_tour_completed
        //
        // Context-sensitive ?-button dispatches via startContextTour().

        /** Quick Intro Tour (6 original steps + optional "Go Deeper" prompt) */
        startQuickTour() {
            if (typeof window.driver === 'undefined') {
                console.warn('driver.js not loaded');
                return;
            }
            if (this.currentLevel !== 1) {
                this.navigateTo(1);
                setTimeout(() => this.startQuickTour(), 400);
                return;
            }

            const self = this;
            const showDeeper = !localStorage.getItem('ammo_lg_l1_deep_completed');
            let goDeeper = false;

            const steps = [
                {
                    element: '.lg-header',
                    popover: {
                        title: 'LIGHTGRID Dashboard',
                        description: 'Welcome to the AMMO campaign dashboard. This is your command center for managing GPU optimization sessions.',
                        side: 'bottom',
                        align: 'center',
                    }
                },
                {
                    element: '.lg-new-session-btn',
                    popover: {
                        title: 'Create a Session',
                        description: 'Launch a new optimization campaign. Pick a HuggingFace model, set TP parallelism and dtype, then let Claude optimize.',
                        side: 'bottom',
                    }
                },
                {
                    element: '.lg-status-chip.active',
                    popover: {
                        title: 'Status Chips',
                        description: 'Live counts of your sessions by status. Active sessions are running optimizations; paused sessions release GPUs but preserve state.',
                        side: 'bottom',
                    }
                },
                {
                    element: '.lg-session-card',
                    popover: {
                        title: 'Campaign Card',
                        description: 'Each card shows a session\'s model, status, speedup progress, and optimization pipeline. Click to drill into the circuit board view.',
                        side: 'right',
                    }
                },
                {
                    popover: {
                        title: 'Three-Level Navigation',
                        description: 'L1: Overview grid of all campaigns. L2: Circuit board view for a single session. L3: Artifact viewer for individual optimization rounds. Use the breadcrumb to navigate between levels.',
                    }
                },
                {
                    popover: {
                        title: 'Terminal Overlay',
                        description: 'Press Ctrl+` to open a live terminal connected to any session. Use it to interact directly with the Claude agent running your optimization. Click the ? button anytime to replay this tour.',
                    }
                },
            ];

            // Step 7: "Go deeper?" prompt (only if L1 deep tour not yet completed)
            if (showDeeper) {
                steps.push({
                    popover: {
                        title: 'Want the full tour?',
                        description: 'That was the quick overview. Ready to learn about every element on this page \u2014 card details, session actions, the create modal, and more?'
                            + '<div class="lg-tour-prompt-btns">'
                            + '<button class="lg-tour-btn-go" id="lg-tour-go-deeper">Yes, show me \u2192</button>'
                            + '<button class="lg-tour-btn-dismiss" id="lg-tour-dismiss">No thanks</button>'
                            + '</div>',
                        showButtons: [],
                    }
                });
            }

            // Filter out steps whose target element doesn't exist in the DOM
            const filteredSteps = steps.filter(s => !s.element || document.querySelector(s.element));

            const driverObj = window.driver.js.driver({
                showProgress: true,
                animate: true,
                popoverClass: 'driver-popover',
                steps: filteredSteps,
                onHighlighted: () => {
                    // Wire custom buttons on the "Go Deeper" step when they appear
                    setTimeout(() => {
                        const goBtn = document.getElementById('lg-tour-go-deeper');
                        const dismissBtn = document.getElementById('lg-tour-dismiss');
                        if (goBtn && !goBtn._wired) {
                            goBtn._wired = true;
                            goBtn.addEventListener('click', () => {
                                goDeeper = true;
                                driverObj.destroy();
                            });
                        }
                        if (dismissBtn && !dismissBtn._wired) {
                            dismissBtn._wired = true;
                            dismissBtn.addEventListener('click', () => {
                                driverObj.destroy();
                            });
                        }
                    }, 50);
                },
                onDestroyStarted: () => {
                    localStorage.setItem('ammo_lg_tour_completed', 'true');
                    self.lgTourCompleted = true;
                    driverObj.destroy();
                },
                onDestroyed: () => {
                    if (goDeeper) {
                        setTimeout(() => self.startL1DeepTour(), 300);
                    }
                },
            });
            driverObj.drive();
        },

        /** Backward-compat alias for existing callers */
        startLgTour() { this.startQuickTour(); },

        /** L1 Deep Dive — detailed walkthrough of every L1 element */
        startL1DeepTour() {
            if (typeof window.driver === 'undefined') return;
            if (this.currentLevel !== 1) {
                this.navigateTo(1);
                setTimeout(() => this.startL1DeepTour(), 400);
                return;
            }

            const self = this;
            const allSteps = [
                {
                    element: '.lg-session-card',
                    popover: {
                        title: 'Campaign Card Anatomy',
                        description: 'Each card represents one optimization session. Let\u2019s break down what you see.',
                        side: 'right',
                    }
                },
                {
                    element: '.lg-card-model-name',
                    popover: {
                        title: 'Model Identity',
                        description: 'The HuggingFace model being optimized. Shows the model name and configuration.',
                        side: 'bottom',
                    }
                },
                {
                    element: '.lg-config-tag.dtype',
                    popover: {
                        title: 'Data Type',
                        description: 'The numerical precision used for inference: fp8, bf16, fp16, etc. Lower precision = faster but potentially less accurate.',
                        side: 'bottom',
                    }
                },
                {
                    element: '.lg-speedup-value',
                    popover: {
                        title: 'Speedup Metric',
                        description: 'Performance improvement of the best optimized kernel vs the original baseline. Measured as wall-clock latency reduction over 100 runs.',
                        side: 'left',
                    }
                },
                {
                    element: '.lg-card-time',
                    popover: {
                        title: 'Session Duration',
                        description: 'Total wall-clock time since this optimization campaign started. Includes all rounds of optimization.',
                        side: 'top',
                    }
                },
                {
                    element: '.lg-pipeline',
                    popover: {
                        title: 'Pipeline Progress',
                        description: 'Shows which optimization stages have been completed for the latest round. Each dot represents: Mining \u2192 Debate \u2192 Implement \u2192 Validate \u2192 Integrate.',
                        side: 'top',
                    }
                },
                {
                    element: '.lg-session-card',
                    popover: {
                        title: 'Active vs Paused',
                        description: 'Active sessions (cyan glow) are running optimizations and using GPUs. Paused sessions (dim, no glow) release GPUs but preserve all state \u2014 resume anytime.',
                        side: 'right',
                    }
                },
                {
                    element: '.lg-new-session-btn',
                    popover: {
                        title: 'Create Session',
                        description: 'Opens the session creator. Pick a HuggingFace model, set tensor parallelism, choose dtype, and Claude starts optimizing.',
                        side: 'bottom',
                    }
                },
                {
                    element: '.lg-server-info',
                    popover: {
                        title: 'GPU Availability',
                        description: 'Shows how many GPUs are available on this server. If no GPUs are free, you\u2019ll need to pause an existing session.',
                        side: 'bottom',
                    }
                },
                {
                    element: '.lg-section-header.paused-section',
                    popover: {
                        title: 'Paused Sessions',
                        description: 'Paused campaigns are collapsed here. Click to expand. Paused sessions don\u2019t use GPUs but keep their optimization progress.',
                        side: 'top',
                    }
                },
            ];

            // Filter to only steps whose target element exists in the DOM
            const steps = allSteps.filter(s => !s.element || document.querySelector(s.element));

            if (steps.length === 0) return;

            const driverObj = window.driver.js.driver({
                showProgress: true,
                animate: true,
                popoverClass: 'driver-popover',
                steps,
                onDestroyStarted: () => {
                    localStorage.setItem('ammo_lg_l1_deep_completed', 'true');
                    self.lgL1DeepCompleted = true;
                    driverObj.destroy();
                },
            });
            driverObj.drive();
        },

        /** Helper: find a circuit-board chip by its designation text (BSL, MNE, DBT, INT) */
        _findCb2Chip(desig) {
            const els = document.querySelectorAll('#cb-mount .cb2-desig');
            for (const el of els) {
                if (el.textContent.trim() === desig) return el.closest('.cb2-hud');
            }
            return null;
        },

        /** Build L2 tour content steps dynamically from rendered circuit board elements */
        _buildL2ContentSteps() {
            const steps = [];
            const ensureId = (el, fallback) => { if (!el.id) el.id = fallback; return '#' + el.id; };

            // 1. Full circuit board
            if (document.querySelector('#cb-mount'))
                steps.push({ element: '#cb-mount', popover: { title: 'Pipeline Overview', description: 'The circuit board reads left to right. Each column is a stage in the optimization pipeline. Signals flow from Baseline through to Integration.', side: 'bottom' } });

            // 2-4. Stage columns by designation
            const stageMap = [
                ['BSL', 'Baseline Stage', 'The starting point. Measures original kernel performance before any optimization.'],
                ['MNE', 'Mining Stage', 'Claude analyzes the kernel, identifying bottlenecks and optimization opportunities.'],
                ['DBT', 'Debate Stage', 'An adversarial review. Advocate argues FOR the proposed optimization, critic argues AGAINST. Best ideas survive.'],
            ];
            stageMap.forEach(([desig, title, desc], i) => {
                const chip = this._findCb2Chip(desig);
                if (chip) steps.push({ element: ensureId(chip, 'cb2-tour-' + desig.toLowerCase()), popover: { title, description: desc, side: 'right' } });
            });

            // 5. Implement (first track chip)
            const trackChip = document.querySelector('#cb-mount .cb2-track');
            if (trackChip) steps.push({ element: ensureId(trackChip, 'cb2-tour-track'), popover: { title: 'Implement Stage', description: 'The optimized kernel code is written and compiled. May include CUDA, Triton, or fused operations.', side: 'right' } });

            // 6. Validate (centered — validation is embedded in track flow)
            steps.push({ popover: { title: 'Validate Stage', description: 'Correctness testing (does it produce the same output?) and performance profiling (is it actually faster?).' } });

            // 7. Integration chip
            const integChip = document.querySelector('#cb-mount .cb2-integ');
            if (integChip) steps.push({ element: ensureId(integChip, 'cb2-tour-integ'), popover: { title: 'Integrate Stage', description: 'Validated optimizations are integrated into the session\u2019s best kernel. The speedup metric updates.', side: 'left' } });

            // 8. A round node (first HUD chip)
            const anyNode = document.querySelector('#cb-mount .cb2-hud');
            if (anyNode) steps.push({ element: ensureId(anyNode, 'cb2-tour-node'), popover: { title: 'Round Nodes', description: 'Each node is one optimization attempt. Green = success, pulsing = in progress, red = failed. Click any node to drill into L3 details.', side: 'bottom' } });

            // 9. SVG traces
            if (document.querySelector('svg.cb2-trace-layer'))
                steps.push({ element: 'svg.cb2-trace-layer', popover: { title: 'Circuit Traces', description: 'Lines connect rounds across stages, showing the flow of each optimization attempt through the pipeline.', side: 'bottom' } });

            // 10. Speedup in topbar
            if (document.querySelector('.l2-topbar'))
                steps.push({ element: '.l2-topbar', popover: { title: 'Speedup Trend', description: 'Tracks cumulative speedup over optimization rounds. An upward trend means the campaign is finding better optimizations.', side: 'bottom' } });

            // 11. Round summary (centered)
            steps.push({ popover: { title: 'Round Summary', description: 'Each card summarizes one round: what was tried, whether it passed validation, and the resulting speedup delta. Hover over any node for details.' } });

            return steps;
        },

        /** L2 Circuit Board Tour — intro prompt + pipeline walkthrough */
        startL2Tour() {
            if (typeof window.driver === 'undefined') return;
            const self = this;
            let continueToFull = false;

            const introStep = {
                popover: {
                    title: 'Welcome to the Circuit Board',
                    description: 'This view shows all optimization rounds for this campaign as a circuit board. Each column is a pipeline stage, and nodes represent individual rounds. Want a quick walkthrough?'
                        + '<div class="lg-tour-prompt-btns">'
                        + '<button class="lg-tour-btn-go" id="lg-tour-l2-go">Yes, let\u2019s go \u2192</button>'
                        + '<button class="lg-tour-btn-dismiss" id="lg-tour-l2-dismiss">No thanks</button>'
                        + '</div>',
                    showButtons: [],
                }
            };

            const driverObj = window.driver.js.driver({
                showProgress: false,
                animate: true,
                popoverClass: 'driver-popover',
                steps: [introStep],
                onHighlighted: () => {
                    setTimeout(() => {
                        const goBtn = document.getElementById('lg-tour-l2-go');
                        const dismissBtn = document.getElementById('lg-tour-l2-dismiss');
                        if (goBtn && !goBtn._wired) {
                            goBtn._wired = true;
                            goBtn.addEventListener('click', () => { continueToFull = true; driverObj.destroy(); });
                        }
                        if (dismissBtn && !dismissBtn._wired) {
                            dismissBtn._wired = true;
                            dismissBtn.addEventListener('click', () => {
                                localStorage.setItem('ammo_lg_l2_tour_completed', 'true');
                                self.lgL2TourCompleted = true;
                                driverObj.destroy();
                            });
                        }
                    }, 50);
                },
                onDestroyStarted: () => {
                    localStorage.setItem('ammo_lg_l2_tour_completed', 'true');
                    self.lgL2TourCompleted = true;
                    driverObj.destroy();
                },
                onDestroyed: () => {
                    if (continueToFull) {
                        // Build steps lazily — circuit board elements should exist by now
                        setTimeout(() => {
                            const contentSteps = self._buildL2ContentSteps();
                            if (contentSteps.length === 0) return;
                            const fullDriver = window.driver.js.driver({
                                showProgress: true,
                                animate: true,
                                popoverClass: 'driver-popover',
                                steps: contentSteps,
                                onDestroyStarted: () => fullDriver.destroy(),
                            });
                            fullDriver.drive();
                        }, 200);
                    }
                },
            });
            driverObj.drive();
        },

        /** Build L3 tour content steps dynamically from rendered artifact viewer elements */
        _buildL3ContentSteps() {
            const steps = [];

            if (document.querySelector('.l3-overview'))
                steps.push({ element: '.l3-overview', popover: { title: 'Round Overview', description: 'The round number, optimization strategy name, and model identity. This header tells you which round you\u2019re examining.', side: 'bottom' } });

            if (document.querySelector('.l3-overview-metrics .l3-metric-cell'))
                steps.push({ element: '.l3-overview-metrics .l3-metric-cell', popover: { title: 'Key Metrics', description: 'Performance metrics for this round. Depending on the stage: baseline latency, top component, decode share, Amdahl ceiling, or track speedup.', side: 'bottom' } });

            if (document.querySelector('.l3-overview .l3-pipeline'))
                steps.push({ element: '.l3-overview .l3-pipeline', popover: { title: 'Pipeline Progress', description: 'Visual indicator of which stages this round completed. Green stages passed, red stages failed. Click a stage to view its artifacts.', side: 'bottom' } });

            if (document.querySelector('.l3-track-table'))
                steps.push({ element: '.l3-track-table', popover: { title: 'Optimization Tracks', description: 'Detailed table of each optimization track attempted in this round. Shows the approach, result, and performance delta.', side: 'top' } });

            if (document.querySelector('.l3-tab-bar'))
                steps.push({ element: '.l3-tab-bar', popover: { title: 'Artifact Tabs', description: 'Switch between different artifacts: source code, compilation logs, profiling data, correctness results. Each tab shows the raw output from that stage.', side: 'top' } });

            if (document.querySelector('.l3-nav-btn'))
                steps.push({ element: '.l3-nav-btn', popover: { title: 'Navigation', description: 'Use the back button to return to the Circuit Board (L2). From there, go back to the Campaign Grid (L1).', side: 'bottom' } });

            return steps;
        },

        /** L3 Artifacts Tour — intro prompt + artifact viewer walkthrough */
        startL3Tour() {
            if (typeof window.driver === 'undefined') return;
            const self = this;
            let continueToFull = false;

            const introStep = {
                popover: {
                    title: 'Welcome to the Artifact Viewer',
                    description: 'Deep dive into a single optimization round. Every metric, artifact, and result from this attempt. Want a walkthrough?'
                        + '<div class="lg-tour-prompt-btns">'
                        + '<button class="lg-tour-btn-go" id="lg-tour-l3-go">Yes, let\u2019s go \u2192</button>'
                        + '<button class="lg-tour-btn-dismiss" id="lg-tour-l3-dismiss">No thanks</button>'
                        + '</div>',
                    showButtons: [],
                }
            };

            const driverObj = window.driver.js.driver({
                showProgress: false,
                animate: true,
                popoverClass: 'driver-popover',
                steps: [introStep],
                onHighlighted: () => {
                    setTimeout(() => {
                        const goBtn = document.getElementById('lg-tour-l3-go');
                        const dismissBtn = document.getElementById('lg-tour-l3-dismiss');
                        if (goBtn && !goBtn._wired) {
                            goBtn._wired = true;
                            goBtn.addEventListener('click', () => { continueToFull = true; driverObj.destroy(); });
                        }
                        if (dismissBtn && !dismissBtn._wired) {
                            dismissBtn._wired = true;
                            dismissBtn.addEventListener('click', () => {
                                localStorage.setItem('ammo_lg_l3_tour_completed', 'true');
                                self.lgL3TourCompleted = true;
                                driverObj.destroy();
                            });
                        }
                    }, 50);
                },
                onDestroyStarted: () => {
                    localStorage.setItem('ammo_lg_l3_tour_completed', 'true');
                    self.lgL3TourCompleted = true;
                    driverObj.destroy();
                },
                onDestroyed: () => {
                    if (continueToFull) {
                        setTimeout(() => {
                            const contentSteps = self._buildL3ContentSteps();
                            if (contentSteps.length === 0) return;
                            const fullDriver = window.driver.js.driver({
                                showProgress: true,
                                animate: true,
                                popoverClass: 'driver-popover',
                                steps: contentSteps,
                                onDestroyStarted: () => fullDriver.destroy(),
                            });
                            fullDriver.drive();
                        }, 200);
                    }
                },
            });
            driverObj.drive();
        },

        /** Context-sensitive tour — dispatches based on current level */
        startContextTour() {
            if (this.currentLevel === 3) {
                this.startL3Tour();
            } else if (this.currentLevel === 2) {
                this.startL2Tour();
            } else {
                // L1: quick tour if not done, deep dive if quick is done
                if (!localStorage.getItem('ammo_lg_tour_completed')) {
                    this.startQuickTour();
                } else {
                    this.startL1DeepTour();
                }
            }
        },

        // ── Terminal Overlay ───────────────────────────────────────────

        /**
         * Open the terminal overlay for a session.
         * @param {string} [sessionId]
         */
        termOpen(sessionId) {
            if (sessionId) this.termSessionId = sessionId;
            this.termSetMode('half');
        },

        /** Close the terminal overlay. */
        termClose() {
            this.termSetMode('closed');
        },

        /** Toggle half/closed for a session. */
        termToggle(sessionId) {
            if (this.termMode !== 'closed' && this.termSessionId === sessionId) {
                this.termClose();
            } else {
                this.termOpen(sessionId || this.termSessionId);
            }
        },

        /** Switch between closed/half/full. Handles animation class. */
        termSetMode(mode) {
            if (mode === this.termMode) return;
            const wasOpen = this.termMode !== 'closed';
            this.termMode = mode;
            if (mode !== 'closed' && !wasOpen) {
                // Trigger open animation
                this.termAnimating = true;
                setTimeout(() => { this.termAnimating = false; }, 700);
            }
        },

        /**
         * Build the ttyd URL for a session. Matches sessionApp.terminalUrl pattern.
         * @param {string} sessionId
         */
        termUrl(sessionId) {
            if (!sessionId) return '';
            const token = this.apiKey ? `?token=${encodeURIComponent(this.apiKey)}` : '';
            return `/sessions/${sessionId}/terminal${token}`;
        },

        /** Session info for terminal header (from allCards or campaignState). */
        termSessionInfo(sessionId) {
            if (!sessionId) return { modelId: '—', speedup: null, round: null };
            const card = this.allCards.find(c => c.session_id === sessionId);
            if (card) {
                const m = card.campaign?.model_id || card.model_id || sessionId.slice(0, 8);
                const speedup = card.campaign?.cumulative_e2e_speedup;
                const round = card.campaign?.current_round;
                return { modelId: m, speedup, round };
            }
            if (this.campaignState && this.currentSessionId === sessionId) {
                return {
                    modelId: this.campaignState.target?.model_id || sessionId.slice(0, 8),
                    speedup: this.campaignState.campaign?.cumulative_e2e_speedup,
                    round: this.campaignState.campaign?.current_round,
                };
            }
            return { modelId: sessionId.slice(0, 8), speedup: null, round: null };
        },

        // ── Terminal overlay public API (test-facing aliases) ──────────

        /** Open the overlay (alias for termOpen). */
        openTerminalOverlay(sessionId) { this.termOpen(sessionId); },

        /** Close the overlay (alias for termClose). */
        closeTerminalOverlay() { this.termClose(); },

        /** Toggle half/closed. Works without a session (toggles mode only). */
        toggleTerminalOverlay(sessionId) {
            if (this.termMode !== 'closed') {
                this.termClose();
            } else {
                const sid = sessionId || this.termSessionId || this.currentSessionId;
                if (sid) {
                    this.termOpen(sid);
                } else {
                    this.termSetMode('half'); // open without session for toggle-only use
                }
            }
        },

        /** Set size: 'half' | 'full' | 'closed' (alias for termSetMode). */
        setTerminalSize(size) { this.termSetMode(size); },

        /** Toggle copy mode — calls tmux mouse-mode API then updates local state. */
        async toggleTermCopyMode() {
            const sid = this.termSessionId;
            if (!sid) return;
            // copy mode ON = mouse mode OFF (so user can select text)
            const desired = this.termCopyMode ? 'on' : 'off';
            try {
                const resp = await this.apiFetch(`/sessions/${sid}/tmux-mouse-mode`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: desired }),
                });
                if (!resp.ok) return;
                const result = await resp.json();
                // mouse_mode === 'off' means copy mode is active
                this.termCopyMode = result.mouse_mode === 'off';
            } catch (e) { /* graceful: no change */ }
        },

        /** Return the ttyd iframe URL for a session (alias for termUrl). */
        terminalIframeSrc(sessionId) {
            if (!sessionId) return '';
            return this.termUrl(sessionId);
        },

        /**
         * Handle a keyboard event — used by tests and the x-on:keydown binding.
         * Esc → close; Ctrl+` → toggle.
         */
        handleKeydown(event) {
            if (event.key === 'Escape' && this.termMode !== 'closed') {
                event.preventDefault && event.preventDefault();
                this.termMode === 'full' ? this.termSetMode('half') : this.termClose();
            }
            if (event.key === '`' && event.ctrlKey) {
                event.preventDefault && event.preventDefault();
                this.toggleTerminalOverlay(this.termSessionId || this.currentSessionId);
            }
        },

        // ── Router ─────────────────────────────────────────────────────
        _onHashChange() {
            const hash = window.location.hash.slice(1); // strip leading #
            const parsed = this._parseHash(hash);

            if (!parsed) {
                // Not a campaign route — hide campaign view, show old session UI
                document.body.classList.remove('lg-active');
                if (this.currentLevel !== 0) {
                    this.currentLevel = 0;
                    this.stopPolling();
                }
                return;
            }

            // Campaign route active — hide old session UI
            document.body.classList.add('lg-active');

            const { sessionId, round, node, artifact } = parsed;

            if (node && round != null && !isNaN(round) && sessionId) {
                if (artifact) this.pendingArtifactPath = artifact;
                this._activateL3(sessionId, round, node);
            } else if (sessionId) {
                this._activateL2(sessionId);
            } else {
                this._activateL1();
            }
        },

        _parseHash(hash) {
            // matches: campaigns, campaigns/{id}, campaigns/{id}/{round}/{node}
            // Stage nodes use the `stage-N` format: stage-0=Baseline, stage-1=Mining,
            // stage-2=Debate, stage-3/4=Implementation, stage-5=Integration, stage-6=Eval.
            // Track nodes use the op_id directly (e.g. op_001). Round is 1-indexed integer.
            // Optional query string supports deep-links like ?artifact=<path>.
            const qIdx = hash.indexOf('?');
            const pathPart = qIdx >= 0 ? hash.slice(0, qIdx) : hash;
            const queryPart = qIdx >= 0 ? hash.slice(qIdx + 1) : '';
            const match = pathPart.match(/^campaigns(?:\/([^/]+)(?:\/([^/]+)\/([^/]+))?)?$/);
            if (!match) return null;
            let artifact = null;
            if (queryPart) {
                try {
                    const params = new URLSearchParams(queryPart);
                    artifact = params.get('artifact');
                } catch (_) { /* ignore malformed query */ }
            }
            return {
                sessionId: match[1] || null,
                round: match[2] ? parseInt(match[2], 10) : null,
                node: match[3] || null,
                artifact,
            };
        },

        /** Trigger zoom-crossfade animation on the content area.
         *  direction: 'in' (forward: L1→L2, L2→L3) or 'out' (back: L3→L2, L2→L1) */
        _triggerLevelFade(direction) {
            const el = document.querySelector('.lg-level-content');
            if (!el) return;
            el.classList.remove('fade-in');
            el.removeAttribute('data-direction');
            void el.offsetWidth; // force reflow
            if (direction === 'out') el.setAttribute('data-direction', 'out');
            el.classList.add('fade-in');
        },

        _activateL1() {
            this._triggerLevelFade(this.currentLevel > 1 ? 'out' : 'in');
            this.currentLevel = 1;
            this._cbLastDataKey = null; // Reset so L2 re-renders fresh on next visit
            this.currentSessionId = null;
            this.currentRound = null;
            this.currentNode = null;
            this.loadCampaigns();
            this.loadServerInfo();
            this.loadVersion();
            this.startPolling(null);
        },

        _activateL2(sessionId) {
            this._triggerLevelFade(this.currentLevel > 2 ? 'out' : 'in');
            this._cbLastDataKey = null;
            this.currentLevel = 2;
            this.currentSessionId = sessionId;
            this.currentRound = null;
            this.currentNode = null;
            this.l3OpenedTabs = [];  // Reset lazy tab cache on L3 exit
            this.campaignState = null;    // Clear stale campaign data before loading new
            this.artifactCatalog = null;
            this.loadCampaignDetail(sessionId);
            if (!this.allSessions.length) this.loadSessions();
            this.startPolling(sessionId);
            // Auto-trigger L2 tour on first visit
            setTimeout(() => {
                if (!localStorage.getItem('ammo_lg_l2_tour_completed')) {
                    this.startL2Tour();
                }
            }, 500);
        },

        _activateL3(sessionId, round, node) {
            // Gate: refuse to land on an unreachable stage. Covers hand-edited
            // URLs like #campaigns/{sid}/2/stage-5 when round 2 hasn't reached
            // integration yet. For track nodes (non-stage-*), the stage-based
            // gate is too coarse — we allow those through (track reachability
            // is enforced separately via the track's presence in parallel_tracks).
            if (this.campaignState && node && node.startsWith('stage-')) {
                const colIdx = parseInt(node.slice('stage-'.length), 10);
                if (!Number.isNaN(colIdx) && !this._stageReachable(round, colIdx)) {
                    console.warn(`[lg-nav] stage-${colIdx} not yet reachable for round ${round} (stage=${currentStage(this.campaignState)}); redirecting to L2`);
                    this.navigateTo(2, sessionId);
                    return;
                }
            }
            this._triggerLevelFade('in');
            const prevSessionId = this.currentSessionId;
            const prevNode = this.currentNode;
            const prevRound = this.currentRound;
            this.currentLevel = 3;
            this.currentSessionId = sessionId;
            this.currentRound = round;
            this.currentNode = node;
            this.stopPolling();
            // If we don't have state yet or switched campaigns, load fresh
            if (!this.campaignState || prevSessionId !== sessionId) {
                this.loadCampaignDetail(sessionId);
            } else {
                // State already cached — recompute catalog data for this node
                this.l3CatalogData = this._buildL3CatalogData();
                // Re-init artifact sections when switching nodes or rounds within L3
                // (x-init only fires once, so pipeline stage clicks need this)
                if (prevNode !== node || prevRound !== round) {
                    this.$nextTick(() => this.initL3Sections());
                }
            }
            // Auto-trigger L3 tour on first visit
            setTimeout(() => {
                if (!localStorage.getItem('ammo_lg_l3_tour_completed')) {
                    this.startL3Tour();
                }
            }, 500);
        },

        // ── Navigation ─────────────────────────────────────────────────
        navigateTo(level, sessionId = null, round = null, node = null) {
            if (level === 1) {
                window.location.hash = 'campaigns';
            } else if (level === 2 && sessionId) {
                window.location.hash = `campaigns/${sessionId}`;
            } else if (level === 3 && sessionId && round != null && node) {
                window.location.hash = `campaigns/${sessionId}/${round}/${node}`;
            } else if (level === 0) {
                // Return to original sessions view
                window.location.hash = '';
            }
        },

        navigateBack() {
            if (this.currentLevel === 3) {
                this.navigateTo(2, this.currentSessionId);
            } else if (this.currentLevel === 2) {
                this.navigateTo(1);
            } else {
                this.navigateTo(0);
            }
        },

        // ── Data Loading ───────────────────────────────────────────────
        async loadCampaigns() {
            this.loading = true;
            this.loadError = null;
            try {
                // Fetch both campaigns and all sessions in parallel
                const [campResp, sessResp] = await Promise.all([
                    this.apiFetch('/api/campaigns'),
                    this.apiFetch('/sessions'),
                ]);
                if (!campResp.ok) throw new Error(`/api/campaigns HTTP ${campResp.status}`);
                if (!sessResp.ok) throw new Error(`/sessions HTTP ${sessResp.status}`);
                const campData = await campResp.json();
                const sessData = await sessResp.json();
                this.campaignOverviews = campData.campaigns || [];
                this.allSessions = sessData.sessions || [];
                this.allCards = this.mergeSessionsAndCampaigns(this.allSessions, this.campaignOverviews);
            } catch (e) {
                console.error('[campaignApp] loadCampaigns failed:', e);
                this.loadError = 'Failed to load campaigns';
            } finally {
                this.loading = false;
            }
        },

        async loadSessions() {
            try {
                const resp = await this.apiFetch('/sessions');
                if (!resp.ok) return;
                const data = await resp.json();
                this.allSessions = data.sessions || [];
                this.allCards = this.mergeSessionsAndCampaigns(this.allSessions, this.campaignOverviews);
                // Auto-close terminal if the viewed session is no longer active
                if (this.termSessionId && this.termMode !== 'closed' && !this.isSessionActive(this.termSessionId)) {
                    this.termClose();
                }
            } catch (e) {
                // graceful: stale data shown
            }
        },

        /**
         * Merge /sessions list with /api/campaigns data.
         * - Excludes terminated sessions.
         * - Attaches campaign data where available.
         * - Sorts: active → creating → paused.
         */
        mergeSessionsAndCampaigns(sessions, campaigns) {
            const campMap = {};
            for (const c of campaigns) {
                const flat = c.campaign
                    ? { ...c.campaign, target: c.target, created_at: c.created_at }
                    : c;
                campMap[c.session_id] = flat;
            }
            const STATUS_ORDER = { active: 0, creating: 1, building: 1, paused: 2 };
            return sessions
                .filter(s => s.status !== 'terminated' && s.status !== 'failed')
                .map(s => {
                    const card = {
                        ...s,
                        hasCampaign: s.session_id in campMap,
                        campaign: campMap[s.session_id] || null,
                    };
                    if (card.campaign) {
                        // Artifact-Layout-V2 Task 5: server keeps state.json
                        // verbatim — apply the v3 → legacy field fallback so
                        // L1 cards display the right speedup.
                        _normalizeCumulativeSpeedup(card.campaign);
                        // B1 (plan §6): the L1 projection nests target as
                        // `campaign.target.{model_id, dtype, tp}` but FE
                        // readers expect flat `campaign.{model_id, dtype, tp}`.
                        // Hoist when the flat field is absent; never overwrite
                        // a real value.
                        if (card.campaign.target) {
                            const t = card.campaign.target;
                            if (card.campaign.model_id == null && t.model_id != null) {
                                card.campaign.model_id = t.model_id;
                            }
                            if (card.campaign.dtype == null && t.dtype != null) {
                                card.campaign.dtype = t.dtype;
                            }
                            if (card.campaign.tp == null && t.tp != null) {
                                card.campaign.tp = t.tp;
                            }
                            if (card.campaign.hardware == null && t.hardware != null) {
                                card.campaign.hardware = t.hardware;
                            }
                        }
                        // Compute pipeline_progress fallback when the server
                        // omits it (transition compat). When present, the
                        // server-supplied value wins.
                        if (!Array.isArray(card.campaign.pipeline_progress)
                                || !card.campaign.pipeline_progress.length) {
                            card.campaign.pipeline_progress =
                                _buildPipelineProgress(card.campaign);
                        }
                        // Compute track counts when the server omits them.
                        if (card.campaign.shipped_count == null) {
                            const counts = _countTrackStatuses({campaign: card.campaign});
                            card.campaign.shipped_count = counts.shipped;
                            card.campaign.failed_count = counts.failed;
                            card.campaign.active_count = counts.active;
                        }
                    }
                    return card;
                })
                .sort((a, b) => {
                    const oa = STATUS_ORDER[a.status] ?? 99;
                    const ob = STATUS_ORDER[b.status] ?? 99;
                    return oa - ob;
                });
        },

        async loadCampaignDetail(sessionId) {
            if (this._loadingCampaignDetail === sessionId) return;
            this._loadingCampaignDetail = sessionId;
            this.loading = true;
            this.loadError = null;
            try {
                // Sidecar removal (2026-05-27): campaign-data endpoint now
                // returns `{state}` only — no artifact_catalog or sidecars.
                // The artifact tree comes from a separate endpoint.
                const [stateResp, treeResp] = await Promise.all([
                    this.apiFetch(`/api/campaign-data/${sessionId}`).catch(() => null),
                    this.apiFetch(`/api/campaigns/${sessionId}/tree`).catch(() => null),
                ]);
                let resp = stateResp;
                if (resp && resp.ok) {
                    const data = await resp.json();
                    this.campaignState = data.state || data;
                    // Port of server's legacy `_normalize_speedup_field`. The
                    // server keeps state.json verbatim now, so we own the
                    // v3 → legacy field fallback. See plan Task 5.
                    if (this.campaignState && this.campaignState.campaign) {
                        _normalizeCumulativeSpeedup(this.campaignState.campaign);
                    }
                } else if (resp && resp.status === 404) {
                    // Fallback: server hasn't deployed Phase 3 yet
                    resp = await this.apiFetch(`/api/campaigns/${sessionId}`);
                    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                    this.campaignState = await resp.json();
                    if (this.campaignState && this.campaignState.campaign) {
                        _normalizeCumulativeSpeedup(this.campaignState.campaign);
                    }
                } else if (!resp) {
                    throw new Error('Network error');
                } else {
                    throw new Error(`HTTP ${resp.status}`);
                }
                // Tree response: `{root, files: [path, ...]}` — fed straight
                // into _catalogEntries which derives labels from path.
                if (treeResp && treeResp.ok) {
                    try { this.artifactCatalog = await treeResp.json(); }
                    catch (_) { this.artifactCatalog = null; }
                } else {
                    this.artifactCatalog = null;
                }
            } catch (e) {
                console.error('[campaignApp] loadCampaignDetail failed:', e);
                // For active/creating sessions, 404 means campaign hasn't started yet
                const card = this.allCards.find(c => c.session_id === sessionId);
                const sessionStatus = (card?.status || '').toLowerCase();
                if (sessionStatus === 'active' || sessionStatus === 'creating') {
                    this.loadError = 'initializing';
                } else {
                    this.loadError = 'Failed to load campaign detail';
                }
            } finally {
                this.loading = false;
                this._loadingCampaignDetail = null;
                // Recompute catalog data if L3 is active (state may have arrived after initL3Sections)
                if (this.currentLevel === 3) {
                    this.l3CatalogData = this._buildL3CatalogData();
                }
            }
        },

        /**
         * Fetch a text artifact for L3 rendering.
         * @param {string} sessionId
         * @param {string} relPath  — relative path within the artifact dir
         * Returns { content, mime } or null on failure.
         * Binary artifacts (.nsys-rep, .png, etc.) skip .text() and return
         * { content: null, mime, binary: true, size } so callers can render
         * a download-only placeholder instead of garbled bytes.
         */
        artifactUrl(path, sessionId = this.currentSessionId) {
            return buildArtifactUrl(sessionId, path);
        },

        async fetchArtifact(sessionId, relPath) {
            try {
                const resp = await this.apiFetch(this.artifactUrl(relPath, sessionId));
                if (!resp.ok) return null;
                const mime = (resp.headers.get('content-type') || 'text/plain').split(';')[0].trim();
                if (this._isBinaryArtifact(relPath, mime)) {
                    const size = parseInt(resp.headers.get('content-length') || '0', 10) || 0;
                    return { content: null, mime, binary: true, size };
                }
                const content = await resp.text();
                return { content, mime, binary: false };
            } catch (e) {
                return null;
            }
        },

        /** Last path extension (lowercased), e.g. '.py'; '' when none. */
        _extOf(path) {
            const m = (path || '').match(/\.[^./\\]+$/);
            return m ? m[0].toLowerCase() : '';
        },

        /** True if the extension belongs to an inline-renderable image format. */
        _isImageExt(ext) {
            return /^\.(png|jpe?g|gif|webp|bmp|ico|svg)$/i.test(ext || '');
        },

        /** Extension → hljs language. Returns '' for unknown extensions
         *  so callers can fall through to `hljs.highlightAuto()`. */
        extToLang(ext) {
            const MAP = {
                '.py':    'python',
                '.cu':    'cpp', '.cuh': 'cpp',
                '.cpp':   'cpp', '.cc':  'cpp', '.cxx': 'cpp',
                '.h':     'cpp', '.hpp': 'cpp',
                '.c':     'c',
                '.js':    'javascript', '.jsx': 'javascript',
                '.ts':    'typescript', '.tsx': 'typescript',
                '.sh':    'bash', '.bash': 'bash',
                '.yaml':  'yaml', '.yml':  'yaml',
                '.json':  'json',
                '.toml':  'ini',  '.ini':  'ini',
                '.md':    'markdown',
                '.rs':    'rust',
                '.go':    'go',
                '.cmake': 'cmake',
                '.diff':  'diff',
                '.patch': 'diff',
            };
            return MAP[(ext || '').toLowerCase()] || '';
        },

        /** Decide whether an image should render inline.
         *  True iff the extension is a known image AND a numeric size
         *  ≤ 5 MB is available. Missing size → safe-default false. */
        _shouldInlineImage(path, mime, size) {
            const ext = this._extOf(path);
            if (!this._isImageExt(ext)) return false;
            if (typeof size !== 'number' || size <= 0) return false;
            return size <= 5 * 1024 * 1024;
        },

        /** True if artifact should be treated as binary (skip .text() decode).
         *  Images are deliberately excluded — they take the inline path via
         *  `_shouldInlineImage()`. Weight-file extensions (.safetensors, .pt,
         *  .onnx, ...) are included so their raw bytes never reach
         *  `resp.text()`. */
        _isBinaryArtifact(path, mime) {
            const ext = this._extOf(path);
            if (this._isImageExt(ext)) return false;
            const m = (mime || '').toLowerCase();
            if (m && !m.startsWith('text/') &&
                m !== 'application/json' && m !== 'application/javascript' &&
                m !== 'application/xml' && !m.endsWith('+json') && !m.endsWith('+xml')) {
                return true;
            }
            const BIN_EXT = /\.(nsys-rep|qdrep|ncu-rep|sqlite|db|pdf|zip|gz|tar|tgz|bz2|xz|7z|pb|bin|so|o|a|dll|dylib|pt|pth|safetensors|onnx|pkl|npy|npz|parquet|arrow)$/i;
            return BIN_EXT.test(path || '');
        },

        /** Human-readable byte size. */
        humanBytes(n) {
            if (!n || n < 0) return '—';
            const units = ['B', 'KB', 'MB', 'GB', 'TB'];
            let i = 0, v = n;
            while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
            return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
        },

        /**
         * Render markdown content safely using DOMPurify + marked.
         * Falls back to plain text if marked is not loaded.
         */
        renderMarkdown(content) {
            if (!content) return '';
            const raw = (typeof marked !== 'undefined')
                ? marked.parse(content)
                : content.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
            return (typeof DOMPurify !== 'undefined')
                ? DOMPurify.sanitize(raw)
                : raw;
        },

        /**
         * Build L3 display data from campaignState for a given (roundId, node).
         * node format: 'stage-{colIdx}' or an op_id string.
         */
        buildL3Data(state, roundId, node) {
            if (!state) return null;
            const campaign   = state.campaign || {};
            const tracks     = currentTracks(state);
            const debate     = currentDebate(state);
            const integration = currentIntegration(state);
            const shippedOps = window.LG_HELPERS._normalizeShippedOps(campaign);

            // Parse node: 'stage-{colIdx}' → colIdx
            const colMatch = String(node || '').match(/^stage-(\d+)$/);
            const colIdx   = colMatch ? parseInt(colMatch[1], 10) : -1;

            // Stage names
            const STAGES = ['Baseline', 'Mining', 'Debate', 'Implement', 'Validate', 'Integrate'];
            const stageName = STAGES[colIdx] ?? node;

            // Find the round summary
            const rounds = campaign.rounds || [];
            const roundSummary = findRound(rounds, roundId);
            const isCurrent = !roundSummary || roundId === campaign.current_round;

            // Build track list for this round.
            // Failure tokens include live schema (FAIL / GPU_BLOCKED) and archived
            // schema (FAILED / GPU_BLOCKED) — GPU_BLOCKED is treated as a failure
            // bucket for UI purposes.
            const _FAIL_TOKENS = new Set(['FAILED', 'FAIL', 'GPU_BLOCKED']);
            const trackList = isCurrent
                ? Object.entries(tracks).map(([opId, t]) => ({
                    opId,
                    status: shippedOps.has(opId) ? 'shipped'
                          : _FAIL_TOKENS.has((t.status || '').toUpperCase()) ? 'failed' : 'active',
                    kernelSpeedup: CircuitBoard.kernelSpeedupFromTrack(t),
                    e2eSpeedup:    t.e2e_speedup,
                    verdict:       t.verdict,
                    classification: t.classification,
                    correctness:   t.correctness,
                    failReason:    _failReason(null, t),
                    commitSha:     t.commit_sha,
                    validationResultsPath: t.validation_results_path,
                }))
                : (roundSummary ? (() => {
                    const pastTracks = (roundSummary.parallel_tracks && roundSummary.parallel_tracks.tracks) || {};
                    const selected = roundSummary.debate?.selected_winners
                        || roundSummary.selected_candidates || Object.keys(pastTracks);
                    return selected.map(opId => {
                        const impl = pastTracks[opId] || {};
                        const isFailed = _FAIL_TOKENS.has((impl.status || '').toUpperCase());
                        return {
                            opId,
                            status: (roundSummary.shipped || []).includes(opId) ? 'shipped'
                                  : isFailed ? 'failed' : 'active',
                            kernelSpeedup: CircuitBoard.kernelSpeedupFromTrack(impl),
                            e2eSpeedup: impl.e2e_speedup || roundSummary.cumulative_speedup_after,
                            verdict: impl.verdict || impl.status,
                            classification: impl.classification,
                            failReason: _failReason(impl, null),
                            correctness: impl.correctness,
                            commitSha: impl.commit_sha,
                        };
                    });
                })() : []);

            // If viewing a specific track and it's not in the round-based list, fall back to root tracks
            if (node && !node.startsWith('stage-') && trackList.length === 0 && tracks[node]) {
                const t = tracks[node];
                trackList.push({
                    opId: node,
                    status: shippedOps.has(node) ? 'shipped'
                          : _FAIL_TOKENS.has((t.status || '').toUpperCase()) ? 'failed' : 'active',
                    kernelSpeedup: CircuitBoard.kernelSpeedupFromTrack(t),
                    e2eSpeedup:    t.e2e_speedup,
                    verdict:       t.verdict,
                    classification: t.classification,
                    correctness:   t.correctness,
                    failReason:    _failReason(null, t),
                    commitSha:     t.commit_sha,
                    validationResultsPath: t.validation_results_path,
                });
            }

            // Past-round debate/integration read directly from the round entry.
            const roundDebate = isCurrent ? debate : (roundSummary?.debate || null);
            const roundInteg = isCurrent ? integration : (roundSummary ? (roundSummary.integration || {
                status: (roundSummary.shipped || []).length > 0 ? 'completed' : 'pending',
                passing_candidates: roundSummary.shipped || [],
            }) : null);

            return {
                stageName,
                colIdx,
                roundId,
                isCurrent,
                roundSummary,
                trackList,
                debate:     colIdx === 2 ? roundDebate : null,
                integration: colIdx === 5 ? roundInteg : null,
                target:     state.target || {},
                campaign,
            };
        },

        // ── Polling ────────────────────────────────────────────────────
        startPolling(sessionId) {
            this.stopPolling();
            this._pollInterval = setInterval(async () => {
                try {
                    if (sessionId) {
                        // Sidecar removal: parallel fetch state + tree.
                        const [stateResp, treeResp] = await Promise.all([
                            this.apiFetch(`/api/campaign-data/${sessionId}`).catch(() => null),
                            this.apiFetch(`/api/campaigns/${sessionId}/tree`).catch(() => null),
                        ]);
                        let resp = stateResp;
                        if (resp && resp.ok) {
                            const data = await resp.json();
                            const nextState = data.state || data;
                            const prevRound = this.campaignState?.campaign?.current_round;
                            const nextRound = nextState?.campaign?.current_round;
                            if (typeof prevRound === 'number' && typeof nextRound === 'number' && nextRound < prevRound) {
                                console.warn(`[lightgrid] state rollback detected from round ${prevRound} to ${nextRound}, ignoring update`);
                                return;
                            }
                            // Apply v3 → legacy field fallback before storing
                            // so all downstream readers see a normalized
                            // `cumulative_e2e_speedup`. Plan Task 5.
                            if (nextState && nextState.campaign) {
                                _normalizeCumulativeSpeedup(nextState.campaign);
                            }
                            this.campaignState = nextState;
                        } else if (resp && resp.status === 404) {
                            resp = await this.apiFetch(`/api/campaigns/${sessionId}`);
                            if (resp.ok) {
                                this.campaignState = await resp.json();
                                if (this.campaignState?.campaign) {
                                    _normalizeCumulativeSpeedup(this.campaignState.campaign);
                                }
                            }
                        }
                        if (treeResp && treeResp.ok) {
                            try { this.artifactCatalog = await treeResp.json(); }
                            catch (_) { /* keep stale tree on parse error */ }
                        }
                    } else {
                        const resp = await this.apiFetch('/api/campaigns');
                        if (!resp.ok) return;
                        const data = await resp.json();
                        this.campaignOverviews = data.campaigns || [];
                        await this.loadSessions();
                    }
                } catch (e) {
                    // graceful: stale data shown
                }
            }, 15000);
        },

        stopPolling() {
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
            }
        },

        // ── API Fetch (mirrors sessionApp.apiFetch) ────────────────────
        async apiFetch(url, options = {}) {
            const headers = {
                ...options.headers,
                'X-Client-ID': this.clientId,
            };
            if (this.apiKey) {
                headers['Authorization'] = `Bearer ${this.apiKey}`;
            }
            const resp = await fetch(url, { ...options, headers });
            // If 401 and we have a stored key, it's stale — show login
            if (resp.status === 401 && this.apiKey) {
                this.apiKey = '';
                this.showLoginModal = true;
            }
            return resp;
        },

        // ── Auth: check / login / logout ──────────────────────────────
        async _checkAuthRequired() {
            try {
                const headers = { 'X-Client-ID': this.clientId };
                if (this.apiKey) {
                    headers['Authorization'] = `Bearer ${this.apiKey}`;
                }
                const resp = await fetch('/sessions', { headers });
                if (resp.status === 401) {
                    if (this.apiKey) this.apiKey = '';
                    this.showLoginModal = true;
                    return true;
                }
            } catch (e) { /* network error — proceed without auth */ }
            return false;
        },

        async submitLogin() {
            this.loginError = '';
            this.loginChecking = true;
            try {
                const resp = await fetch('/sessions', {
                    headers: {
                        'Authorization': `Bearer ${this.loginKeyInput}`,
                        'X-Client-ID': this.clientId,
                    },
                });
                if (resp.ok) {
                    this.apiKey = this.loginKeyInput;
                    this.loginKeyInput = '';
                    this.showLoginModal = false;
                    this.loginError = '';
                    this._postAuthInit();
                } else {
                    this.loginError = 'Invalid API key';
                }
            } catch (e) {
                this.loginError = 'Connection error. Please try again.';
            } finally {
                this.loginChecking = false;
            }
        },

        logout() {
            this.apiKey = '';
            location.reload();
        },

        // ── Report viewer ─────────────────────────────────────────────
        // Delegates to sessionApp's reportOverlay via custom event (the overlay
        // DOM lives in sessionApp scope, so we dispatch rather than duplicate).
        // Forwards gpu_count + status too so sessionApp can populate the
        // Console header spec strip (SESSION · MODEL · GPU · STATUS).
        openReport(session) {
            const sid = session.session_id || session;
            const sessionObj = {
                session_id: sid,
                model_id: session.model_id || '',
                dtype: session.dtype || '',
                gpu_count: (typeof session.gpu_count === 'number') ? session.gpu_count : null,
                status: session.status || '',
            };
            window.dispatchEvent(new CustomEvent('campaign-open-report', { detail: sessionObj }));
        },

        closeReport() {
            window.dispatchEvent(new CustomEvent('campaign-close-report'));
        },

        // ── Card status helpers ────────────────────────────────────────
        /**
         * Returns CSS modifier class for status-based card styling.
         * active → shimmer+breathing, paused → opacity 0.82, creating → sweep.
         */
        cardClass(session) {
            const s = (session && session.status) ? session.status.toLowerCase() : '';
            if (s === 'active') return 'active';
            if (s === 'paused') return 'paused';
            if (s === 'creating') return 'creating';
            return '';
        },

        /**
         * Open the terminal overlay for a session.
         * Uses the LIGHTGRID terminal overlay (not the legacy hash route).
         * Delegates paused sessions through resumeAndOpenTerminal() so the
         * user is never dead-ended with a "session is paused" toast.
         */
        openTerminal(sessionId, card) {
            this.resumeAndOpenTerminal(sessionId, card);
        },

        /**
         * Resume (if needed) then open the terminal overlay.
         * - active  → termOpen immediately
         * - paused  → POST /resume, wait for state flip, then termOpen
         * - unknown → fall through to termOpen (lets backend handle it)
         */
        async resumeAndOpenTerminal(sessionId, card) {
            if (!sessionId) return;
            const ref = card || this.allCards.find(c => c.session_id === sessionId);
            const status = (ref && ref.status ? ref.status : '').toLowerCase();

            if (status !== 'paused') {
                this.termOpen(sessionId);
                return;
            }

            if (this.loadingActions[sessionId] === 'resuming') {
                this.showToast('Resume already in progress\u2026');
                return;
            }

            this.loadingActions = { ...this.loadingActions, [sessionId]: 'resuming' };
            this.showToast('Resuming session\u2026 this can take ~10s', 6000);
            try {
                const r = await this.apiFetch(`/sessions/${sessionId}/resume`, { method: 'POST' });
                if (!r.ok) {
                    let detail = 'Resume failed';
                    try { const e = await r.json(); detail = e.detail || detail; } catch {}
                    throw new Error(detail);
                }
                await this.loadSessions();
                const now = this.allCards.find(c => c.session_id === sessionId);
                if (now && now.status === 'active') {
                    this.showToast('Session resumed \u2014 connecting\u2026', 2500);
                    this.termOpen(sessionId);
                } else {
                    this.showToast('Resume started \u2014 retry in a moment', 4000);
                }
            } catch (e) {
                console.error('Resume:', e.message);
                this.showToast('Resume failed: ' + e.message, 5000);
            } finally {
                const a = { ...this.loadingActions };
                delete a[sessionId];
                this.loadingActions = a;
            }
        },

        // ── Session Actions (ported from sessionApp) ──────────────────
        async pauseSession(sid) {
            this.loadingActions = { ...this.loadingActions, [sid]: 'pausing' };
            try {
                const r = await this.apiFetch(`/sessions/${sid}/pause`, { method: 'POST' });
                if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Pause failed'); }
                if (this.termSessionId === sid) this.termClose();
                await this.loadSessions();
            } catch (e) { console.error('Pause:', e.message); this.showToast('Pause failed: ' + e.message); }
            finally { const a = { ...this.loadingActions }; delete a[sid]; this.loadingActions = a; }
        },
        async resumeSession(sid) {
            this.loadingActions = { ...this.loadingActions, [sid]: 'resuming' };
            try {
                const r = await this.apiFetch(`/sessions/${sid}/resume`, { method: 'POST' });
                if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Resume failed'); }
                await this.loadSessions();
            } catch (e) { console.error('Resume:', e.message); this.showToast('Resume failed: ' + e.message); }
            finally { const a = { ...this.loadingActions }; delete a[sid]; this.loadingActions = a; }
        },
        async terminateSession(sid) {
            if (this.confirmTerminate !== sid) {
                this.confirmTerminate = sid;
                setTimeout(() => { if (this.confirmTerminate === sid) this.confirmTerminate = null; }, 3000);
                return;
            }
            this.confirmTerminate = null;
            this.loadingActions = { ...this.loadingActions, [sid]: 'terminating' };
            try {
                const r = await this.apiFetch(`/sessions/${sid}`, { method: 'DELETE' });
                if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Terminate failed'); }
                if (this.termSessionId === sid) this.termClose();
                await this.loadSessions();
            } catch (e) { console.error('Terminate:', e.message); this.showToast('Terminate failed: ' + e.message); }
            finally { const a = { ...this.loadingActions }; delete a[sid]; this.loadingActions = a; }
        },
        async prepareDownload(sid) {
            this.downloadingSession = sid;
            try {
                const r = await this.apiFetch(`/sessions/${sid}/prepare-download`, { method: 'POST' });
                const d = await r.json();
                if (!r.ok) throw new Error(d.detail || d.error || 'Download failed');
                if (d.download_url) window.location.href = d.download_url;
                else throw new Error(d.error || 'No download URL');
            } catch (e) { console.error('Download:', e.message); }
            finally { this.downloadingSession = null; }
        },
        isActionLoading(sid) { return !!this.loadingActions[sid]; },
        actionLabel(sid) { return this.loadingActions[sid] || ''; },

        // ── Eval decision derivation ──────────────────────────────────
        evalDecision(round) {
            if (!round || !round.status) return 'CONTINUE';
            if (round.status === 'EXHAUSTED' || round.status === 'FAILED') return 'STOP';
            return round.status.toUpperCase();
        },

        // ── Helpers ────────────────────────────────────────────────────
        formatSpeedup(val) {
            const n = Number(val);
            if (!Number.isFinite(n) || n <= 1) return '1.00x';
            return n.toFixed(2) + 'x';
        },

        /**
         * Coerce a possibly-string/null numeric field to Number and format via toFixed,
         * returning `fallback` when the value is not a finite number. Campaign state.json
         * and artifact catalogs sometimes serialize speedups / timings as strings
         * ("1.42", "3.6e-2") which pass truthy checks but crash .toFixed calls.
         */
        fmtNum(val, digits = 2, fallback = '\u2014') {
            const n = Number(val);
            if (!Number.isFinite(n)) return fallback;
            return n.toFixed(digits);
        },

        /**
         * Convert snake_case metric keys to readable Title Case labels.
         * E.g., "baseline_avg_s (bs=1)" → "Baseline Avg (bs=1)"
         * Strips trailing _s (seconds unit shown on value side).
         */
        formatMetricLabel(key) {
            if (!key) return '';
            const match = key.match(/^([^(]+?)(\s*\(.*\))?$/);
            let base = (match ? match[1] : key).trim();
            const suffix = match ? (match[2] || '') : '';
            base = base.replace(/_s$/, '');
            const words = base.split('_').map(w => {
                if (w === 'opt') return 'Optimized';
                if (w === 'avg') return 'Avg';
                if (w === 'e2e') return 'E2E';
                if (/^p\d+$/i.test(w)) return w.toUpperCase();
                return w.charAt(0).toUpperCase() + w.slice(1);
            });
            return words.join(' ') + suffix;
        },

        statusColor(status) {
            switch ((status || '').toLowerCase()) {
                case 'active':    return 'var(--mint)';
                case 'complete':
                case 'completed': return 'var(--mint)';
                case 'failed':    return 'var(--red)';
                case 'paused':    return 'var(--amber)';
                default:          return 'var(--ghost)';
            }
        },

        /** Check if a session is active (for terminal FAB visibility + the
         * loadSessions auto-close guard). 'building' counts as watchable so the
         * live build console isn't force-closed on the next poll. */
        isSessionActive(sid) {
            if (!sid) return false;
            const card = this.allCards.find(c => c.session_id === sid);
            return card && (card.status === 'active' || card.status === 'building');
        },

        pipelineStageClass(stage) {
            // stage.status: completed | active | failed | paused | pending/ghost
            switch (stage.status) {
                case 'completed': return 'completed';
                case 'active':    return 'active';
                case 'failed':    return 'failed';
                case 'paused':    return 'paused';
                default:          return 'ghost';
            }
        },

        // ── Server / GPU info ────────────────────────────────────────

        async loadServerInfo() {
            try {
                const hResp = await this.apiFetch('/health');
                const health = hResp.ok ? await hResp.json() : {};
                const gm = health.gpu_manager || {};
                this.serverInfo = {
                    gpuType: health.gpu?.type || health.gpu_type || 'GPU',
                    totalGpus: gm.total_gpus || health.gpu?.total_gpus || 0,
                    availableGpus: gm.available_gpus || health.gpu?.available_gpus || 0,
                    serverName: 'local server',
                };
            } catch (e) { /* graceful */ }
        },
        get gpuLabel() {
            if (!this.serverInfo || !this.serverInfo.totalGpus) return '';
            return this.serverInfo.totalGpus + 'x ' + this.serverInfo.gpuType;
        },
        get gpuFreeLabel() {
            if (!this.serverInfo || !this.serverInfo.totalGpus) return '';
            return this.serverInfo.availableGpus + ' / ' + this.serverInfo.totalGpus + ' free';
        },
        get serverFooterLabel() {
            if (!this.serverInfo) return this.allCards.length + ' sessions';
            const ci = this.serverInfo;
            return ci.serverName + ' online';
        },

        // ── Version info ─────────────────────────────────────────────
        ammoVersion: null,
        changelogData: null,
        showChangelog: false,

        async loadVersion() {
            try {
                const resp = await this.apiFetch('/api/changelog');
                if (!resp.ok) return;
                const data = await resp.json();
                this.ammoVersion = data.version || null;
                this.changelogData = data.entries || [];
            } catch (e) { /* graceful */ }
        },
        get versionLabel() {
            return this.ammoVersion ? 'v' + this.ammoVersion : 'v2.1.0';
        },

        // ── Cleanup + Toast ──────────────────────────────────────────
        cleaningUp: false,
        toastMsg: '',
        _toastTimer: null,

        async cleanupTerminated() {
            if (this.cleaningUp) return;
            this.cleaningUp = true;
            try {
                const resp = await this.apiFetch('/sessions/terminated', { method: 'DELETE' });
                if (!resp.ok) throw new Error('Cleanup failed');
                const data = await resp.json().catch(() => ({}));
                const count = data.deleted_count || data.count || 0;
                this.showToast(count > 0 ? count + ' terminated session' + (count !== 1 ? 's' : '') + ' removed' : 'No terminated sessions to clean up');
                await this.loadSessions();
            } catch (e) {
                this.showToast('Cleanup failed: ' + e.message);
            } finally {
                this.cleaningUp = false;
            }
        },

        showToast(msg, duration = 3000) {
            this.toastMsg = msg;
            clearTimeout(this._toastTimer);
            this._toastTimer = setTimeout(() => { this.toastMsg = ''; }, duration);
        },

        // ── L1 Grid helpers ──────────────────────────────────────────

        /** Section collapse state for paused campaigns */
        pausedSectionCollapsed: false,

        togglePausedSection() {
            this.pausedSectionCollapsed = !this.pausedSectionCollapsed;
        },

        /** Active + creating + building cards (shown in ACTIVE section).
         * 'building' is a fork from-source build (~15-20 min, holds GPUs) — a
         * setup phase like 'creating', so it lives in the ACTIVE section and is
         * clickable to open its live build console. */
        get activeCards() {
            return this.allCards.filter(c => c.status === 'active' || c.status === 'creating' || c.status === 'building');
        },
        /** Paused cards (shown in PAUSED section) */
        get pausedCards() {
            return this.allCards.filter(c => c.status === 'paused');
        },

        /**
         * Trigger entrance animation on a card element.
         * Called from x-init — fires every time Alpine creates a new DOM element
         * (including when a card moves between x-for loops on resume).
         */
        cardEnterAnimation(el, idx) {
            el.classList.add('lg-card-fly-in');
            el.style.animationDelay = (idx * 0.15 + 0.05) + 's';
            const handler = () => {
                el.classList.remove('lg-card-fly-in');
                el.style.animationDelay = '';
                el.removeEventListener('animationend', handler);
            };
            el.addEventListener('animationend', handler);
        },

        /** Status counts for header chips */
        get activeCount() {
            return this.allCards.filter(c => c.status === 'active').length;
        },
        get pausedCount() {
            return this.allCards.filter(c => c.status === 'paused').length;
        },
        get creatingCount() {
            return this.allCards.filter(c => c.status === 'creating' || c.status === 'building').length;
        },

        /** Short model name: strip org prefix, e.g. "nvidia/Qwen3.5-397B" → "Qwen3.5-397B" */
        cardModelName(card) {
            const raw = card.model_name || (card.campaign && card.campaign.model_id);
            if (raw) {
                const parts = raw.split('/');
                return parts[parts.length - 1];
            }
            if (card.config_preset) return card.config_preset;
            if (card.repo_name) return card.repo_name;
            if (card.session_id) return 'Session ' + card.session_id.slice(0, 8);
            return 'Session';
        },

        /** TP size for a card badge (not total pool size) */
        cardTp(card) {
            // After gpu_count decouple, requested_gpu_count may exceed tp_size
            // for parallel experiment tracks — fall back on tp_size so the
            // "TP N" badge never mislabels with the pool total.
            return (card.campaign && card.campaign.tp) || card.tp_size || '?';
        },

        /** Returns vLLM source label for card config tag */
        cardSourceLabel(card) {
            const b = card.branch || '';
            // Gate: when the card is pinned to the Docker image's commit AND the
            // image exposes a release version (v0.20.0 etc), show the release
            // version instead of the short-hash fallback. Any other branch —
            // even another 40-char commit — keeps its own identity.
            if (this.cmVllmVersion && b && b === this.cmDockerCommit) {
                return 'vllm@' + this.cmVllmVersion;
            }
            // 40-char hex = commit hash (non-default / legacy-image stable mode)
            if (/^[0-9a-f]{40}$/i.test(b)) return 'vllm@' + b.slice(0, 7);
            if (b === 'main') return 'vllm@main';
            return 'vllm@' + b;
        },

        /** Returns card state class for visual treatment */
        cardStateClass(card) {
            const action = this.loadingActions[card.session_id];
            if (action === 'resuming') return 'state-creating';
            if (action === 'pausing') return 'state-pausing';
            if (action === 'terminating') return 'state-terminating';
            const s = (card.status || '').toLowerCase();
            if (s === 'active') return 'state-active';
            if (s === 'paused') return 'state-paused';
            if (s === 'creating') return 'state-creating';
            if (s === 'building') return 'state-creating';
            return '';
        },

        /** True if this card is visually in "creating/resuming" transition */
        isResuming(card) {
            return this.loadingActions[card.session_id] === 'resuming';
        },
        isPausing(card) {
            return this.loadingActions[card.session_id] === 'pausing';
        },
        isTerminating(card) {
            return this.loadingActions[card.session_id] === 'terminating';
        },

        /**
         * Returns flat array of pipeline elements (stages + connectors) for
         * rendering in a single x-for loop. Each element is either:
         *   { type:'stage', label:'BA', status:'completed', index:0 }
         *   { type:'connector', status:'completed' }
         */
        cardPipelineElements(card) {
            const stages = card.campaign?.pipeline_progress;
            if (!stages?.length) return [];
            const elements = [];
            stages.forEach((s, i) => {
                if (i > 0) {
                    const left = stages[i - 1].status;
                    const right = s.status || 'ghost';
                    let cls = '';
                    if (left === 'completed' && right === 'completed') cls = 'completed';
                    else if (left === 'completed' && (right === 'active')) cls = 'active';
                    elements.push({ type: 'connector', status: cls });
                }
                elements.push({
                    type: 'stage',
                    label: s.stage.slice(0, 2).toUpperCase(),
                    status: s.status || 'ghost',
                    index: i,
                });
            });
            return elements;
        },

        /** Speedup display text: "1.34×" or "—" */
        cardSpeedupText(card) {
            const n = Number(card.campaign?.cumulative_e2e_speedup);
            if (!Number.isFinite(n) || n <= 0) return '\u2014';
            return n.toFixed(2) + '\u00d7';
        },

        /** Speedup CSS class: mint (good), amber (paused), ghost (none) */
        cardSpeedupClass(card) {
            if (!card.campaign?.cumulative_e2e_speedup) return 'ghost';
            if (card.status === 'paused') return 'amber';
            return 'mint';
        },

        /** Human-readable elapsed time for card footer */
        cardElapsed(card) {
            if (card.status === 'creating') return 'Just now';
            const created = card.created_at;
            if (!created) return '';
            // created_at is a unix timestamp (seconds); convert to ms.
            // Guard against ISO strings / NaN so the label never shows "NaNm".
            const createdNum = typeof created === 'number' ? created : Date.parse(created) / 1000;
            if (!Number.isFinite(createdNum)) return '';
            const createdMs = createdNum > 1e12 ? createdNum : createdNum * 1000;
            const ms = Date.now() - createdMs;
            if (ms < 0) return '';
            const days = Math.floor(ms / 86400000);
            const mins = Math.floor(ms / 60000);
            const hours = Math.floor(mins / 60);
            // 'building' is a from-source compile in progress — label it Building, not Running.
            const verb = card.status === 'building' ? 'Building' : 'Running';
            if (days > 30) {
                return card.status === 'paused' ? 'Paused > 30d ago' : verb + ' > 30d';
            }
            if (card.status === 'paused') {
                if (days > 0) return 'Paused ' + days + 'd ago';
                if (hours > 0) return 'Paused ' + hours + 'h ago';
                return 'Paused ' + mins + 'm ago';
            }
            if (days > 0) return verb + ' ' + days + 'd ' + (hours % 24) + 'h';
            if (hours > 0) return verb + ' ' + hours + 'h ' + String(mins % 60).padStart(2, '0') + 'm';
            return verb + ' ' + mins + 'm';
        },

        /**
         * True when this L1 card has a final report sidecar (kind=report_section)
         * or a shipped report file, indicating the campaign has produced a REPORT.md.
         */
        hasFinalReport(card) {
            if (!card) return false;
            if (card.has_report === true) return true;
            if (card.has_final_report === true) return true;
            const status = (card.campaign?.status || '').toLowerCase();
            if (status === 'completed' && card.has_report !== false) return true;
            return false;
        },

        /** Open the final REPORT for this L1 card via the existing report viewer. */
        openFinalReport(card) {
            if (!card) return;
            // L1 card GPU count lives under a few different fields depending on
            // the card shape (live, past-round, or campaign-only). Pick the first
            // that looks like a positive integer so the Console spec strip gets
            // populated regardless of source.
            let gpuCount = null;
            if (Array.isArray(card.gpu_ids) && card.gpu_ids.length) gpuCount = card.gpu_ids.length;
            else if (typeof card.gpu_count === 'number' && card.gpu_count > 0) gpuCount = card.gpu_count;
            else if (typeof card.requested_gpu_count === 'number' && card.requested_gpu_count > 0) gpuCount = card.requested_gpu_count;
            else if (typeof card.campaign?.tp === 'number' && card.campaign.tp > 0) gpuCount = card.campaign.tp;

            const session = {
                session_id: card.session_id,
                model_id: card.campaign?.model_id || card.model_id || card.model_name,
                dtype: card.campaign?.dtype || card.dtype,
                // Forward gpu_count + status so the Console header spec strip
                // can show SESSION · MODEL · GPU · STATUS (matches sidebar cards).
                gpu_count: gpuCount,
                status: card.status || (card.campaign && card.campaign.status) || '',
            };
            this.openReport(session);
        },

        /** Round badge text: "R6 / 8" or "R0" */
        cardRoundText(card) {
            if (!card.campaign) return '';
            const round = card.campaign.current_round || 0;
            const total = card.campaign.total_rounds;
            if (total) return 'R' + round + ' / ' + total;
            return 'R' + round;
        },

        /** Status text for creating/active-stage/resuming state cards, or null */
        cardStatusLine(card) {
            if (this.loadingActions[card.session_id] === 'resuming') return 'Resuming session...';
            if (card.status === 'creating') return 'Setting up workspace...';
            // Paused cards: show PAUSED or last completed stage
            if (card.status === 'paused') {
                const pipeline = card.campaign && card.campaign.pipeline_progress;
                if (pipeline && pipeline.length > 0) {
                    const STAGE_NAMES = {
                        baseline: 'BASELINE', mining: 'MINING', debate: 'DEBATE',
                        implementation: 'IMPLEMENTATION', validation: 'VALIDATION', integration: 'INTEGRATION',
                    };
                    // Find the last completed stage
                    const completed = pipeline.filter(s => s.status === 'completed');
                    if (completed.length > 0) {
                        const last = completed[completed.length - 1];
                        const key = last.stage.toLowerCase();
                        for (const [k, label] of Object.entries(STAGE_NAMES)) {
                            if (key.includes(k)) return 'PAUSED \u2014 ' + label;
                        }
                    }
                }
                return 'PAUSED';
            }
            const pipeline = card.campaign && card.campaign.pipeline_progress;
            if (pipeline && pipeline.length > 0) {
                const active = pipeline.find(s => s.status === 'active');
                if (active) {
                    const STAGE_LABELS = {
                        baseline: 'Baseline profiling...',
                        mining: 'Mining bottlenecks...',
                        debate: 'Debating strategies...',
                        implementation: 'Implementing optimizations...',
                        validation: 'Validating correctness...',
                        integration: 'Integrating changes...',
                    };
                    const key = active.stage.toLowerCase();
                    for (const [k, label] of Object.entries(STAGE_LABELS)) {
                        if (key.includes(k)) return label;
                    }
                    return active.stage + '...';
                }
            }
            return null;
        },

        // ── L3 Artifact Viewer ────────────────────────────────────────────

        /** State for L3 viewer */
        l3Loading: false,
        l3Error: null,
        l3Sections: [],           // [{name, path, content, mime, loaded, error}]
        l3ActiveSection: null,    // currently selected section path
        l3CatalogData: null,      // {track: {...}, stage: {...}} — enriched catalog metrics
        l3OpenedTabs: [],         // paths of tabs whose content has been fetched (lazy rendering)
        pendingArtifactPath: null, // deep-link target; loadArtifact registers it even before browsing
        // Natural, lazy directory browser. Each root/child is a directory node;
        // l3ArtifactRows() flattens only expanded nodes for a non-recursive template.
        l3ArtifactRoots: [],
        l3ArtifactBrowserReady: false,
        l3ArtifactBrowserToken: 0,

        /**
         * True if the given (roundId, colIdx) is reachable from L2 — i.e. the
         * stage has run for that round and clicking it should land on usable L3
         * content. Used by circuit-board to grey out future stages and by the
         * hash router to bounce stealth URLs back to L2.
         *
         * colIdx semantics (matches circuit-board's callback contract):
         *   0 → baseline  / re-profile
         *   1 → mining           (R1 always; R2+ if prev round didn't exhaust/fail)
         *   2 → debate
         *   3,4 → implementation / validation (parallel_tracks)
         *   5 → integration
         *   6 → campaign eval (diamond)
         *
         * Past rounds: reachable if round archive has data for that column.
         * Current round: reachable if colIdx <= current stage index.
         * Future rounds: never reachable.
         */
        _stageReachable(roundId, colIdx) {
            const state = this.campaignState;
            if (!state || roundId == null) return true; // optimistic when state not loaded
            const campaign = state.campaign || {};
            const currentRoundId = campaign.current_round || 1;
            // Future rounds are never reachable
            if (roundId > currentRoundId) return false;

            // Past round — check archived round data
            if (roundId < currentRoundId) {
                const rounds = campaign.rounds || [];
                const r = findRound(rounds, roundId);
                if (!r) return false;
                switch (colIdx) {
                    case 0: return true; // baseline/re-profile always recorded per past round
                    case 1: {
                        // Mining runs in R1 always; in R2+ only if prev round didn't exhaust/fail
                        if (roundId === 1) return true;
                        const prevR = findRound(rounds, roundId - 1);
                        const prevSt = String(prevR?.status || '').toUpperCase();
                        return prevSt !== 'EXHAUSTED' && prevSt !== 'FAILED';
                    }
                    case 2: return true; // debate always happens per past round
                    case 3:
                    case 4: {
                        const pt = (r.parallel_tracks && r.parallel_tracks.tracks) || {};
                        const shipped = r.shipped || [];
                        return Object.keys(pt).length > 0 || shipped.length > 0;
                    }
                    case 5: {
                        const integ = r.integration || null;
                        const integStatus = String(integ?.status || '').toLowerCase();
                        return integStatus && integStatus !== 'pending';
                    }
                    case 6: return true; // past round always has a terminal CONTINUE/STOP decision
                    default: return false;
                }
            }

            // Current round
            const stageKey = currentStage(state) || '1_baseline';
            // stage_order mirrored from orchestration/campaign_data_service.py.
            const STAGE_ORDER = {
                '1_baseline': 0,
                '2_bottleneck_mining': 1,
                '3_debate': 2,
                '4_5_parallel_tracks': 3,
                '6_integration': 5,
                '7_campaign_eval': 6,
                '7b_report': 7,
            };
            const curIdx = STAGE_ORDER[stageKey] ?? 0;

            // colIdx 4 is an artificial split (validation). The server collapses
            // impl+validation into '4_5_parallel_tracks' at idx 3 — so treat both
            // 3 and 4 as reachable when curIdx >= 3.
            if (colIdx === 4) return curIdx >= 3;
            // Mining (colIdx=1): R2+ only if prev round didn't exhaust/fail
            if (colIdx === 1 && roundId >= 2) {
                const rounds = campaign.rounds || [];
                const prevR = findRound(rounds, roundId - 1);
                const prevSt = String(prevR?.status || '').toUpperCase();
                if (prevSt === 'EXHAUSTED' || prevSt === 'FAILED') return false;
            }
            return colIdx <= curIdx;
        },

        /** Highest stage index the campaign has fully COMPLETED.
         *
         *  Used by the L3 pipeline indicator (`frontend/index.html`) which
         *  renders stages as:
         *    idx === currentIdx                → ● current  (the L3 page we're on)
         *    idx <= l3CampaignMaxStage         → ✓ completed (fully done)
         *    else                              → ○ pending   (not yet reached OR in-progress)
         *
         *  Source of truth: iterate every round's per-stage `completed_at`
         *  timestamp (campaign.rounds[*].{baseline,bottleneck_mining,debate,
         *  parallel_tracks,integration,campaign_eval}.completed_at). A stage
         *  is considered "done" once any round's stage object has a
         *  completed_at. Terminal pseudo-stages are gone — campaign.status
         *  drives terminal UI separately. */
        get l3CampaignMaxStage() {
            if (!this.campaignState) return 0;
            const campaign = this.campaignState.campaign || {};
            const rounds = campaign.rounds || [];
            // Map round sub-object key → column index used by the pipeline
            // indicator. parallel_tracks collapses implementation+validation
            // to idx 4 (validation); baseline→0, mining→1, debate→2,
            // integration→5, campaign_eval→6.
            const SUB_TO_IDX = {
                baseline: 0,
                bottleneck_mining: 1,
                debate: 2,
                parallel_tracks: 4,
                integration: 5,
                campaign_eval: 6,
            };
            let max = -1;
            for (const rnd of rounds) {
                if (!rnd || typeof rnd !== 'object') continue;
                for (const [subKey, idx] of Object.entries(SUB_TO_IDX)) {
                    const entry = rnd[subKey];
                    if (entry && entry.completed_at) max = Math.max(max, idx);
                }
            }
            return max >= 0 ? max : 0;
        },

        /** Computed node type for L3 dispatch: baseline|mining|debate|implementation|integration|track
         *
         * Accepts both route forms:
         *   - `stage-N` (N in 0..6) — emitted by the circuit-board stage
         *     chip click handlers (campaign-app.js:4022, 4062, 4066).
         *   - bare stage name (`baseline`, `mining`, `debate`, `integration`)
         *     — shareable/bookmarked URLs and the legacy `buildL2Nodes` ids.
         *     Without this branch, `/1/debate` would fall through the
         *     `stage-` prefix check and be classified as a TRACK, causing
         *     the L3 overview to render the `N/A / N/A / NOT STARTED`
         *     placeholder (Bug 3, 2026-04-21).
         */
        get l3NodeType() {
            const node = this.currentNode;
            if (!node || this.currentLevel !== 3) return null;
            // Audit-gate nodes (`audit-stage_45`) come from the L2 fuse-hex
            // click and the pipeline sub-list. Checked before the track
            // fallthrough so a gate key is never misread as an op_id.
            if (node.startsWith('audit-')) return 'audit';
            // Stage column index by numeric suffix.
            if (node.startsWith('stage-')) {
                const idx = parseInt(node.replace('stage-', ''));
                switch (idx) {
                    case 0: return 'baseline';
                    case 1: return 'mining';
                    case 2: return 'debate';
                    case 3: case 4: return 'implementation';
                    case 5: return 'integration';
                    case 6: return 'integration';
                    default: return null;
                }
            }
            // Bare stage-name aliases — URL-friendly synonyms for stage-N.
            switch (node) {
                case 'baseline':       return 'baseline';
                case 'mining':         return 'mining';
                case 'debate':         return 'debate';
                case 'implementation': return 'implementation';
                case 'validation':     return 'implementation';
                case 'integration':    return 'integration';
                default:               return 'track';
            }
        },

        // ── L3 audit-gate node helpers ──────────────────────────────────
        // An `audit-stage_X` node renders the same L3 modal as stages and
        // tracks; these helpers feed its overview hero, the pipeline
        // sub-list, and the GATE RECORD section from auditGateStates —
        // the same derivation the L2 fuse hexes use, so L2 and L3 can
        // never disagree about a gate's state.

        /** T_AUDIT_SXX designator for a gate key. */
        _auditGateLabel(gateKey) {
            return 'T_AUDIT_' + String(gateKey || '').replace('stage_', 'S').toUpperCase();
        },

        /** What the gate audits, for the GATE RECORD scope row. */
        _auditGateScope(gateKey) {
            return {
                stage_1: 'Stage 1 · Baseline profile',
                stage_2: 'Stage 2 · Bottleneck mining',
                stage_45: 'Stages 4–5 · Parallel tracks',
                stage_67: 'Stages 6–7 · Integration & campaign eval',
            }[gateKey] || gateKey;
        },

        /** Current round's gate records ({stage_1: {state,…}, …}) or {}. */
        _auditGates() {
            const state = this.campaignState;
            if (!state || !window.CircuitBoard?.auditGateStates) return {};
            const byRound = window.CircuitBoard.auditGateStates(state);
            return byRound[Number(this.currentRound) || 1] || {};
        },

        /**
         * Gate rendered as a pipeline sub-item under stage column `idx`,
         * or null. Each gate sits under the last stage it audits: S1 under
         * Baseline, S2 under Mining, S45 under Validation, S67 under
         * Integration. bypass (legacy round without an audit key) renders
         * nothing — same rule as the L2 hex.
         */
        _l3PipelineAuditGate(idx) {
            const KEY_BY_IDX = { 0: 'stage_1', 1: 'stage_2', 4: 'stage_45', 5: 'stage_67' };
            const key = KEY_BY_IDX[idx];
            if (!key) return null;
            const gate = this._auditGates()[key];
            if (!gate || gate.state === 'bypass') return null;
            return { key, ...gate };
        },

        /** Pipeline column the current L3 node highlights as `current`. */
        _l3PipelineCurrentIdx() {
            const n = this.currentNode || 'stage-0';
            if (n.startsWith('stage-')) return parseInt(n.replace('stage-', ''));
            if (n.startsWith('audit-')) {
                const KEY_TO_IDX = { stage_1: 0, stage_2: 1, stage_45: 4, stage_67: 5 };
                return KEY_TO_IDX[n.slice('audit-'.length)] ?? 3;
            }
            return 3; // track nodes live under Implementation
        },

        /** Overview + GATE RECORD data for the current audit node, or null. */
        _auditOverviewData() {
            const node = String(this.currentNode || '');
            if (!node.startsWith('audit-')) return null;
            const key = node.slice('audit-'.length);
            const gate = this._auditGates()[key];
            if (!gate) return null;
            const escalation = gate.state === 'escalated'
                ? (this.campaignState?.campaign?.auditor_escalation || null)
                : null;
            return {
                key,
                label: this._auditGateLabel(key),
                scope: this._auditGateScope(key),
                escalation,
                ...gate,
            };
        },

        /**
         * Filesystem roots visible for the selected L3 node. These are the
         * only semantic mappings in the browser; everything below a root is
         * rendered exactly as returned by artifact-children.
         */
        _l3ArtifactRootSpecs() {
            const round = Number(this.currentRound) || 1;
            const base = `rounds/${round}`;
            const root = (label, path, type = 'directory') => ({ label, path, type });
            switch (this.l3NodeType) {
                case 'baseline':
                    return [
                        root('CONSTRAINTS', `${base}/constraints.md`, 'file'),
                        root('BASELINE SWEEP', `${base}/sweeps/baseline`),
                        root('PROFILING', `${base}/profiling`),
                    ];
                case 'mining':
                    return [root('MINING', `${base}/mining`)];
                case 'debate':
                    return [root('DEBATE', `${base}/debate`)];
                case 'implementation':
                    return [
                        root('TRACK FILES', `${base}/tracks`),
                        root('SWEEP RESULTS', `${base}/sweeps/opt`),
                    ];
                case 'integration':
                    return [
                        root('INTEGRATION FILES', `${base}/integration`),
                        root('INTEGRATION SWEEP', `${base}/sweeps/integration`),
                    ];
                case 'audit':
                    // One flat directory holds every gate's verdict + cycle
                    // files for the round; the browser lists it as-is and the
                    // overview card names which files belong to this gate.
                    return [root('AUDIT VERDICTS', `${base}/audits`)];
                case 'track': {
                    const opId = String(this.currentNode || '');
                    return [
                        root('TRACK FILES', `${base}/tracks/${opId}`),
                        root('SWEEP RESULTS', `${base}/sweeps/opt/${opId}`),
                    ];
                }
                default:
                    return [];
            }
        },

        _newArtifactDirectory(spec, depth = 0, isRoot = false) {
            return {
                name: spec.name || spec.label || spec.path.split('/').pop(),
                label: spec.label || null,
                path: spec.path,
                type: 'directory',
                depth,
                isRoot,
                exists: null,
                loaded: false,
                loading: false,
                expanded: isRoot,
                error: null,
                children: [],
            };
        },

        _newArtifactFileRoot(spec) {
            return {
                name: spec.label || spec.path.split('/').pop(),
                label: spec.label || null,
                path: spec.path,
                type: 'file',
                depth: 0,
                isRoot: true,
                exists: this._catalogHasPath(spec.path),
                size: null,
                mime: null,
                ext: extBadge(spec.path),
            };
        },

        _artifactBrowserEntry(entry, parent) {
            const isDirectory = entry?.type === 'directory' || entry?.type === 'dir';
            if (isDirectory) {
                return this._newArtifactDirectory({ name: entry.name, path: entry.path }, parent.depth + 1);
            }
            return {
                name: entry?.name || String(entry?.path || '').split('/').pop(),
                path: entry?.path,
                type: 'file',
                depth: parent.depth + 1,
                size: entry?.size ?? null,
                mime: entry?.mime || null,
                ext: extBadge(entry?.path || ''),
            };
        },

        _isHiddenArtifactPath(path) {
            const hidden = new Set(['cache', 'triton_cache', 'torch_compile_cache']);
            return String(path || '').split('/').some(segment => hidden.has(segment));
        },

        /** Load one directory level. Descendants are never prefetched. */
        async _loadArtifactDirectory(node) {
            if (!node || node.type !== 'directory' || !this.currentSessionId) return;
            node.loading = true;
            node.error = null;
            this.l3ArtifactRoots = [...this.l3ArtifactRoots];
            try {
                const url = `/api/campaigns/${this.currentSessionId}/artifact-children?path=${encodeURIComponent(node.path)}`;
                const response = await this.apiFetch(url);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const payload = await response.json();
                if (payload.exists === false) {
                    node.exists = false;
                    node.expanded = false;
                    node.children = [];
                    node.loaded = true;
                    return;
                }
                const entries = Array.isArray(payload.entries) ? payload.entries : [];
                node.exists = true;
                node.children = entries
                    .filter(entry => entry?.path && !this._isHiddenArtifactPath(entry.path))
                    .map(entry => this._artifactBrowserEntry(entry, node))
                    .sort((a, b) => {
                        if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
                        return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' });
                    });
                node.loaded = true;
            } catch (error) {
                node.exists = true;
                node.error = `Failed to list ${node.path}: ${error.message}`;
            } finally {
                node.loading = false;
                this.l3ArtifactRoots = [...this.l3ArtifactRoots];
            }
        },

        /** Flat visible-row model; Alpine does not recursively instantiate templates. */
        l3ArtifactRows() {
            const rows = [];
            const visit = node => {
                if (node.exists === false) return;
                rows.push(node);
                if (node.type === 'directory' && node.expanded) {
                    (node.children || []).forEach(visit);
                }
            };
            this.l3ArtifactRoots.forEach(visit);
            return rows;
        },

        async toggleArtifactDirectory(node) {
            if (!node || node.type !== 'directory' || node.loading) return;
            if (node.expanded && node.loaded) {
                node.expanded = false;
                this.l3ArtifactRoots = [...this.l3ArtifactRoots];
                return;
            }
            if (!node.loaded || node.error) await this._loadArtifactDirectory(node);
            if (node.exists !== false && !node.error) node.expanded = true;
            this.l3ArtifactRoots = [...this.l3ArtifactRoots];
        },

        selectArtifactEntry(entry) {
            if (!entry) return;
            if (entry.type === 'directory') {
                this.toggleArtifactDirectory(entry);
                return;
            }
            this.selectArtifactTab({
                name: entry.name,
                path: entry.path,
                size: entry.size,
                mime: entry.mime,
            });
        },


        /**
         * Click a nested artifact tab. Updates URL hash if the target node differs
         * from the current L3 node, then loads the artifact.
         * @param {{path: string, nodeId: string, round: number}} tab
         */
        selectArtifactTab(tab) {
            if (!tab || !tab.path) return;
            // Route to the owning L3 node if it differs from the current one.
            const targetNode = tab.nodeId;
            const targetRound = tab.round || this.currentRound;
            if (targetNode && (targetNode !== this.currentNode || targetRound !== this.currentRound)) {
                // Defer artifact selection to the new L3 init via pendingArtifactPath.
                this.pendingArtifactPath = tab.path;
                this.navigateTo(3, this.currentSessionId, targetRound, targetNode);
                return;
            }
            // Same node — just load the artifact (loadArtifact is tolerant to missing sections).
            const known = this.l3Sections.find(s => s.path === tab.path);
            if (!known) {
                // Append the tab to l3Sections so loadArtifact can track it.
                this.l3Sections = [
                    ...this.l3Sections,
                    { name: tab.name, path: tab.path, content: null, mime: tab.mime || 'text/plain',
                      size: tab.size ?? null,
                      loaded: false, loading: false, error: null, available: true },
                ];
            }
            this.loadArtifact(tab.path);
        },

        /**
         * Map MIME type to highlight.js language string.
         * @param {string} mime
         * @returns {string}
         */
        mimeToLang(mime) {
            switch (mime) {
                case 'text/x-python':      return 'python';
                case 'application/json':   return 'json';
                case 'text/markdown':      return 'markdown';
                case 'text/plain':         return 'plaintext';
                default:                   return 'plaintext';
            }
        },

        /** Escape a string for safe interpolation into HTML text content
         *  or double-quoted attribute values. */
        _escapeHtml(s) {
            return String(s).replace(/[&<>"']/g, c => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;',
            }[c]));
        },

        /**
         * Render a plain-text block wrapped in `.log-view`.
         * All five metachars escaped. Trailing newline preserved so a
         * `.log` that ends with a blank line still has its spacing.
         */
        renderPlaintextBlock(text) {
            const escaped = this._escapeHtml(text == null ? '' : text);
            return `<div class="log-view">${escaped}</div>`;
        },

        /**
         * Render markdown via marked + DOMPurify with hljs fenced-code
         * highlighting. Pure function (returns a string); safe to pass
         * to `x-html` since DOMPurify strips script/event-handler attrs.
         */
        _renderMarkdownBlock(text) {
            const src = text == null ? '' : String(text);
            if (typeof marked === 'undefined') {
                return this.renderPlaintextBlock(src);
            }
            // Cheap local hljs hook so markdown fences get the retheme.
            try {
                marked.setOptions({
                    highlight: (code, lang) => {
                        if (typeof hljs === 'undefined') return this._escapeHtml(code);
                        try {
                            return hljs.highlight(code, {
                                language: lang || 'plaintext',
                                ignoreIllegals: true,
                            }).value;
                        } catch (_) {
                            return hljs.highlightAuto(code).value;
                        }
                    },
                });
            } catch (_) { /* marked < 5 lacks setOptions('highlight'); fall through */ }
            const raw = marked.parse(src);
            return (typeof DOMPurify !== 'undefined')
                ? DOMPurify.sanitize(raw)
                : raw;
        },

        /**
         * Render a JSON tree to a pure HTML string.
         * - Depth 0 (root) and depth 1 expanded; depth 2+ collapsed via
         *   `.jt-children.hidden` (document-level click handler toggles).
         * - Cycle-safe via an object-only `WeakSet` guard. Scalars (0, '',
         *   false, null) do NOT touch the WeakSet — avoids the classic
         *   "Invalid value used in weak set" TypeError.
         * - Keys AND string leaves are HTML-escaped.
         */
        renderJsonTree(obj) {
            const escape = (s) => this._escapeHtml(s);
            const seen = new WeakSet();

            const typeOf = (v) => {
                if (v === null) return 'null';
                if (Array.isArray(v)) return 'array';
                return typeof v;
            };
            const renderLeaf = (value, t) => {
                if (t === 'string')  return `<span class="jt-str">"${escape(value)}"</span>`;
                if (t === 'number')  return `<span class="jt-num">${escape(value)}</span>`;
                if (t === 'boolean') return `<span class="jt-bool">${value ? 'true' : 'false'}</span>`;
                if (t === 'null')    return `<span class="jt-null">null</span>`;
                return `<span>${escape(String(value))}</span>`;
            };
            const node = (value, key, depth) => {
                const t = typeOf(value);
                // Circular-ref guard — only objects/arrays enter the WeakSet.
                if (t === 'object' || t === 'array') {
                    if (seen.has(value)) {
                        const keyHtml = (key !== null && key !== undefined)
                            ? `<span class="jt-key">${escape(String(key))}</span>` +
                              `<span class="jt-punct">:</span> `
                            : '';
                        return (
                            `<div class="jt-node${depth === 0 ? ' jt-root' : ''}">` +
                                `<div class="jt-line">` +
                                    `${keyHtml}<span class="jt-null">⟲ circular ref</span>` +
                                `</div>` +
                            `</div>`
                        );
                    }
                    seen.add(value);

                    const entries = (t === 'array')
                        ? value.map((v, i) => [i, v])
                        : Object.entries(value);
                    const open  = t === 'array' ? '[' : '{';
                    const close = t === 'array' ? ']' : '}';
                    const unit  = t === 'array' ? 'items' : 'keys';
                    const collapsedDefault = depth >= 2;
                    const hiddenCls = collapsedDefault ? ' hidden' : '';
                    const toggleCls = collapsedDefault ? ' collapsed' : '';
                    const keyHtml = (key !== null && key !== undefined)
                        ? `<span class="jt-key">${escape(String(key))}</span>` +
                          `<span class="jt-punct">:</span> `
                        : '';
                    const header =
                        `<div class="jt-line jt-expandable">` +
                            `<span class="jt-toggle${toggleCls}">▾</span>` +
                            `${keyHtml}` +
                            `<span class="jt-punct">${open}</span>` +
                            `<span class="jt-count">${entries.length} ${unit}</span>` +
                        `</div>`;
                    const children =
                        `<div class="jt-children${hiddenCls}">` +
                            entries.map(([k, v]) => node(v, k, depth + 1)).join('') +
                        `</div>`;
                    const closeLine =
                        `<div class="jt-close"` +
                            (collapsedDefault ? ' style="display:none;"' : '') +
                        `><span class="jt-punct">${close}</span></div>`;
                    return (
                        `<div class="jt-node${depth === 0 ? ' jt-root' : ''}">` +
                            header + children + closeLine +
                        `</div>`
                    );
                }

                const keyHtml = (key !== null && key !== undefined)
                    ? `<span class="jt-key">${escape(String(key))}</span>` +
                      `<span class="jt-punct">:</span> `
                    : '';
                return (
                    `<div class="jt-node${depth === 0 ? ' jt-root' : ''}">` +
                        `<div class="jt-line">${keyHtml}${renderLeaf(value, t)}</div>` +
                    `</div>`
                );
            };

            return `<div class="json-tree">${node(obj, null, 0)}</div>`;
        },

        /**
         * Render a line-numbered code view. Language is resolved via the
         * explicit argument; unknown langs fall back to `hljs.highlightAuto`.
         * When hljs is unavailable the raw source is emitted escaped so
         * XSS is impossible.
         */
        renderCodeView(text, lang) {
            const src = text == null ? '' : String(text);
            const lines = src.split('\n');
            let codeHtml;
            let resolvedLang = (lang || '').toLowerCase();
            if (typeof hljs !== 'undefined') {
                try {
                    if (resolvedLang) {
                        codeHtml = hljs.highlight(src, {
                            language: resolvedLang,
                            ignoreIllegals: true,
                        }).value;
                    } else {
                        const auto = hljs.highlightAuto(src);
                        codeHtml = auto.value;
                        resolvedLang = auto.language || '';
                    }
                } catch (_) {
                    codeHtml = this._escapeHtml(src);
                }
            } else {
                codeHtml = this._escapeHtml(src);
            }
            const gutter = lines
                .map((_, i) => `<div>${i + 1}</div>`)
                .join('');
            const langCls = resolvedLang ? ` language-${resolvedLang}` : '';
            return (
                `<div class="code-view">` +
                    `<div class="code-gutter">${gutter}</div>` +
                    `<div class="code-content">` +
                        `<pre><code class="hljs${langCls}">${codeHtml}</code></pre>` +
                    `</div>` +
                `</div>`
            );
        },

        /**
         * Render an inline image block. No remote source validation — the
         * gating is done before this is called (see `_shouldInlineImage`
         * / `loadArtifact` image branch). `path` is passed through the
         * same `/api/campaigns/.../artifacts/...` route; the server
         * ownership gate is authoritative.
         */
        renderImageBlock(path, size) {
            const safePath = this._escapeHtml(path || '');
            const url = this._escapeHtml(this.artifactUrl(path));
            const caption = size
                ? `${safePath} · ${this._escapeHtml(this.humanBytes(size))}`
                : safePath;
            return (
                `<div class="image-view">` +
                    `<div class="image-frame">` +
                        `<img src="${url}" alt="${safePath}" loading="lazy" />` +
                    `</div>` +
                    `<div class="image-caption"><b>${caption}</b></div>` +
                `</div>`
            );
        },

        /**
         * Render a binary download card. Re-used by the oversized-image
         * fallback path (`dispatchRenderer`) and by the x-if branch in
         * `index.html` for native binary artifacts.
         */
        _renderBinaryCard(path, mime, size) {
            const safePath = this._escapeHtml(path || '');
            const safeMime = this._escapeHtml(mime || 'application/octet-stream');
            const safeSize = this._escapeHtml(this.humanBytes(size) || '—');
            const url = this._escapeHtml(this.artifactUrl(path));
            return (
                `<div class="l3-binary-card" style="padding:20px;border:1px solid var(--wire);` +
                    `background:var(--panel);font-family:var(--font-mono);">` +
                    `<div style="font-size:11px;color:var(--ghost);letter-spacing:2px;margin-bottom:8px;">` +
                        `BINARY ARTIFACT` +
                    `</div>` +
                    `<div style="font-size:13px;color:var(--text);margin-bottom:4px;">${safePath}</div>` +
                    `<div style="font-size:11px;color:var(--dim);margin-bottom:12px;">` +
                        `${safeMime} · ${safeSize}` +
                    `</div>` +
                    `<a href="${url}" download ` +
                        `style="display:inline-block;padding:6px 12px;border:1px solid var(--cyan);` +
                        `color:var(--cyan);text-decoration:none;font-size:11px;letter-spacing:1px;">` +
                        `↓ DOWNLOAD` +
                    `</a>` +
                `</div>`
            );
        },

        /**
         * Lazy-load the diff2html CSS + UI bundle from unpkg. Returns a Promise
         * that resolves once `window.Diff2HtmlUI` (or the fallback
         * `window.Diff2Html`) is available. The promise is memoized on
         * `window._lgDiff2HtmlPromise` so concurrent renderers share one fetch
         * and the bundle is loaded at most once per page lifetime.
         *
         * Failure mode: if the CDN is unreachable, the returned promise
         * rejects; callers fall back to a plain-text rendering of the patch.
         */
        _ensureDiff2Html() {
            if (window._lgDiff2HtmlPromise) return window._lgDiff2HtmlPromise;
            const CSS_URL = 'https://unpkg.com/diff2html/bundles/css/diff2html.min.css';
            const JS_URL  = 'https://unpkg.com/diff2html/bundles/js/diff2html-ui.min.js';

            // Inject the upstream CSS once (LIGHTGRID overrides come from
            // frontend/css/lightgrid-diff.css and are loaded from index.html).
            if (!document.querySelector(`link[data-lg-diff2html]`)) {
                const link = document.createElement('link');
                link.rel = 'stylesheet';
                link.href = CSS_URL;
                link.dataset.lgDiff2html = '1';
                document.head.appendChild(link);
            }

            window._lgDiff2HtmlPromise = new Promise((resolve, reject) => {
                if (window.Diff2HtmlUI || window.Diff2Html) { resolve(window.Diff2HtmlUI || window.Diff2Html); return; }
                const existing = document.querySelector(`script[data-lg-diff2html]`);
                if (existing) {
                    existing.addEventListener('load', () => resolve(window.Diff2HtmlUI || window.Diff2Html));
                    existing.addEventListener('error', () => reject(new Error('diff2html CDN load failed')));
                    return;
                }
                const script = document.createElement('script');
                script.src = JS_URL;
                script.async = true;
                script.dataset.lgDiff2html = '1';
                script.onload  = () => resolve(window.Diff2HtmlUI || window.Diff2Html);
                script.onerror = () => reject(new Error('diff2html CDN load failed'));
                document.head.appendChild(script);
            });
            return window._lgDiff2HtmlPromise;
        },

        /**
         * Render a unified diff/patch using diff2html in side-by-side mode.
         *
         * Synchronous return contract: `dispatchRenderer` is sync and stamps
         * `section.mode` atomically. We therefore return a placeholder
         * `<div data-lg-diff-host>` immediately and hydrate it once the CDN
         * bundle resolves. The placeholder carries the raw patch as a base64
         * payload on a data attribute so a re-render of the same x-html block
         * is idempotent (the hydration handler reads it back rather than
         * relying on a closure that Alpine may have torn down).
         *
         * Empty diffs render as a "No changes" notice — keeps tracks with
         * zero churn discoverable rather than silently empty.
         */
        renderDiff2Html(content) {
            const src = (content == null ? '' : String(content));
            // Empty / whitespace-only diff → friendly notice, no CDN trip.
            if (!src.trim()) {
                return (
                    `<div class="l3-diff-empty" style="padding:24px;border:1px dashed var(--wire);` +
                        `background:var(--panel);color:var(--ghost);font-family:var(--font-mono);` +
                        `font-size:12px;letter-spacing:1px;text-align:center;">` +
                        `// NO CHANGES IN THIS DIFF` +
                    `</div>`
                );
            }

            // Stable host id; embed the patch on a data attribute as base64
            // (UTF-8 safe via encodeURIComponent → unescape → btoa). Hydration
            // reads from the DOM, not a closure, so re-mounts stay idempotent.
            window._lgDiffSeq = (window._lgDiffSeq || 0) + 1;
            const hostId = `lg-diff-${window._lgDiffSeq}`;
            let payload = '';
            try {
                payload = btoa(unescape(encodeURIComponent(src)));
            } catch (_) {
                // Fallback: skip payload, hydrate from a global cache.
                window._lgDiffCache = window._lgDiffCache || {};
                window._lgDiffCache[hostId] = src;
            }

            // Schedule hydration on next microtask — by then Alpine has
            // committed the x-html assignment and the host element is live.
            queueMicrotask(() => this._hydrateDiff2Html(hostId));

            const safePayload = this._escapeHtml(payload);
            return (
                `<div class="lg-diff-shell">` +
                    `<div id="${hostId}" class="lg-diff-host" ` +
                        `data-lg-diff-host="1" data-lg-diff-payload="${safePayload}">` +
                        `<div class="lg-diff-loading" style="padding:24px;color:var(--ghost);` +
                            `font-family:var(--font-mono);font-size:12px;letter-spacing:1px;">` +
                            `// LOADING DIFF VIEWER…` +
                        `</div>` +
                    `</div>` +
                `</div>`
            );
        },

        /**
         * Hydrate a `renderDiff2Html` placeholder once the CDN bundle is
         * loaded. Pulls the patch text from the host element's data
         * attribute (or `_lgDiffCache` fallback), invokes diff2html in
         * side-by-side mode with file list + line matching, and decorates
         * each file header with a copy-path button. No-op when the host
         * has gone (Alpine torn it down between dispatch and hydration).
         */
        _hydrateDiff2Html(hostId) {
            const host = document.getElementById(hostId);
            if (!host) return;
            // Decode the patch text (data attribute first, in-memory cache fallback).
            let patch = '';
            const enc = host.dataset.lgDiffPayload || '';
            if (enc) {
                try { patch = decodeURIComponent(escape(atob(enc))); }
                catch (_) { patch = ''; }
            }
            if (!patch && window._lgDiffCache && window._lgDiffCache[hostId]) {
                patch = window._lgDiffCache[hostId];
            }
            if (!patch) {
                host.innerHTML =
                    `<div class="lg-diff-error" style="padding:24px;color:var(--err);` +
                        `font-family:var(--font-mono);font-size:12px;">// EMPTY OR UNREADABLE PATCH</div>`;
                return;
            }

            this._ensureDiff2Html().then(() => {
                // Re-locate after async — Alpine may have re-rendered.
                const liveHost = document.getElementById(hostId);
                if (!liveHost) return;
                try {
                    const opts = {
                        drawFileList: true,
                        outputFormat: 'side-by-side',
                        matching: 'lines',
                        renderNothingWhenEmpty: false,
                    };
                    if (window.Diff2HtmlUI) {
                        // UI bundle exposes a class that injects HTML +
                        // wires up file-list collapse handlers.
                        const ui = new window.Diff2HtmlUI(liveHost, patch, opts);
                        ui.draw();
                        if (typeof ui.fileListToggle === 'function') ui.fileListToggle(false);
                    } else if (window.Diff2Html && typeof window.Diff2Html.html === 'function') {
                        liveHost.innerHTML = window.Diff2Html.html(patch, opts);
                    } else {
                        throw new Error('diff2html global not found after load');
                    }
                    this._decorateDiff2HtmlCopyButtons(liveHost);
                } catch (err) {
                    liveHost.innerHTML =
                        `<div class="lg-diff-error" style="padding:24px;color:var(--err);` +
                            `font-family:var(--font-mono);font-size:12px;">` +
                            `// DIFF RENDER FAILED: ${this._escapeHtml(err.message || String(err))}` +
                        `</div>`;
                }
            }).catch((err) => {
                const liveHost = document.getElementById(hostId);
                if (!liveHost) return;
                // Graceful fallback: render the raw patch as a plain code block.
                const escaped = this._escapeHtml(patch);
                liveHost.innerHTML =
                    `<div class="lg-diff-fallback" style="margin-bottom:8px;color:var(--err);` +
                        `font-family:var(--font-mono);font-size:11px;letter-spacing:1px;">` +
                        `// DIFF VIEWER OFFLINE — SHOWING RAW PATCH (${this._escapeHtml(err.message || 'unknown error')})` +
                    `</div>` +
                    `<pre class="lg-code-block"><code>${escaped}</code></pre>`;
            });
        },

        /**
         * Decorate each diff2html file header with a "copy path" button and
         * each code hunk with a "copy" button. Buttons fade in on hover via
         * the LIGHTGRID override CSS; click handlers use the Clipboard API
         * with a textarea fallback for non-secure contexts.
         */
        _decorateDiff2HtmlCopyButtons(host) {
            if (!host) return;
            const copy = (text, btn) => {
                const done = () => {
                    if (!btn) return;
                    const orig = btn.textContent;
                    btn.textContent = '✓ COPIED';
                    btn.classList.add('lg-diff-copy-ok');
                    setTimeout(() => {
                        btn.textContent = orig;
                        btn.classList.remove('lg-diff-copy-ok');
                    }, 1200);
                };
                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(text).then(done).catch(() => done());
                    return;
                }
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                try { document.execCommand('copy'); } catch (_) { /* ignore */ }
                document.body.removeChild(ta);
                done();
            };

            // File-header copy buttons — copy the file path.
            host.querySelectorAll('.d2h-file-header').forEach((header) => {
                if (header.querySelector('.lg-diff-copy-path')) return;
                const nameEl = header.querySelector('.d2h-file-name');
                const path = (nameEl ? nameEl.textContent : '').trim();
                if (!path) return;
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'lg-diff-copy-path';
                btn.textContent = '⧉ COPY PATH';
                btn.title = 'Copy file path';
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    copy(path, btn);
                });
                header.appendChild(btn);
            });

            // Per-file hunk copy — copies the file's full diff text.
            host.querySelectorAll('.d2h-file-wrapper').forEach((wrap) => {
                if (wrap.querySelector('.lg-diff-copy-hunk')) return;
                const header = wrap.querySelector('.d2h-file-header');
                if (!header) return;
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'lg-diff-copy-hunk';
                btn.textContent = '⧉ COPY DIFF';
                btn.title = 'Copy this file’s diff';
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    // Reconstruct from rendered cells — falls back to text content of the side-by-side panes.
                    const text = Array.from(wrap.querySelectorAll('.d2h-code-side-line, .d2h-code-line'))
                        .map(el => el.textContent)
                        .join('\n');
                    copy(text, btn);
                });
                header.appendChild(btn);
            });

            // File-list anchor rebind — diff2html generates `<a href="#d2h-XXXX">`
            // links in the summary list. The bare hash change (e.g. `#d2h-12345`)
            // collides with our hash-based router (`#campaigns/...`), nuking the
            // current view. Intercept and scrollIntoView() instead, leaving the
            // route hash untouched. Marks the anchor so re-renders are idempotent.
            host.querySelectorAll('.d2h-file-list-line a[href^="#"]').forEach((a) => {
                if (a.dataset.lgDiffBound === '1') return;
                a.dataset.lgDiffBound = '1';
                a.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const id = (a.getAttribute('href') || '').slice(1);
                    if (!id) return;
                    // Use querySelector with attribute (id has dashes, safe selector).
                    const target = host.querySelector(`#${CSS.escape(id)}`)
                        || document.getElementById(id);
                    if (target && typeof target.scrollIntoView === 'function') {
                        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        // Brief cyan-flash to confirm the jump.
                        const header = target.querySelector('.d2h-file-header') || target;
                        header.style.transition = 'box-shadow 0.6s ease';
                        header.style.boxShadow = '0 0 0 1px var(--cyan), 0 0 16px rgba(0,243,255,0.45)';
                        setTimeout(() => { header.style.boxShadow = ''; }, 700);
                    }
                });
            });
        },

        /**
         * Extension-first renderer dispatch. Returns { mode, html, rendererLabel }.
         * Path extension wins when the backend ships a misleading mime (e.g.
         * `application/octet-stream` for a `.md`).
         *
         * Order:
         *   1. Markdown (.md)
         *   2. JSON (.json)
         *   3. Image (.png/.jpg/.jpeg/.gif/.webp/.bmp/.ico/.svg)
         *      ↳ 5 MB gate → `binary` (download card) when oversized.
         *   4. Diff / Patch (.diff/.patch) — diff2html side-by-side
         *   5. Code (known code extensions → hljs language)
         *   6. Plaintext / log (fallback)
         *
         * Never awaits anything — callers can stamp `section.mode` atomically.
         * The diff2html branch returns a synchronous placeholder; the actual
         * rendering hydrates asynchronously once the CDN bundle has loaded.
         */
        dispatchRenderer(path, mime, content, size) {
            const ext = this._extOf(path);
            const m = (mime || '').toLowerCase();

            // 1. Markdown
            if (ext === '.md' || m === 'text/markdown') {
                return {
                    mode: 'md',
                    html: this._renderMarkdownBlock(content),
                    rendererLabel: 'markdown',
                };
            }

            // 2. JSON
            if (ext === '.json' || m === 'application/json' || m.endsWith('+json')) {
                let parsed;
                try { parsed = JSON.parse(content || 'null'); }
                catch (_) { parsed = content; }
                return {
                    mode: 'json',
                    html: this.renderJsonTree(parsed),
                    rendererLabel: 'json-tree',
                };
            }

            // 3. Image — gate by size. Oversize → binary download card.
            if (this._isImageExt(ext) || m.startsWith('image/')) {
                if (this._shouldInlineImage(path, mime, size)) {
                    return {
                        mode: 'image',
                        html: this.renderImageBlock(path, size),
                        rendererLabel: 'image (inline)',
                    };
                }
                return {
                    mode: 'binary',
                    html: this._renderBinaryCard(path, mime, size),
                    rendererLabel: 'download',
                };
            }

            // 4. Diff / Patch — side-by-side viewer (LIGHTGRID-themed) via diff2html
            //    Routed BEFORE the generic code highlighter so unified-diff payloads
            //    don't fall back to plain syntax highlighting.
            if (ext === '.diff' || ext === '.patch' || m === 'text/x-diff' || m === 'text/x-patch') {
                return {
                    mode: 'diff',
                    html: this.renderDiff2Html(content),
                    rendererLabel: 'diff (side-by-side)',
                };
            }

            // 5. Code
            const codeLang = this.extToLang(ext);
            const CODE_EXT = /\.(py|cu|cuh|cpp|cc|cxx|h|hpp|c|js|jsx|ts|tsx|sh|bash|yaml|yml|toml|ini|rs|go|cmake)$/i;
            if (codeLang || CODE_EXT.test(ext) || m === 'text/x-python' || m === 'text/javascript') {
                return {
                    mode: 'code',
                    html: this.renderCodeView(content, codeLang || this.mimeToLang(mime)),
                    rendererLabel: codeLang ? `code (${codeLang})` : 'code',
                };
            }

            // 6. Plaintext / log fallback
            return {
                mode: 'log',
                html: this.renderPlaintextBlock(content),
                rendererLabel: 'plaintext',
            };
        },

        /**
         * Render artifact content to safe HTML. Accepts the 4-tuple
         * `(content, mime, path, size)`; `path` + `size` enable the
         * extension-first / image-gate dispatch without changing callers.
         * The legacy 2-arg form is preserved (tests/unit A3 regression).
         *
         * @returns {string} safe HTML
         */
        renderArtifactContent(content, mime, path, size) {
            if (content == null && !path) return '';
            // Legacy 2-arg shape (path unavailable) — derive the renderer
            // from the mime only so downstream callers that never learned
            // the 4-arg signature still work. Preserved for test
            // regression coverage (TestRenderArtifactContent / TestEdgeCasesL3).
            if (path === undefined) {
                if (mime === 'text/markdown') return this._renderMarkdownBlock(content);
                const lang = this.mimeToLang(mime);
                if (lang !== 'plaintext' && typeof hljs !== 'undefined') {
                    return this.renderCodeView(content, lang);
                }
                // hljs missing or plaintext mime → simple `<pre>` wrapper
                // with HTML-escaped content. Matches the pre-port behavior
                // (TestEdgeCasesL3::test_hljs_unavailable_falls_back_to_plain).
                return `<pre class="lg-code-block"><code>${this._escapeHtml(content)}</code></pre>`;
            }
            return this.dispatchRenderer(path, mime, content, size).html;
        },

        /**
         * Load a specific artifact file and cache it in l3Sections.
         *
         * Atomicity contract (A9): the final "tail" block of this method
         * must not contain any `await`. All size/mime/content decisions
         * happen there in one synchronous sweep so consumers of
         * `sec.mode` / `sec.content` always see a consistent pair.
         *
         * - Images: HEAD probe first → `_shouldInlineImage(...)` gate →
         *   either `mode:'image'` (no body download) or `mode:'binary'`.
         *   HEAD failure is a conservative "binary download card" fallback.
         * - Binary (`_isBinaryArtifact`): GET for headers only; content
         *   stays `null` so `.text()` isn't called on raw bytes.
         * - Text: GET body, `mode:'text'` with string `content`.
         *
         * @param {string} path - relative artifact path
         */
        async loadArtifact(path) {
            if (!this.currentSessionId || !path) return;
            // Lazy browsing and deep links can select a file that was never
            // enumerated by /tree. Register it on demand so content loading is
            // independent of catalog timing and directory expansion state.
            let section = this.l3Sections.find(s => s.path === path);
            if (!section) {
                section = {
                    name: String(path).split('/').pop(),
                    path,
                    content: null,
                    mime: 'text/plain',
                    size: null,
                    loaded: false,
                    loading: false,
                    error: null,
                    available: true,
                };
                this.l3Sections = [...this.l3Sections, section];
            }
            this.l3ActiveSection = path;
            if (!this.l3OpenedTabs.includes(path)) this.l3OpenedTabs.push(path);
            if (section.loaded) return;
            section.loading = true;
            section.error = null;
            try {
                const ext = this._extOf(path);
                const url = this.artifactUrl(path);

                if (this._isImageExt(ext)) {
                    // HEAD probe so we don't fetch the whole image just to
                    // discover it's 12 MB.
                    let head;
                    try {
                        head = await this.apiFetch(url, { method: 'HEAD' });
                    } catch (_) {
                        head = null;
                    }
                    const size = head && head.ok
                        ? (parseInt(head.headers.get('content-length') || '0', 10) || null)
                        : null;
                    const mime = head && head.ok
                        ? (head.headers.get('content-type') || '').split(';')[0].trim()
                        : '';
                    // ATOMIC TAIL — no awaits below this line.
                    const mode = this._shouldInlineImage(path, mime, size) ? 'image' : 'binary';
                    section.size    = size;
                    section.mime    = mime || `image/${ext.slice(1)}`;
                    section.content = null;
                    section.binary  = (mode === 'binary');  // legacy flag, kept for existing templates
                    section.mode    = mode;
                    section._rendererLabel = (mode === 'image') ? 'image (inline)' : 'download';
                    section.loaded  = true;
                    section.available = true;
                    return;
                }

                const resp = await this.apiFetch(url);
                if (!resp.ok) {
                    section.available = false;
                    throw new Error(`HTTP ${resp.status}`);
                }
                const mime = (resp.headers.get('content-type') || 'text/plain').split(';')[0].trim();
                const size = parseInt(resp.headers.get('content-length') || '0', 10) || null;
                const isBin = this._isBinaryArtifact(path, mime);
                const content = isBin ? null : await resp.text();

                // ATOMIC TAIL — no awaits below this line.
                const mode = isBin ? 'binary' : 'text';
                section.size    = size;
                section.mime    = mime;
                section.content = content;
                section.binary  = isBin;
                section.mode    = mode;
                section._rendererLabel = isBin
                    ? 'download'
                    : this.dispatchRenderer(path, mime, content, size).rendererLabel;
                section.loaded  = true;
                section.available = true;
            } catch (e) {
                section.error  = `Failed to load ${path}: ${e.message}`;
                section.loaded = false;
            } finally {
                section.loading = false;
            }
        },

        /** Get the currently active section object. */
        get activeSection() {
            return this.l3Sections.find(s => s.path === this.l3ActiveSection) || null;
        },

        /** Initialize the node-scoped natural artifact browser. */
        async initL3Sections() {
            const token = ++this.l3ArtifactBrowserToken;
            const roots = this._l3ArtifactRootSpecs()
                .map(spec => spec.type === 'file'
                    ? this._newArtifactFileRoot(spec)
                    : this._newArtifactDirectory(spec, 0, true));
            this.l3ArtifactRoots = roots;
            this.l3ArtifactBrowserReady = false;
            this.l3OpenedTabs = [];
            this.l3Sections = [];
            this.l3ActiveSection = null;
            this.l3CatalogEmpty = !this.artifactCatalog;
            this.l3CatalogData = this._buildL3CatalogData();

            // Load only each root's immediate children. Missing roots disappear;
            // nested directories remain untouched until clicked.
            // Load through Alpine's reactive proxies, not the pre-assignment raw
            // objects. Otherwise keyed x-for rows can retain the initial
            // `loading: true` scope even after the raw root is set back to false.
            await Promise.all(this.l3ArtifactRoots
                .filter(root => root.type === 'directory')
                .map(root => this._loadArtifactDirectory(root)));
            if (token !== this.l3ArtifactBrowserToken) return;
            this.l3ArtifactBrowserReady = true;

            // A deep-linked file loads directly even when no ancestor directory
            // has been expanded (or the flat /tree catalog is stale).
            if (this.pendingArtifactPath) {
                const target = this.pendingArtifactPath;
                this.pendingArtifactPath = null;
                await this.loadArtifact(target);
            }
        },

        /** Check if a path exists in the in-memory artifact tree response.
         *  Sidecar removal (2026-05-27): the tree endpoint returns a flat
         *  list of relative paths; matching is a single Set lookup against
         *  `tree.files`. Empty/null tree → optimistic true so tabs render
         *  while polling is still in flight. */
        _catalogHasPath(path) {
            const tree = this.artifactCatalog;
            if (!tree) return true; // tree not loaded yet — optimistic
            const files = Array.isArray(tree.files) ? tree.files
                : (Array.isArray(tree) ? tree : null);
            if (!files || !files.length) return true; // empty tree → optimistic
            // Cache the file Set once per tree poll so repeated calls during
            // a single render don't rebuild it.
            if (this._catalogPathSetTree !== tree) {
                this._catalogPathSet = new Set(files);
                this._catalogPathSetTree = tree;
            }
            return this._catalogPathSet.has(path);
        },

        /** True when artifact_catalog hasn't been generated yet */
        l3CatalogEmpty: false,

        /**
         * Build catalog-enriched metrics for the current L3 node.
         * Returns { track: {...}|null, stage: {...}|null }.
         */
        _buildL3CatalogData() {
            const catalog = this.artifactCatalog;
            const node = this.currentNode;
            const roundId = this.currentRound;
            // Sidecar removal: catalog is the tree endpoint response.
            // hasCatalog is true when the tree carries any files.
            const _hasFiles = (t) => {
                if (!t) return false;
                if (Array.isArray(t.files)) return t.files.length > 0;
                if (Array.isArray(t)) return t.length > 0;
                if (typeof t === 'object') return Object.keys(t).length > 0;
                return false;
            };
            const hasCatalog = _hasFiles(catalog);

            // Route both `stage-N` and bare stage-name aliases to the stage
            // branch so the overview hero renders the correct stage metrics
            // instead of falling through to the TRACK branch (Bug 3).
            const BARE_STAGE_IDX = {
                baseline: 0, mining: 1, debate: 2,
                implementation: 3, validation: 4, integration: 5,
            };
            // Audit nodes carry no catalog metrics — their overview hero and
            // GATE RECORD read auditGateStates directly. Bail before the
            // track fallthrough misreads `audit-stage_45` as an op_id.
            if (String(node || '').startsWith('audit-')) return null;
            const colMatch = String(node || '').match(/^stage-(\d+)$/);
            let colIdx = colMatch ? parseInt(colMatch[1], 10) : -1;
            if (colIdx < 0 && node && node in BARE_STAGE_IDX) {
                colIdx = BARE_STAGE_IDX[node];
            }
            const isTrack = !!node && colIdx < 0;

            // Stage-5 (integrate) uses state.json, not catalog
            const stageData = colIdx === 5
                ? this._integrationStageData(roundId)
                : (hasCatalog && colIdx >= 0 ? this._stageCatalogMetrics(colIdx, roundId, catalog) : null);

            const trackSource = isTrack ? this._trackSourceInfo(node) : null;
            // Scope catalog metrics to the requested round — without this, a track
            // that only ran in R1 leaks its gate_5_2/gate_5_1a numbers onto R2's
            // PROFILING + CORRECTNESS panels (Task #51).
            const trackCat = isTrack && hasCatalog ? this._trackCatalogMetrics(node, catalog, roundId) : null;
            // Enrich source info with catalog description if available
            if (trackSource && trackCat?.description) {
                trackSource.description = trackCat.description;
            }

            // Return non-null when we have any data (track or stage)
            if (!isTrack && !stageData) return null;

            return {
                track: trackCat,
                stage: stageData,
                trackSource,
                isTrack,
            };
        },

        /**
         * Look up catalog metrics for a specific track (op_id) within a given round.
         * Extracts profiling (gate_5_2), correctness (gate_5_1a), description
         * (validation_results.md).
         *
         * `roundId` scoping (Task #51): a track's gate artifacts are emitted once
         * per campaign round. Without this filter, a track that ran in R1 leaked
         * its numbers onto the R2 L3 page even though R2 hadn't dispatched the op
         * yet. We match `entry.round === roundId`, and fall back to "unstamped"
         * entries only when the caller is on round 1 (the earliest round).
         */
        _trackCatalogMetrics(opId, catalog, roundId) {
            // Sidecar removal (2026-05-27): per-track metrics come straight
            // from `state.json`'s round[N].parallel_tracks.tracks[op_id]
            // record (Track 1 added gate_5_1a_metrics / gate_5_2_metrics
            // alongside the existing pass/fail strings). The `catalog`
            // (tree response) is used only to confirm the gate file exists
            // on disk for the L3 "Open →" link.
            const state = this.campaignState;
            const round = state?.campaign?.rounds?.[roundId != null ? roundId - 1 : 0];
            // Case-insensitive lookup so `op-002` URL hits the `OP-002`
            // record. lookupOp() handles upper/lower/title casing.
            const tracks = round?.parallel_tracks?.tracks || {};
            const stateTrack = lookupOp(tracks, opId) || null;

            // Gate 5.2 → profiling. Prefer the orchestrator-enriched
            // `gate_5_2_metrics` block; fall back to scalar speedups.
            const g52 = stateTrack?.gate_5_2_metrics || null;
            let profiling = null;
            if (g52 && (g52.weighted_speedup_cold != null || g52.weighted_speedup_warm != null)) {
                profiling = {
                    speedupCold: g52.weighted_speedup_cold ?? null,
                    speedupWarm: g52.weighted_speedup_warm ?? null,
                    shapesTested: g52.shapes_tested ?? null,
                };
            } else if (stateTrack && (stateTrack.kernel_speedup_cold != null
                                   || stateTrack.kernel_speedup_warm != null)) {
                profiling = {
                    speedupCold: stateTrack.kernel_speedup_cold ?? null,
                    speedupWarm: stateTrack.kernel_speedup_warm ?? null,
                    shapesTested: null,
                };
            }

            // Gate 5.1a → correctness. Same precedence: enriched metrics
            // block > scalar `gate_5_1a` PASS/FAIL > derived from `correctness`.
            const g51a = stateTrack?.gate_5_1a_metrics || null;
            let correctness = null;
            if (g51a && (g51a.overall || g51a.max_abs_err != null)) {
                correctness = {
                    overall: g51a.overall || null,
                    shapesTested: g51a.shapes_tested ?? null,
                    maxAbsErr: g51a.max_abs_err ?? null,
                };
            } else if (stateTrack && (stateTrack.gate_5_1a || stateTrack.correctness != null)) {
                correctness = {
                    overall: stateTrack.gate_5_1a
                        ?? (stateTrack.correctness ? 'PASS'
                          : stateTrack.correctness === false ? 'FAIL' : null),
                    shapesTested: null,
                    maxAbsErr: null,
                };
                if (correctness.overall == null) correctness = null;
            }

            const verdict = stateTrack?.verdict || null;
            const classification = stateTrack?.classification || null;
            // No description source post-sidecar-removal — the validation_results.md
            // body is read on demand via the L3 viewer, not pre-aggregated.
            const description = null;

            if (!profiling && !correctness) return null;
            return { profiling, correctness, description, verdict, classification };
        },

        /**
         * Count shipped/failed/active across ALL campaign rounds + current parallel_tracks.
         * Verdict mapping:
         *   shipped  = integrated into trunk (in shipped_optimizations)
         *   failed   = verdict == FAIL/FAILED
         *   active   = still running (no terminal verdict yet)
         * PASS / GATED_PASS (validated, awaiting merge) are folded into SHIPPED so the
         * footer reflects "completed successfully" from the user's perspective — until
         * a dedicated VALIDATED bucket is added to the footer HTML.
         */
        _countAllTrackStatuses() {
            const state = this.campaignState;
            if (!state) return { shipped: 0, failed: 0, active: 0 };
            const shippedSet = window.LG_HELPERS._normalizeShippedOps(state.campaign || {});
            const rounds = state.campaign?.rounds || [];
            const countedOps = new Set();
            // Fold the 7-way trackStatus ladder into the 3-way footer bucket:
            // gated + validated + shipped → shipped (the footer has no VALIDATED slot yet);
            // failed + blocked → failed (GPU_BLOCKED is a terminal-fail outcome from the UI's POV);
            // gating + in_progress + pending + unknown → active.
            const bucketOf = (status) => {
                if (status === 'shipped' || status === 'gated' || status === 'validated') return 'shipped';
                if (status === 'failed' || status === 'blocked') return 'failed';
                return 'active';
            };
            let shipped = 0, failed = 0, active = 0;
            const currentRoundId = state.campaign?.current_round || 1;
            for (const rnd of rounds) {
                for (const opId of (rnd.shipped || [])) {
                    if (!countedOps.has(opId)) { shipped++; countedOps.add(opId); }
                }
                const pt = (rnd.parallel_tracks && rnd.parallel_tracks.tracks) || {};
                const isCurrent = rnd.round_id === currentRoundId;
                for (const [opId, track] of Object.entries(pt)) {
                    if (countedOps.has(opId)) continue;
                    const b = bucketOf(trackStatus({ ...track, op_id: opId }, shippedSet));
                    if (b === 'shipped') { shipped++; countedOps.add(opId); }
                    else if (b === 'failed') { failed++; countedOps.add(opId); }
                    else if (isCurrent) { active++; countedOps.add(opId); }
                    // past-round non-terminals are not counted as active
                }
            }
            return { shipped, failed, active };
        },

        /**
         * Build track source/git info from state.json for the given opId.
         * Returns {commitSha, branch, classification, status, speedup} or null.
         */
        _trackSourceInfo(opId) {
            const state = this.campaignState;
            if (!state) return null;
            const tracks = currentTracks(state);
            // Case-insensitive lookup (URL uses op-002, state uses OP-002)
            const t = lookupOp(tracks, opId);
            if (!t) {
                // Fallback: look in historical round data
                const campaign = state.campaign || {};
                for (const rnd of (campaign.rounds || [])) {
                    const pt = (rnd.parallel_tracks && rnd.parallel_tracks.tracks) || {};
                    const result = lookupOp(pt, opId);
                    if (result) {
                        const shippedOps = window.LG_HELPERS._normalizeShippedOps(campaign);
                        const isShipped = shippedOps.has(opId) || shippedOps.has(opId.toUpperCase()) || shippedOps.has(opId.toLowerCase());
                        const rStatus = String(result.status || '').toUpperCase();
                        const isFailed = rStatus === 'FAIL' || rStatus === 'FAILED' || rStatus === 'GPU_BLOCKED';
                        return {
                            commitSha: result.commit_sha || null,
                            branch: result.worktree_branch || null,
                            classification: result.classification || null,
                            status: isShipped ? 'shipped' : isFailed ? 'failed' : 'active',
                            cumulativeSpeedup: campaign.cumulative_e2e_speedup || null,
                            description: null,
                        };
                    }
                }
                return null;
            }
            const shippedOps = window.LG_HELPERS._normalizeShippedOps(state.campaign || {});
            const tStatus = (t.status || '').toUpperCase();
            const isFailed = tStatus === 'FAILED' || tStatus === 'FAIL' || tStatus === 'GPU_BLOCKED';
            const status = (shippedOps.has(opId) || shippedOps.has(opId.toUpperCase())) ? 'shipped'
                : isFailed ? 'failed' : 'active';
            return {
                commitSha: t.commit_sha || null,
                branch: t.worktree_branch || null,
                classification: t.classification || null,
                status,
                cumulativeSpeedup: state.campaign?.cumulative_e2e_speedup || null,
                description: null, // filled from catalog if available
            };
        },

        /**
         * Task #46 — L3 TRACK overview panel data.
         * Returns null for non-track nodes or when state is missing.
         *
         * Picks the correct source for (opId, roundId):
         *   - current round  → campaign.rounds[cr-1].parallel_tracks.tracks[op_id]
         *   - past round     → campaign.rounds[N-1].parallel_tracks.tracks[op_id]
         *
         * Kernel speedup uses CircuitBoard.kernelSpeedupScalar for preference-order
         * parity with the L2 chip: cold+bs8 > bs8 NOT warm > cold > max.
         *
         * Classification → verdict color mapping (matches L2 chip semantics):
         *   LOSSLESS    → mint
         *   GATED_PASS  → amber
         *   LOSSY (pass)→ amber
         *   FAILED      → red
         *   (unknown)   → ghost
         *
         * Returns { opId, e2e, kernel, classification, verdict, status,
         *          classColor, e2eNote } or null.
         */
        // C1. Memoization cache for _trackOverviewData. Called 13-17× per
        // L3 render; keyed on (catalog.last_updated, currentNode, currentRound,
        // state._etag or parallel_tracks value-signature). See buildTrackOverviewKey.
        _trackOverviewCache: { key: null, value: null },
        _trackOverviewData() {
            if (this.l3NodeType !== 'track') return null;
            const state = this.campaignState;
            const opId = this.currentNode;
            const roundId = this.currentRound;
            if (!state || !opId) return null;

            const cacheKey = buildTrackOverviewKey(
                this.artifactCatalog, opId, roundId, state,
            );
            if (this._trackOverviewCache.key === cacheKey) {
                return this._trackOverviewCache.value;
            }
            const result = this._trackOverviewDataImpl(state, opId, roundId);
            this._trackOverviewCache = { key: cacheKey, value: result };
            return result;
        },
        _trackOverviewDataImpl(state, opId, roundId) {

            const campaign = state.campaign || {};
            const currentRound = campaign.current_round || 1;
            const isCurrent = roundId === currentRound;

            // Resolve per-round track record — strictly scoped to roundId so R2
            // track pages don't inherit R1 archive data when R2 tracks haven't
            // been dispatched yet.
            const findTrack = (ptMap) => lookupOp(ptMap, opId) || null;
            let t = null;
            if (isCurrent) {
                t = findTrack(currentTracks(state));
            }
            if (!t) {
                // Past round (or current-round fallback when current pointer
                // hasn't been initialized yet): look up the round's own
                // parallel_tracks.tracks map.
                const rounds = campaign.rounds || [];
                const rnd = findRound(rounds, roundId);
                if (rnd) {
                    const pt = (rnd.parallel_tracks && rnd.parallel_tracks.tracks) || {};
                    t = findTrack(pt);
                }
            }
            // No data for this (roundId, opId) → return a "not started" shell
            // so the template can render a ghost placeholder instead of
            // accidentally showing another round's numbers.
            if (!t) {
                return {
                    opId, roundId,
                    e2e: null, e2eNote: null,
                    kernel: null,
                    classification: 'NOT STARTED',
                    verdict: null,
                    status: null,
                    isFailed: false,
                    classColor: 'ghost',
                    notStarted: true,
                };
            }

            // Shipped detection (matches L2 chip behavior, both str + dict shapes).
            // Casing preserved verbatim; the lookup below tries all three cases.
            const shippedSet = window.LG_HELPERS._normalizeShippedOps(campaign);

            // E2E speedup — scalar first, fall back to obj { speedup_x, measured }
            const e2eSpeedup = t.e2e_speedup;
            let e2eScalar = null;
            let e2eNote = null;
            if (typeof e2eSpeedup === 'number' && Number.isFinite(e2eSpeedup) && e2eSpeedup > 0) {
                e2eScalar = e2eSpeedup;
            } else if (e2eSpeedup && typeof e2eSpeedup === 'object') {
                if (typeof e2eSpeedup.speedup_x === 'number' && e2eSpeedup.speedup_x > 0) {
                    e2eScalar = e2eSpeedup.speedup_x;
                } else if (typeof e2eSpeedup.measured === 'number' && e2eSpeedup.measured > 0) {
                    e2eScalar = e2eSpeedup.measured;
                }
                if (e2eScalar == null) {
                    const amCold = e2eSpeedup.amdahl_prediction_cold_pp;
                    const amWarm = e2eSpeedup.amdahl_prediction_warm_pp;
                    if (typeof amCold === 'number' && Number.isFinite(amCold)) {
                        e2eNote = `amdahl: +${amCold.toFixed(2)}pp`;
                    } else if (typeof amWarm === 'number' && Number.isFinite(amWarm)) {
                        e2eNote = `amdahl: +${amWarm.toFixed(2)}pp`;
                    }
                }
            }
            // Also check `e2e_result` (current model field — per spec).
            const eR = t.e2e_result;
            if (e2eScalar == null && eR && typeof eR === 'object' &&
                typeof eR.speedup === 'number' && eR.speedup > 0) {
                e2eScalar = eR.speedup;
            }

            // Kernel speedup — reuse circuit-board helper for preference-order parity.
            // Falls back to kernel_speedup_cold / kernel_speedup_warm siblings when
            // the legacy kernel_speedup field is null (orchestrator currently writes
            // only the suffixed pair after the sidecar refactor).
            let kernelScalar = null;
            if (typeof CircuitBoard !== 'undefined' && typeof CircuitBoard.kernelSpeedupFromTrack === 'function') {
                kernelScalar = CircuitBoard.kernelSpeedupFromTrack(t);
            } else {
                const ks = t.kernel_speedup;
                if (typeof CircuitBoard !== 'undefined' && typeof CircuitBoard.kernelSpeedupScalar === 'function') {
                    kernelScalar = CircuitBoard.kernelSpeedupScalar(ks);
                } else if (typeof ks === 'number' && Number.isFinite(ks) && ks > 0) {
                    kernelScalar = ks;
                }
            }

            // Verdict + classification
            const rawClass = (t.classification || '').toString();
            const verdict = (t.verdict || t.status || '').toString().toUpperCase();
            const isFailed = verdict === 'FAIL' || verdict === 'FAILED' || verdict === 'GPU_BLOCKED';
            const isShipped = shippedSet.has(opId) || shippedSet.has(opId.toUpperCase()) || shippedSet.has(opId.toLowerCase());

            // Display classification — the TRACK card should show LOSSLESS /
            // GATED_PASS / FAILED primarily, but respect raw fields from state.json.
            let displayClass;
            let classColor;
            if (verdict === 'GPU_BLOCKED') {
                displayClass = 'GPU_BLOCKED';
                classColor = 'amber';
            } else if (isFailed) {
                displayClass = 'FAILED';
                classColor = 'red';
            } else if (verdict === 'GATED_PASS' || verdict === 'GATED-PASS') {
                displayClass = 'VALIDATED';
                classColor = 'mint';
            } else if (verdict === 'GATING_REQUIRED') {
                displayClass = 'GATING';
                classColor = 'amber';
            } else if (rawClass.toLowerCase() === 'lossless') {
                displayClass = 'LOSSLESS';
                classColor = 'mint';
            } else if (rawClass.toLowerCase() === 'lossy') {
                // Lossy but not gated/failed = passed with precision trade-off.
                displayClass = 'LOSSY';
                classColor = 'amber';
            } else if (isShipped || verdict === 'PASS' || verdict === 'PASSED') {
                displayClass = 'LOSSLESS';
                classColor = 'mint';
            } else {
                displayClass = rawClass ? rawClass.toUpperCase() : '—';
                classColor = 'ghost';
            }

            return {
                opId,
                roundId,
                e2e: e2eScalar,
                e2eNote,
                kernel: kernelScalar,
                classification: displayClass,
                verdict: verdict || null,
                status: t.status || null,
                isFailed,
                classColor,
            };
        },

        /**
         * Get track IDs belonging to a specific campaign round.
         * Sources, in preference order:
         *   1. rounds[N-1].debate.selected_winners
         *   2. rounds[N-1].parallel_tracks.tracks keys
         *   3. (for current round) current debate's selected_winners
         */
        _roundTrackIds(roundId) {
            const state = this.campaignState;
            if (!state) return [];
            const campaign = state.campaign || {};
            const rounds = campaign.rounds || [];
            const roundData = findRound(rounds, roundId);

            if (roundData) {
                const winners = roundData.debate?.selected_winners;
                if (winners?.length) return winners;
                const ptKeys = Object.keys((roundData.parallel_tracks && roundData.parallel_tracks.tracks) || {});
                if (ptKeys.length) return ptKeys;
            }

            // Fallback for current/active round whose debate winners have been
            // scaffolded but parallel_tracks.tracks is still empty.
            if (roundId === campaign.current_round) {
                const trackKeys = Object.keys(currentTracks(state));
                if (trackKeys.length) return trackKeys;
                return currentDebate(state).selected_winners || [];
            }

            return [];
        },

        /**
         * Look up catalog metrics for a stage-level view.
         * Mining → bottleneck, Debate → winners, Baseline → latency.
         */
        _stageCatalogMetrics(colIdx, roundId, _tree) {
            // Sidecar removal (2026-05-27): all stage-level numbers now
            // live on the per-round state.json record — orchestrator
            // populates them at gate-pass / stage-completion (see
            // skills/ammo/SKILL.md § FE Metric Enrichment).
            const state = this.campaignState;
            const rounds = state?.campaign?.rounds || [];
            const round = rounds.find(r => (r.round_id ?? r.round) === roundId)
                ?? rounds[(roundId || 1) - 1]
                ?? null;
            if (!round) return null;
            switch (colIdx) {
                case 0: { // Baseline
                    const lat = round.baseline?.e2e_latency;
                    if (!lat || typeof lat !== 'object' || !Object.keys(lat).length) return null;
                    // Flatten the workload-bucket map into display-friendly k/v.
                    const flat = {};
                    for (const bucket of bucketRecords(lat)) {
                        const vals = lat[bucket.tag];
                        if (!vals || typeof vals !== 'object') continue;
                        for (const [k, v] of Object.entries(vals)) {
                            flat[`${k} (${bucket.label})`] = v;
                        }
                    }
                    if (!Object.keys(flat).length) return null;
                    return { type: 'baseline', latency: flat };
                }
                case 1: { // Mining
                    const m = round.bottleneck_mining || {};
                    if (m.top_component == null && m.top_f_decode_pct == null
                        && m.amdahl_ceiling == null
                        && m.top_bottleneck_share_pct == null) return null;
                    return {
                        type: 'mining',
                        topComponent: m.top_component ?? null,
                        // Field-name compatibility: the FE display uses
                        // `topPct`; orchestrator writes `top_f_decode_pct`.
                        // `top_bottleneck_share_pct` is the legacy field
                        // name still present on the round object.
                        topPct: m.top_f_decode_pct ?? m.top_bottleneck_share_pct ?? null,
                        amdahlCeiling: m.amdahl_ceiling ?? null,
                    };
                }
                case 2: { // Debate
                    const d = round.debate || {};
                    const winners = Array.isArray(d.winners) && d.winners.length
                        ? d.winners
                        : (Array.isArray(d.selected_winners) ? d.selected_winners : []);
                    if (!winners.length && d.champions_count == null && d.result == null) {
                        return null;
                    }
                    return {
                        type: 'debate',
                        winners,
                        championsCount: d.champions_count ?? null,
                        result: d.result ?? null,
                    };
                }
                default:
                    return null;
            }
        },

        /**
         * Build integration stage data from state.json (no catalog needed).
         *
         * Round-scoped (Bug fix): the previous version called
         * `currentIntegration(state)` which hard-codes the campaign's
         * `current_round`. When a user navigated to L3 integration for an
         * older SHIPPED round (R1, R2), this returned the in-progress
         * round's `pending` status and the page rendered "Awaiting data..."
         * forever. We now look up the integration record on the requested
         * `roundId` so each round renders its own integration outcome.
         *
         * Cumulative speedup also walks a `??` fallback chain — the canonical
         * field is `roundRec.cumulative_speedup_after`, with a
         * `campaign.cumulative_speedup_vs_round1` fallback. The legacy
         * `campaign.cumulative_e2e_speedup` field is null in production.
         */
        _integrationStageData(roundId) {
            const state = this.campaignState;
            if (!state) return null;
            const rounds = state.campaign?.rounds || [];
            const roundRec = rounds.find(r => r.round_id === roundId) || rounds[roundId - 1] || {};
            const integ = roundRec.integration || {};
            const campaign = state.campaign || {};
            if (!integ.status || integ.status === 'pending') return null;
            return {
                type: 'integration',
                status: integ.status,
                passingCandidates: integ.passing_candidates || [],
                finalDecision: integ.final_decision,
                cumulativeSpeedup: roundRec.cumulative_speedup_after
                    ?? campaign.cumulative_speedup_vs_round1
                    ?? campaign.cumulative_e2e_speedup,
            };
        },

        /**
         * Extract baseline hero latency (first deterministic workload bucket) for overview display.
         * Schema v4.0: reads roundRec.baseline.e2e_latency (map) first, then falls
         * back through integration.e2e_latency_combined, legacy combined_e2e_result
         * scalars, and finally the e2e_latency_results sidecar.
         *
         * Returns { value: '23.4', unit: 'ms', label: 'IL64 \u00b7 OL512 \u00b7 BS128 MEDIAN LATENCY' } or null.
         */
        _baselineHeroLatency() {
            const state = this.campaignState;
            const roundId = this.currentRound;
            const roundRec = ((state?.campaign?.rounds) || []).find(
                r => (r.round_id ?? r.round) === roundId
            );

            // The live app always uses the shared parser. Keep a tiny
            // self-contained fallback for consumers that extract this method
            // in isolation (for example archived dashboard harnesses).
            const _records = (map) => {
                if (typeof bucketRecords === 'function') return bucketRecords(map);
                if (!map || typeof map !== 'object') return [];
                return Object.keys(map).map(tag => {
                    let match = /^il(\d+)_ol(\d+)_bs(\d+)$/.exec(tag);
                    if (match) {
                        const inputLen = Number(match[1]);
                        const outputLen = Number(match[2]);
                        const batchSize = Number(match[3]);
                        if (![inputLen, outputLen, batchSize].every(value => Number.isSafeInteger(value) && value > 0)) return null;
                        return {
                            tag, inputLen, outputLen, batchSize,
                            heterogeneous: true, legacyNumeric: false,
                            label: `IL${match[1]} \u00b7 OL${match[2]} \u00b7 BS${match[3]}`,
                            compactLabel: `IL${match[1]}/OL${match[2]}/BS${match[3]}`,
                        };
                    }
                    match = /^(?:bs)?(\d+)$/.exec(tag);
                    if (!match) return null;
                    const batchSize = Number(match[1]);
                    return Number.isSafeInteger(batchSize) && batchSize > 0 ? {
                        tag, inputLen: null, outputLen: null, batchSize,
                        heterogeneous: false, legacyNumeric: /^\d+$/.test(tag),
                        label: `BS${match[1]}`, compactLabel: `BS${match[1]}`,
                    } : null;
                }).filter(Boolean).sort((a, b) =>
                    (a.inputLen ?? -1) - (b.inputLen ?? -1)
                    || (a.outputLen ?? -1) - (b.outputLen ?? -1)
                    || a.batchSize - b.batchSize
                    || (a.tag < b.tag ? -1 : (a.tag > b.tag ? 1 : 0))
                );
            };

            // Build result from a v4.0 map while preserving its exact bucket tag.
            const _fromMap = (map) => {
                if (!map || typeof map !== 'object') return null;
                const bucket = _records(map)[0];
                if (!bucket) return null;
                const entry = map[bucket.tag];
                if (!entry || typeof entry.avg !== 'number' || entry.avg <= 0) return null;
                return {
                    value: (entry.avg * 1000).toFixed(1),
                    unit: 'ms',
                    label: `${bucket.label} MEDIAN LATENCY`,
                };
            };

            // 1. Primary — baseline.e2e_latency map (normalizer-populated for v3 too).
            const primary = _fromMap(roundRec?.baseline?.e2e_latency);
            if (primary) return primary;

            // 2. Carry-forward — integration.e2e_latency_combined on the same round.
            const carry = _fromMap(roundRec?.integration?.e2e_latency_combined);
            if (carry) return carry;

            // 3. Legacy scalar — combined_e2e_result.latency_baseline_s.
            const cer = roundRec?.integration?.combined_e2e_result;
            const latS = cer?.latency_baseline_s;
            if (typeof latS === 'number' && Number.isFinite(latS) && latS > 0) {
                const perBs = cer?.per_bs_verdict || {};
                const bucket = _records(perBs)[0];
                return {
                    value: (latS * 1000).toFixed(1),
                    unit: 'ms',
                    label: bucket ? `${bucket.label} MEDIAN LATENCY` : 'BASELINE MEDIAN LATENCY',
                };
            }

            // Sidecar removal (2026-05-27): the catalog is now a tree manifest
            // with no embedded metrics, so the previous e2e_latency sidecar
            // fallback can never produce a result. State-driven sources above
            // (1-3) are the authoritative path; legacy campaigns are migrated
            // by the backend normalizer that backfills baseline.e2e_latency.
            return null;
        },

        /**
         * Build implementation stage track table data from state.json parallel_tracks.
         * Returns array of { opId, status, kernelSpeedup, e2eSpeedup, classification, verdict, failReason }.
         */
        _implementationTrackTable() {
            const state = this.campaignState;
            if (!state) return [];
            const roundTrackIds = new Set(this._roundTrackIds(this.currentRound));
            const campaign = state.campaign || {};
            const shippedOps = window.LG_HELPERS._normalizeShippedOps(campaign);

            // Pull tracks from the target round's parallel_tracks.tracks map.
            const roundData = findRound(campaign.rounds || [], this.currentRound);
            const tracks = (roundData?.parallel_tracks && roundData.parallel_tracks.tracks) || {};

            return Object.entries(tracks)
                .filter(([opId]) => !roundTrackIds.size || roundTrackIds.has(opId))
                .map(([opId, track]) => {
                    const isShipped = shippedOps.has(opId);
                    const _tStatus = String(track.status || '').toUpperCase();
                    const isFailed = _tStatus === 'FAIL' || _tStatus === 'FAILED' || _tStatus === 'GPU_BLOCKED';
                    return {
                        name: opId,
                        status: isShipped ? 'shipped' : isFailed ? 'failed' : 'active',
                        speedup: track.e2e_speedup,
                        classification: track.classification || null,
                        failReason: _failReason(track, null),
                    };
                });
        },

        // ── L2 Circuit Board ───────────────────────────────────────────────
        /**
         * Render the circuit board into #cb-mount once state is loaded.
         * Called from the x-init on the L2 template after Alpine renders it.
         */
        _cbLastDataKey: null,
        mountCircuitBoard() {
            if (!this.campaignState) return;
            const el = document.getElementById('cb-mount');
            if (!el || typeof window.renderCircuitBoard !== 'function') return;

            // Only re-render if campaign data structurally changed.
            // Without this guard, every 5s poll re-triggers x-effect → full SVG rebuild
            // → all animations restart from scratch, making them look broken.
            const state = this.campaignState;
            const campaign = state.campaign || {};
            const cat = this.artifactCatalog || {};
            const liveTracks = currentTracks(state);
            const dataKey = JSON.stringify({
                stage: currentStage(state),
                currentRound: campaign.current_round,
                roundCount: (campaign.rounds || []).length,
                shippedCount: window.LG_HELPERS._normalizeShippedOps(campaign).size,
                trackKeys: Object.keys(liveTracks).sort(),
                trackStatuses: Object.values(liveTracks).map(t => t.status),
                trackVerdicts: Object.values(liveTracks).map(t => t.verdict),
                integStatus: currentIntegration(state).status,
                speedup: campaign.cumulative_e2e_speedup,
                // Include catalog signature so sidecar-driven UI (rationale cards,
                // resolutions, debate chip counts) re-renders when entries land even
                // if parallel_tracks hasn't moved.
                catalogEntryCount: cat?.files?.length ?? 0,
                // Audit gate signature: started_at/passed_at per gate per round,
                // so the board re-renders live when an audit starts or passes
                // (independent of track/integration status changing).
                // stage_6/stage_7 are the pre-consolidation aliases the state
                // engine still accepts for stage_67.
                auditSig: (campaign.rounds || []).map(r => {
                    const audit = r.audit || {};
                    return ['stage_1', 'stage_2', 'stage_45', 'stage_67', 'stage_6', 'stage_7']
                        .map(k => {
                            const g = audit[k] || {};
                            return `${g.started_at || ''}|${g.passed_at || ''}`;
                        })
                        .join(',');
                }).join(';'),
                // An escalation write moves neither a gate stamp nor a track
                // status, so without these two the most urgent hex state has no
                // repaint trigger at all.
                campaignStatus: campaign.status ?? null,
                escalationSig: JSON.stringify(campaign.auditor_escalation ?? null),
            });
            if (this._cbLastDataKey === dataKey) return;
            this._cbLastDataKey = dataKey;

            // Reachability function exposed to the circuit-board render so it
             // can grey out future/not-yet-run chips AND skip their click wiring.
            const reachable = (roundId, colIdx) => this._stageReachable(roundId, colIdx);
            window.renderCircuitBoard(el, state, (roundId, nodeId, isStage) => {
                // Gate stage clicks: if the user clicks a chip whose stage hasn't
                // run for this round, no-op. Track clicks (isStage=false) are
                // allowed through — tracks only exist when they ran.
                if (isStage && !this._stageReachable(roundId, nodeId)) {
                    console.warn(`[lg-nav] stage-${nodeId} click ignored for round ${roundId} — stage not yet reached`);
                    return;
                }
                this.navigateTo(3, this.currentSessionId, roundId, isStage ? `stage-${nodeId}` : nodeId);
            }, this.artifactCatalog, (path) => {
                // Sidecar deep-link: route to L3 node that owns this artifact, auto-open the tab
                this.openSidecarArtifact(path);
            }, reachable, (roundId, gateKey) => {
                // Audit-gate fuse hex → dedicated L3 audit node. Same modal as
                // every other drill-in; `audit-{gateKey}` routes l3NodeType to
                // the audit overview + gate record + scoped artifact browser.
                this.navigateTo(3, this.currentSessionId, roundId, `audit-${gateKey}`);
            });
        },

        /**
         * Route an artifact path to the correct L3 node + tab.
         *
         * Sidecar removal (2026-05-27): label metadata is no longer server-
         * stamped, so all routing decisions are derived from the path itself
         * via `parseArtifactPath`. The function name retains "Sidecar" for
         * call-site compatibility — it's now a misnomer for "deep-link to
         * artifact path".
         */
        openSidecarArtifact(path) {
            const parsed = window.LG_HELPERS?.parseArtifactPath?.(path) || {};
            const kind = parsed.kind;
            const opId = parsed.op_id || parsed.track_id;
            const round = parsed.round || this.currentRound || 1;
            this.pendingArtifactPath = path;
            if (kind === 'source_code' || kind === 'debate_rationale') {
                // Land on the track node; tab auto-selects based on pendingArtifactPath.
                if (opId) {
                    this.navigateTo(3, this.currentSessionId, round, opId);
                    return;
                }
                // Rationale paths without an op_id (e.g. debate summaries) have
                // no L3 destination — open inline.
                this.pendingArtifactPath = null;
                const champ = parsed.champion_id ? parsed.champion_id.toUpperCase() : '';
                const stance = parsed.stance ? parsed.stance.toUpperCase() : '';
                const title = [champ, stance].filter(Boolean).join(' · ') ||
                    (path.split('/').pop() || 'Rationale');
                this.openSidecarMarkdown(path, title);
                return;
            }
            if (kind === 'diff') {
                // Track-scoped diffs land on the track node; orphan diffs fall back to stage-5.
                const target = opId || 'stage-5';
                this.navigateTo(3, this.currentSessionId, round, target);
                return;
            }
            if (kind === 'report_section') {
                this.navigateTo(3, this.currentSessionId, round, 'stage-6');
                return;
            }
            // Fallback — clear pending and just open the file inline.
            this.pendingArtifactPath = null;
            this.openSidecarMarkdown(path, path.split('/').pop() || 'Artifact');
        },

        /**
         * Open a markdown sidecar artifact in a lightweight viewer.
         * Used when the sidecar has no op_id to deep-link into L3.
         */
        async openSidecarMarkdown(path, title) {
            if (!this.currentSessionId) return;
            this.sidecarOverlay.open = true;
            this.sidecarOverlay.leaving = false;
            this.sidecarOverlay.loading = true;
            this.sidecarOverlay.path = path;
            this.sidecarOverlay.title = title;
            this.sidecarOverlay.renderedHtml = '';
            this.sidecarOverlay.errorMsg = '';
            try {
                const res = await this.fetchArtifact(this.currentSessionId, path);
                if (!res || res.content == null) {
                    this.sidecarOverlay.errorMsg = 'Artifact could not be loaded.';
                } else {
                    const isMd = /\.md$/i.test(path);
                    if (isMd) {
                        this.sidecarOverlay.renderedHtml = this.renderMarkdown(res.content);
                    } else {
                        const escaped = String(res.content)
                            .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                        this.sidecarOverlay.renderedHtml =
                            `<pre style="white-space:pre-wrap;word-break:break-word;">${escaped}</pre>`;
                    }
                }
            } catch (e) {
                this.sidecarOverlay.errorMsg = (e && e.message) || 'Failed to load artifact.';
            } finally {
                this.sidecarOverlay.loading = false;
            }
        },

        closeSidecar() {
            this.sidecarOverlay.leaving = true;
            setTimeout(() => {
                this.sidecarOverlay.open = false;
                this.sidecarOverlay.leaving = false;
                this.sidecarOverlay.renderedHtml = '';
            }, 180);
        },
    }));
});
