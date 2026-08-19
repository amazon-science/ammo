// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0
/**
 * circuit-board.js — LIGHTGRID L2 HUD Circuit Board Component
 *
 * Data-driven hybrid circuit board: HTML chips + SVG trace overlay.
 * Called from campaignApp._renderCircuitBoard(state) when L2 is active.
 *
 * Layout: 6 pipeline stage columns × N round rows. Each intersection is a "node".
 * Right sidebar: session info, per-round speedup bar chart, round cards.
 *
 * Graceful degradation:
 *   - Current round: full detail from root-level parallel_tracks + debate + integration
 *   - Past rounds:   summary from campaign.rounds[n] (selected_candidates, shipped, etc.)
 */

// ── Constants ──────────────────────────────────────────────────────────────────
const CB = {
    STAGE_COLS: ['Baseline', 'Mining', 'Debate', 'Implement', 'Validate', 'Integrate'],
    COL_W: 160,       // column width (stage chip width)
    COL_GAP: 40,      // gap between columns
    ROW_H: 90,        // height per round row
    HEADER_H: 50,     // top header height
    TOP_PAD: 30,      // padding above first row
    CHIP_H: 30,       // stage chip header height
    NODE_W: 130,      // node foreignObject width
    NODE_H: 64,       // node foreignObject height
    SVG_NS: 'http://www.w3.org/2000/svg',
    XHTML_NS: 'http://www.w3.org/1999/xhtml',

    // Stage ordering map — terminal pseudo-stages eliminated; use
    // campaign.status (campaign_complete/campaign_exhausted) instead.
    STAGE_ORDER: {
        '1_baseline': 0,
        '2_bottleneck_mining': 1,
        '3_debate': 2,
        '4_5_parallel_tracks': 3,
        '6_integration': 5,
        '7_campaign_eval': 6,
        '7b_report': 7,
    },

    // v1→v2 stage-name → round sub-object key mapping.
    STAGE_KEY_MAP: {
        '1_baseline': 'baseline',
        '2_bottleneck_mining': 'bottleneck_mining',
        '3_debate': 'debate',
        '4_5_parallel_tracks': 'parallel_tracks',
        '6_integration': 'integration',
        '7_campaign_eval': 'campaign_eval',
    },

    // Status -> color token
    STATUS_COLOR: {
        shipped:   'var(--mint)',
        gated:     'var(--mint)',
        validated: 'var(--mint)',
        active:    'var(--cyan)',
        failed:    'var(--red)',
        blocked:   'var(--amber)',
        gating:    'var(--amber)',
        pending:   'var(--ghost)',
        in_progress: 'var(--cyan)',
        SHIPPED:     'var(--mint)',
        PASSED:      'var(--mint)',
        PASS:        'var(--mint)',
        GATED_PASS:  'var(--mint)',
        FAILED:      'var(--red)',
        FAIL:        'var(--red)',
        GPU_BLOCKED: 'var(--amber)',
        GATING_REQUIRED: 'var(--amber)',
        IN_PROGRESS: 'var(--cyan)',
    },
};

// ── v2 round accessors ─────────────────────────────────────────────────────────
// All per-round/per-stage state lives under campaign.rounds[current_round-1].
// Past-round access uses `round.{parallel_tracks.tracks, integration, debate}`
// directly (same shape as current).

function _currentRoundObj(state) {
    const campaign = (state && state.campaign) || {};
    const rounds = campaign.rounds || [];
    const idx = (campaign.current_round || 1) - 1;
    return (idx >= 0 && idx < rounds.length) ? (rounds[idx] || {}) : {};
}

function currentTracks(state) {
    const pt = _currentRoundObj(state).parallel_tracks || {};
    return pt.tracks || {};
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

// ── Display names ──────────────────────────────────────────────────────────────
// op_id → short display name minted at debate winner selection
// (campaign.rounds[N].debate.selected_candidates[].name). The field is optional
// (pre-4.2 records omit it), so every consumer falls back to the raw op_id.
// The op_id stays the node id / click key / shipped-set key everywhere; only
// the visible label changes.
function opDisplayNames(state) {
    const out = new Map();
    for (const r of (((state && state.campaign) || {}).rounds || [])) {
        for (const c of ((r && r.debate && r.debate.selected_candidates) || [])) {
            if (c && c.op_id && typeof c.name === 'string' && c.name.trim()) {
                out.set(c.op_id, c.name.trim());
            }
        }
    }
    return out;
}

// ── Audit-gate ("fuse hex") state derivation ─────────────────────────────────
// Maps campaign.rounds[i].audit.{stage_1,stage_2,stage_45,stage_67} (each
// {started_at, passed_at, cycle, verdict_file, note}) to a render state per
// gate. Pure function — no DOM, easy to unit test.
const AUDIT_STAGE_KEYS = ['stage_1', 'stage_2', 'stage_45', 'stage_67'];

// auditor_escalation.stage (a campaign pipeline stage id) -> the audit gate
// key it corresponds to. 6_integration and 7_campaign_eval both escalate the
// consolidated stage_67 gate.
const AUDIT_ESCALATION_STAGE_MAP = {
    '1_baseline': 'stage_1',
    '2_bottleneck_mining': 'stage_2',
    '4_5_parallel_tracks': 'stage_45',
    '6_integration': 'stage_67',
    '7_campaign_eval': 'stage_67',
};

// Tracks are "audited-stage complete" for stage_45 once every track has left
// IN_PROGRESS (any of PASS/GATING_REQUIRED/GATED_PASS/FAIL/GPU_BLOCKED).
function _auditTracksTerminal(tracks) {
    const ids = Object.keys(tracks || {});
    if (ids.length === 0) return false;
    return ids.every(id => ((tracks[id] || {}).status || '').toUpperCase() !== 'IN_PROGRESS');
}

// integration.status is terminal for stage_67 once it has left pending/in_progress.
function _auditIntegrationTerminal(status) {
    const s = String(status || '').toLowerCase();
    return s !== '' && s !== 'pending' && s !== 'in_progress';
}

// Pre-consolidation aliases for stage_67. ammo_state.py's gates accept either
// key as the S67 PASS, so a fully-audited legacy round must not render as
// "not audited". Later passed_at wins when both keys carry one.
const AUDIT_LEGACY_S67_KEYS = ['stage_7', 'stage_6'];

function _resolveAuditGate(auditObj, key) {
    const direct = auditObj[key] || null;
    if (key !== 'stage_67') return direct;
    const candidates = [direct];
    for (const legacy of AUDIT_LEGACY_S67_KEYS) candidates.push(auditObj[legacy] || null);
    const passed = candidates.filter(g => g && g.passed_at);
    if (passed.length) {
        return passed.reduce((a, b) => (String(b.passed_at) > String(a.passed_at) ? b : a));
    }
    return candidates.find(g => g && (g.started_at || g.cycle || g.verdict_file)) || direct;
}

// How much campaign data a round entry carries. Duplicate `round_id` entries
// occur in real state (parseRounds has deduped them since the L2 multi-round
// fix), and every consumer must pick the SAME entry or the board renders one
// round's row beside another entry's audit gates.
function _roundRichness(round) {
    const debate = round.debate || {};
    return [
        (round.selected_candidates || []).length,
        (debate.selected_winners || []).length,
        (debate.selected_candidates || []).length,
        Object.keys(round.audit || {}).length,
        (round.shipped || []).length,
        Object.keys(round).length,
    ];
}

// Dedup by round_id, keeping the richest entry. Shared by parseRounds and
// auditGateStates so the two can never disagree about which entry is the round.
function dedupRoundsById(rounds) {
    const seen = new Map();
    for (const r of rounds || []) {
        const rid = r.round_id ?? r.round;
        if (rid == null) continue;
        const existing = seen.get(rid);
        if (!existing) {
            seen.set(rid, { ...r, round_id: rid });
            continue;
        }
        const a = _roundRichness(r), b = _roundRichness(existing);
        for (let i = 0; i < a.length; i++) {
            if (a[i] === b[i]) continue;
            if (a[i] > b[i]) seen.set(rid, { ...r, round_id: rid });
            break;
        }
    }
    return [...seen.values()].sort((a, b) => a.round_id - b.round_id);
}

function auditGateStates(state) {
    const campaign = (state && state.campaign) || {};
    const rounds = dedupRoundsById(campaign.rounds);
    const currentRoundId = campaign.current_round || 1;
    const escalation = campaign.auditor_escalation || null;
    const escalatedStageKey = escalation ? AUDIT_ESCALATION_STAGE_MAP[escalation.stage] : null;
    // Mirror of ammo_state.py's post-SHIP exemption: on round N>1 whose
    // predecessor carries an audit key, the state engine drops the same-round
    // stage_1 requirement, so no auditor is ever dispatched and no stamp lands.
    const prevCarriesAudit = new Map();
    rounds.forEach((r, i) => {
        prevCarriesAudit.set(r.round_id, i > 0 && !!rounds[i - 1].audit);
    });

    const out = {};
    for (const round of rounds) {
        const rid = round.round_id;
        const isCurrentRound = rid === currentRoundId;
        const stageComplete = {
            stage_1: !!((round.baseline || {}).completed_at),
            stage_2: !!((round.bottleneck_mining || {}).completed_at),
            stage_45: _auditTracksTerminal((round.parallel_tracks || {}).tracks),
            stage_67: _auditIntegrationTerminal((round.integration || {}).status),
        };
        const auditObj = round.audit || null;
        const stage1Exempt = prevCarriesAudit.get(rid) === true;

        const gates = {};
        for (const key of AUDIT_STAGE_KEYS) {
            const gate = auditObj ? _resolveAuditGate(auditObj, key) : null;
            const isEscalated = !!(escalation && escalatedStageKey === key && escalation.round === rid);
            let gState;
            if (!auditObj) {
                // Whole round has no audit key at all — zero change from
                // today's plain via.
                gState = 'bypass';
            } else if (gate && gate.passed_at) {
                // passed_at outranks the escalation flag. A track-scoped
                // escalation quarantines tracks and the surviving scope is
                // re-audited, and nothing ever clears auditor_escalation, so an
                // escalation-first ladder would keep a passed gate red for the
                // rest of the round.
                gState = 'passed';
            } else if (isEscalated) {
                gState = 'escalated';
            } else if (gate && gate.started_at) {
                gState = 'running';
            } else if (stageComplete[key] && isCurrentRound && !(key === 'stage_1' && stage1Exempt)) {
                // Heuristic fallback: hook hasn't stamped started_at yet, but
                // the audited stage is done on the live round — the auditor
                // is presumably already running. Skipped for a gate the state
                // engine exempted, which no auditor will ever be spawned for.
                gState = 'running';
            } else {
                gState = 'pending';
            }
            gates[key] = {
                state: gState,
                started_at: (gate && gate.started_at) ?? null,
                passed_at: (gate && gate.passed_at) ?? null,
                cycle: (gate && gate.cycle) ?? null,
                verdict_file: (gate && gate.verdict_file) ?? null,
                note: (gate && gate.note) ?? null,
            };
        }
        out[rid] = gates;
    }
    return out;
}

// ── Coordinate Helpers ─────────────────────────────────────────────────────────

function rowY(row) {
    return CB.HEADER_H + CB.TOP_PAD + row * CB.ROW_H + CB.ROW_H / 2;
}

function colX(col) {
    return 60 + col * (CB.COL_W + CB.COL_GAP) + CB.COL_W / 2;
}

function rightAngleTrace(ax, ay, bx, by) {
    const mid = ay + (by - ay) / 2;
    return `M ${ax} ${ay} L ${ax} ${mid} L ${bx} ${mid} L ${bx} ${by}`;
}

// ── Speedup coercion helpers ──────────────────────────────────────────────────
// parallel_tracks[*].kernel_speedup is a dict keyed by benchmark variants (e.g.
// {triton_vs_two_linear_bs8: 2.306, warm_bs8: 1.0, target: 1.414, floor: 1.283}).
// Pick a single scalar for the hero badge. Preference order:
//   1. "*cold*bs8*" (shipped production config: cold-cache, BS=8 decode).
//   2. any "*bs8*" key that is NOT warm-cache (warm_bs8 is an upper-bound reference,
//      not the ship gate; prefer the cold-cache measurement even if the key doesn't
//      literally contain "cold" — e.g. `triton_vs_two_linear_bs8` on GDN-FUSE).
//   3. any "*cold*" key (fallback when no bs8-tagged variant exists).
//   4. max of remaining non-meta numeric values.
// Meta keys (targets, floors, gates) are always excluded from the hero number.
const _KERN_META_KEYS = new Set(['target', 'floor', 'ship_gate', 'threshold', 'ceiling']);

function _kernelSpeedupScalar(ks) {
    if (ks == null) return null;
    if (typeof ks === 'number' && Number.isFinite(ks) && ks > 0) return ks;
    if (typeof ks !== 'object') return null;
    const numeric = Object.entries(ks)
        .filter(([k, v]) => !_KERN_META_KEYS.has(k) && typeof v === 'number' && Number.isFinite(v) && v > 0);
    if (!numeric.length) return null;
    const coldBs8 = numeric.find(([k]) => /cold.*bs8|bs8.*cold/i.test(k));
    if (coldBs8) return coldBs8[1];
    const bs8NotWarm = numeric.find(([k]) => /bs8/i.test(k) && !/warm/i.test(k));
    if (bs8NotWarm) return bs8NotWarm[1];
    const cold = numeric.find(([k]) => /cold/i.test(k));
    if (cold) return cold[1];
    return numeric.reduce((m, [, v]) => Math.max(m, v), 0);
}

// Resolve the hero kernel-speedup scalar for a parallel_tracks[op_id] record.
// State.schema.json defines three sibling fields:
//   - kernel_speedup        (legacy scalar OR dict of variants)
//   - kernel_speedup_cold   (orchestrator-written scalar, cold-cache)
//   - kernel_speedup_warm   (orchestrator-written scalar, warm-cache)
// _kernelSpeedupScalar() handles the legacy scalar/dict shapes; this wrapper
// adds the cold/warm fallback so post-sidecar state.json (where the
// orchestrator only writes the suffixed pair) still renders a hero number.
function _kernelSpeedupFromTrack(track) {
    if (!track || typeof track !== 'object') return null;
    const primary = _kernelSpeedupScalar(track.kernel_speedup);
    if (primary != null) return primary;
    const cold = track.kernel_speedup_cold;
    if (typeof cold === 'number' && Number.isFinite(cold) && cold > 0) return cold;
    const warm = track.kernel_speedup_warm;
    if (typeof warm === 'number' && Number.isFinite(warm) && warm > 0) return warm;
    return null;
}

// parallel_tracks[*].e2e_speedup is either null, a number, or a dict with
// `speedup_x` / `measured`. Pick a single scalar; null if unmeasured.
// NOTE: amdahl_prediction_*_pp is NOT a measured e2e speedup — it's a ceiling
// derived from the kernel number. Never return it as an e2e scalar. Callers that
// want the prediction should use `_amdahlPredictionPp` separately and render it
// with a distinct label ("amdahl: +X.XXpp") so the user doesn't confuse model
// with measurement.
function _e2eSpeedupScalar(es) {
    if (es == null) return null;
    if (typeof es === 'number' && Number.isFinite(es) && es > 0) return es;
    if (typeof es !== 'object') return null;
    const candidates = [es.speedup_x, es.measured, es.value];
    for (const c of candidates) {
        if (typeof c === 'number' && Number.isFinite(c) && c > 0) return c;
    }
    return null;
}

// Extract Amdahl-predicted e2e delta (in percentage points). Returns null when
// the track has an actual measurement or no amdahl key at all. Prefers cold over
// warm since cold-cache is the ship-gate relevant number.
function _amdahlPredictionPp(es) {
    if (!es || typeof es !== 'object') return null;
    if (typeof es.speedup_x === 'number' && es.speedup_x > 0) return null;
    if (typeof es.measured === 'number' && es.measured > 0) return null;
    const cold = es.amdahl_prediction_cold_pp;
    if (typeof cold === 'number' && Number.isFinite(cold)) return cold;
    const warm = es.amdahl_prediction_warm_pp;
    if (typeof warm === 'number' && Number.isFinite(warm)) return warm;
    return null;
}

// ── E2E latency helpers (schema v4.0 map-based) ────────────────────────────────

// campaign-app.js owns the live parser shared through LG_HELPERS. Keep this
// fallback shape identical so the standalone circuit-board harness can still
// render archived state without loading the Alpine component first.
function _parseBucketTagFallback(rawTag) {
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

function _bucketRecords(bucketMap) {
    if (!bucketMap || typeof bucketMap !== 'object') return [];
    const shared = (typeof window !== 'undefined' && window.LG_HELPERS)
        ? window.LG_HELPERS.bucketRecords : null;
    if (typeof shared === 'function') return shared(bucketMap);
    return Object.keys(bucketMap)
        .map(_parseBucketTagFallback)
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

function _bucketIdentity(bucket) {
    return bucket.heterogeneous
        ? `il${bucket.inputLen}_ol${bucket.outputLen}_bs${bucket.batchSize}`
        : `bs${bucket.batchSize}`;
}

function _matchingBucket(bucketMap, sourceBucket) {
    const identity = _bucketIdentity(sourceBucket);
    return _bucketRecords(bucketMap).find(bucket => _bucketIdentity(bucket) === identity) || null;
}

// Pick the first deterministic workload bucket. The exact source tag is
// returned for map lookup; bare numeric tags therefore remain read-compatible.
function _primaryBsKey(latencyMap) {
    return _bucketRecords(latencyMap)[0]?.tag ?? null;
}

// Derive deterministic first/last bucket records plus the legacy BS range.
function _bsRange(latencyMap) {
    const buckets = _bucketRecords(latencyMap);
    if (!buckets.length) return null;
    const batchSizes = buckets.map(bucket => bucket.batchSize);
    return {
        first: buckets[0],
        last: buckets[buckets.length - 1],
        min: Math.min(...batchSizes),
        max: Math.max(...batchSizes),
        count: buckets.length,
        heterogeneous: buckets.some(bucket => bucket.heterogeneous),
    };
}

// Aggregate verdicts across workload buckets into a single traffic-light state:
//   GREEN — every verdict is PASS
//   RED   — every verdict is REGRESSED or CATASTROPHIC
//   AMBER — mixed (any non-PASS that isn't all-regress)
//   NONE  — no verdicts known (round 1 baseline, or pre-measurement)
function _verdictAggregate(perBsVerdict) {
    if (!perBsVerdict || typeof perBsVerdict !== 'object') return 'NONE';
    const vals = _bucketRecords(perBsVerdict)
        .map(bucket => perBsVerdict[bucket.tag])
        .filter(v => typeof v === 'string');
    if (!vals.length) return 'NONE';
    if (vals.every(v => v === 'PASS')) return 'GREEN';
    if (vals.every(v => v === 'REGRESSED' || v === 'CATASTROPHIC')) return 'RED';
    return 'AMBER';
}

// Synthesize a {bucket_tag: 'PASS' | 'REGRESSED'} verdict map by comparing this
// round's baseline latency map against the previous round's. Used to drive
// R2+ baseline (re-profile) chip coloring when no server-emitted per-bucket
// verdict is available — the re-profile is a pure measurement, not a
// gated diff, so the verdict is purely "this round faster vs prev round?".
//
//   curMap / prevMap shape: {<bucket_tag>: {avg, p50, ...}, ...}
//   PASS       — curAvg <= prevAvg (faster or equal)
//   REGRESSED  — curAvg >  prevAvg (slower)
// Tags present in only one map are skipped (no comparison possible).
// Returns null when the comparison yields zero usable exact-tag pairs.
function _synthesizeReprofileVerdict(curMap, prevMap) {
    if (!curMap || !prevMap) return null;
    if (typeof curMap !== 'object' || typeof prevMap !== 'object') return null;
    const out = {};
    let any = false;
    for (const bucket of _bucketRecords(curMap)) {
        const previousBucket = _matchingBucket(prevMap, bucket);
        if (!previousBucket) continue;
        const curAvg = curMap[bucket.tag]?.avg;
        const prevAvg = prevMap[previousBucket.tag]?.avg;
        if (typeof curAvg !== 'number' || typeof prevAvg !== 'number') continue;
        if (curAvg <= 0 || prevAvg <= 0) continue;
        out[bucket.tag] = curAvg <= prevAvg ? 'PASS' : 'REGRESSED';
        any = true;
    }
    return any ? out : null;
}

// Detect whether the previous round's integration failed/skipped/exhausted —
// the signal that nothing was integrated and this round's "baseline" is
// just a re-statement of prev round's baseline. Drives the disabled+⟲
// state on R2+ baseline chips.
function _prevIntegrationDisabled(rounds, ri) {
    if (ri <= 0) return false;
    const prev = rounds[ri - 1];
    const integStatus = String(prev?.roundData?.integration?.status || '').toLowerCase();
    if (integStatus === 'failed' || integStatus === 'skipped') return true;
    const roundStatus = String(prev?.roundData?.status || '').toUpperCase();
    if (roundStatus === 'EXHAUSTED' || roundStatus === 'FAILED') return true;
    return false;
}

// Format a latency-in-seconds for display. Large values in seconds, sub-1s in ms.
function _fmtLatency(seconds) {
    if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds <= 0) return '';
    return seconds >= 1 ? seconds.toFixed(2) + 's' : (seconds * 1000).toFixed(1) + 'ms';
}

// Client-side delta_pp: percentage-point improvement of `combinedMap` vs
// `baselineMap` at the first shared canonical bucket identity. Bare numeric
// legacy keys alias only their homogeneous `bsN` form; heterogeneous tuples
// still require exact IL/OL/BS identity.
function _computeDeltaPpFromMaps(baselineMap, combinedMap) {
    if (!baselineMap || !combinedMap) return null;
    if (typeof baselineMap !== 'object' || typeof combinedMap !== 'object') return null;
    const bucket = _bucketRecords(baselineMap).find(record => _matchingBucket(combinedMap, record));
    if (!bucket) return null;
    const combinedBucket = _matchingBucket(combinedMap, bucket);
    if (!combinedBucket) return null;
    const baseEntry = baselineMap[bucket.tag];
    const combEntry = combinedMap[combinedBucket.tag];
    const b = baseEntry?.avg;
    const c = combEntry?.avg;
    if (typeof b !== 'number' || typeof c !== 'number' || b <= 0 || c <= 0) return null;
    return ((b - c) / b) * 100;
}

function _computeSpeedupFromMaps(baselineMap, combinedMap) {
    if (!baselineMap || !combinedMap) return null;
    if (typeof baselineMap !== 'object' || typeof combinedMap !== 'object') return null;
    const bucket = _bucketRecords(combinedMap).find(record => _matchingBucket(baselineMap, record));
    if (!bucket) return null;
    const baselineBucket = _matchingBucket(baselineMap, bucket);
    if (!baselineBucket) return null;
    const baselineAvg = baselineMap[baselineBucket.tag]?.avg;
    const combinedAvg = combinedMap[bucket.tag]?.avg;
    if (typeof baselineAvg !== 'number' || typeof combinedAvg !== 'number') return null;
    if (baselineAvg <= 0 || combinedAvg <= 0) return null;
    return baselineAvg / combinedAvg;
}

// Render the tooltip HTML for a baseline / re-profile chip. Draws a full
// percentile table (bucket × {p10, p25, p50, avg, p75, p90, p99}) with the AVG
// column highlighted and row colouring from per_bs_verdict. Missing percentile
// cells render as em-dash "—". When perBsVerdict is absent the verdict legend
// is omitted (round 1 baseline).
function _renderPercentileTable(latencyMap, perBsVerdict) {
    if (!latencyMap || typeof latencyMap !== 'object') return '';
    const buckets = _bucketRecords(latencyMap);
    if (!buckets.length) return '';
    const heterogeneous = buckets.some(bucket => bucket.heterogeneous);
    const cols = ['p10', 'p25', 'p50', 'avg', 'p75', 'p90', 'p99'];
    const headerCells = cols.map(c => {
        const cls = c === 'avg' ? 'cb2-tt-avg-col' : '';
        return `<th class="${cls}">${c.toUpperCase()}</th>`;
    }).join('');
    const rows = buckets.map(bucket => {
        const entry = latencyMap[bucket.tag] || {};
        const verdict = (perBsVerdict || {})[bucket.tag];
        let rowClass = '';
        if (verdict === 'PASS') rowClass = 'row-pass';
        else if (verdict === 'NOISE') rowClass = 'row-noise';
        else if (verdict === 'REGRESSED' || verdict === 'CATASTROPHIC') rowClass = 'row-regress';
        const cells = cols.map(c => {
            const v = entry[c];
            const cls = c === 'avg' ? 'cb2-tt-avg-col' : '';
            const txt = (typeof v === 'number' && Number.isFinite(v)) ? v.toFixed(2) : '&mdash;';
            return `<td class="${cls}">${txt}</td>`;
        }).join('');
        const identityCells = heterogeneous
            ? `<td>${bucket.inputLen ?? '&mdash;'}</td><td>${bucket.outputLen ?? '&mdash;'}</td><td>${bucket.batchSize}</td>`
            : `<td>${bucket.batchSize}</td>`;
        return `<tr class="${rowClass}">${identityCells}${cells}</tr>`;
    }).join('');
    const identityHeader = heterogeneous ? '<th>IL</th><th>OL</th><th>BS</th>' : '<th>BS</th>';
    return `<table class="cb2-tt-table"><thead><tr>${identityHeader}${headerCells}</tr></thead><tbody>${rows}</tbody></table>`;
}

// Rich tooltip body builder — assembles full tooltip content per approved mockup.
// Args: { summary, note, latencyMap, perBsVerdict, legend, source }
// - summary: array of {label, value, cls} stat objects (cls = 'mint'|'cyan'|'amber'|'red')
// - note: string with {TERM} placeholders to highlight as <span class="hl">TERM</span>
// - latencyMap / perBsVerdict: passed to _renderPercentileTable
// - legend: boolean — show PASS/NOISE/REGRESSED dot legend
// - source: footer line (e.g. "vs R2 baseline · threshold ±5%")
function _buildRichTooltipBody({ summary, note, latencyMap, perBsVerdict, legend, source }) {
    let html = '';
    // Summary row
    if (summary && Array.isArray(summary) && summary.length) {
        html += `<div class="cb2-tt-summary">`;
        summary.forEach(s => {
            html += `<div class="stat"><div class="stat-label">${_esc(s.label)}</div>`;
            html += `<div class="stat-val ${s.cls || ''}">${_esc(s.value)}</div></div>`;
        });
        html += `</div>`;
    }
    // Note row with {TERM} highlights
    if (note) {
        const noteHtml = note.replace(/\{([^}]+)\}/g, (_, term) => `<span class="hl">${_esc(term)}</span>`);
        html += `<div class="cb2-tt-note">${noteHtml}</div>`;
    }
    // Percentile table (full latency map), or verdict-only table as fallback
    if (latencyMap && typeof latencyMap === 'object' && Object.keys(latencyMap).length) {
        html += _renderPercentileTable(latencyMap, perBsVerdict);
    } else if (perBsVerdict && typeof perBsVerdict === 'object' && Object.keys(perBsVerdict).length) {
        const buckets = _bucketRecords(perBsVerdict);
        if (buckets.length) {
            const heterogeneous = buckets.some(bucket => bucket.heterogeneous);
            const rows = buckets.map(bucket => {
                const v = perBsVerdict[bucket.tag] || '';
                let cls = 'row-noise';
                if (v === 'PASS') cls = 'row-pass';
                else if (v === 'REGRESSED' || v === 'CATASTROPHIC') cls = 'row-regress';
                const identityCells = heterogeneous
                    ? `<td>${bucket.inputLen ?? '&mdash;'}</td><td>${bucket.outputLen ?? '&mdash;'}</td><td>${bucket.batchSize}</td>`
                    : `<td>${bucket.batchSize}</td>`;
                return `<tr class="${cls}">${identityCells}<td>${_esc(v)}</td></tr>`;
            }).join('');
            const identityHeader = heterogeneous ? '<th>IL</th><th>OL</th><th>BS</th>' : '<th>BS</th>';
            html += `<table class="cb2-tt-table"><thead><tr>${identityHeader}<th>VERDICT</th></tr></thead><tbody>${rows}</tbody></table>`;
        }
    }
    // Footer: legend + source
    if (legend || source) {
        html += `<div class="cb2-tt-footer">`;
        if (legend) {
            html += `<span class="legend"><span class="dot pass"></span> PASS</span>`;
            html += `<span class="legend"><span class="dot noise"></span> NOISE</span>`;
            html += `<span class="legend"><span class="dot regress"></span> REGRESSED</span>`;
        }
        if (source) {
            html += `<div class="cb2-tt-src">${_esc(source)}</div>`;
        }
        html += `</div>`;
    }
    return html;
}

// ── Mining tooltip (§5 · profiling breakdown) ───────────────────────────────
// Builds the rich hover body for a MINING chip, matching mockup §5 exactly:
//   title row   : "MINING · R{n} — BOTTLENECK PROFILE" + meta (emitter)
//   tt-summary  : top share / amdahl × / decode frac
//   tt-note     : narrative sentence with {TERM} highlights
//   phase-bar   : DECODE vs PREFILL horizontal stacked bar
//   tt-breakdown: top-5 components with fill bars (ellipsis names)
//   tt-footer   : source path + methodology
// Args: miningData = { pct, component, amdahlCeiling, decodeFrac,
//                       componentBreakdown: [{name, pct}], emitter, sourcePath }
// Fields are all optional; builder degrades row-by-row.
function _buildMiningTooltipBody({ roundId, miningData, metaSuffix }) {
    const md = miningData || {};
    const topPct = (typeof md.pct === 'number') ? Math.round(md.pct) : null;
    const amd = (typeof md.amdahlCeiling === 'number') ? md.amdahlCeiling.toFixed(1) + '×' : '—';
    const decodeFrac = (typeof md.decodeFrac === 'number') ? md.decodeFrac : null;
    const decodePct = decodeFrac != null ? Math.round(decodeFrac * 100) : null;
    const prefillPct = decodePct != null ? (100 - decodePct) : null;
    const titleLabel = `MINING · R${roundId} — BOTTLENECK PROFILE`;
    const meta = metaSuffix ? `<span class="meta">${_esc(metaSuffix)}</span>` : '';
    let html = `<div class="cb2-tt-title"><span>${_esc(titleLabel)}</span>${meta}</div>`;

    // Summary row — violet/amber/mint stat columns matching the mockup.
    html += `<div class="cb2-tt-summary">`;
    html += `<div class="stat"><div class="stat-label">top share</div>` +
            `<div class="stat-val violet">${topPct != null ? topPct + '%' : '—'}</div></div>`;
    html += `<div class="stat"><div class="stat-label">amdahl ×</div>` +
            `<div class="stat-val amber">${_esc(amd)}</div></div>`;
    if (decodeFrac != null) {
        html += `<div class="stat"><div class="stat-label">decode frac</div>` +
                `<div class="stat-val mint">${decodeFrac.toFixed(2)}</div></div>`;
    }
    html += `</div>`;

    // Narrative line with {TERM} highlights.
    if (md.component || topPct != null) {
        const comp = md.component || '—';
        const phaseBit = decodePct != null
            ? `Of the {decode} phase ({${decodePct}%} of total E2E), `
            : '';
        const pctBit = topPct != null ? `{${topPct}%}` : '—';
        const noteRaw = `${phaseBit}${pctBit} sits in {${comp}}. ` +
            `Perfectly optimizing it caps at {${amd}} speedup.`;
        const noteHtml = noteRaw.replace(/\{([^}]+)\}/g, (_, t) => `<span class="hl">${_esc(t)}</span>`);
        html += `<div class="cb2-tt-note">${noteHtml}</div>`;
    }

    // Phase bar — only when decode_frac is known.
    if (decodePct != null && decodePct > 0 && decodePct <= 100) {
        html += `<div class="cb2-phase-bar" aria-hidden="true">`;
        html += `<div class="seg decode" style="flex: ${decodePct};">DECODE · ${decodePct}%</div>`;
        if (prefillPct > 0) {
            html += `<div class="seg prefill" style="flex: ${prefillPct};">PREFILL · ${prefillPct}%</div>`;
        }
        html += `</div>`;
    }

    // Component breakdown table.
    const rows = Array.isArray(md.componentBreakdown) ? md.componentBreakdown.slice(0, 5) : null;
    if (rows && rows.length) {
        const maxPct = Math.max(...rows.map(r => Number.isFinite(r.pct) ? r.pct : 0), 0.01);
        html += `<table class="cb2-tt-breakdown">` +
            `<thead><tr><th>COMPONENT</th><th style="width:40%">SHARE</th><th class="num">%</th></tr></thead><tbody>`;
        rows.forEach((r, i) => {
            const isTop = i === 0;
            const name = (r.name || '').length > 24 ? r.name.slice(0, 22) + '…' : (r.name || '—');
            const pctNum = Number.isFinite(r.pct) ? r.pct : 0;
            const fillPct = Math.max(0, Math.min(100, (pctNum / maxPct) * 100));
            html += `<tr class="${isTop ? 'top' : ''}">` +
                `<td>${_esc(name)}</td>` +
                `<td class="bar"><span class="bg"><span class="fill" style="width: ${fillPct.toFixed(1)}%;"></span></span></td>` +
                `<td class="num pct">${pctNum.toFixed(1)}</td>` +
                `</tr>`;
        });
        html += `</tbody></table>`;
    } else if (md.component && topPct != null) {
        // Fallback: we know the top component + pct but no multi-row breakdown.
        // Render a one-row degraded table so the reader still sees the data in
        // the same layout slot, with a muted hint about missing sub-shares.
        html += `<table class="cb2-tt-breakdown"><thead><tr>` +
            `<th>COMPONENT</th><th style="width:40%">SHARE</th><th class="num">%</th>` +
            `</tr></thead><tbody>`;
        const name = md.component.length > 24 ? md.component.slice(0, 22) + '…' : md.component;
        html += `<tr class="top">` +
            `<td>${_esc(name)}</td>` +
            `<td class="bar"><span class="bg"><span class="fill" style="width: 100%;"></span></span></td>` +
            `<td class="num pct">${topPct.toFixed(1)}</td>` +
            `</tr>`;
        html += `<tr><td colspan="3" style="color:rgba(232,232,240,0.30); font-style: italic;">` +
            `(sub-component breakdown not emitted for this round)</td></tr>`;
        html += `</tbody></table>`;
    }

    // Footer.
    const src = md.sourcePath || 'bottleneck_analysis.json';
    html += `<div class="cb2-tt-footer">` +
        `Source: <span style="color:#e8e8f0">${_esc(src)}</span>` +
        (md.emitter ? ` · ${_esc(md.emitter)}` : '') +
        `<div class="src">Amdahl ceiling assumes the bottleneck → 0 ms with remaining components unchanged</div>` +
        `</div>`;
    return html;
}

// ── Debate tooltip (D1 · scoreboard) ─────────────────────────────────────────
// Builds the rich hover body for a DEBATE chip, matching mockup D1 exactly:
//   title row   : "DEBATE · R{n} — SCOREBOARD" + meta (N champions · M rounds · K picked)
//   tt-note     : ranked-by narrative
//   scoreboard  : one row per candidate, sorted by weighted_total desc,
//                 winners in mint on top, eliminated grey below. Verdict
//                 column ✓/✗, TIER column with T1/T2/T3 color tokens.
//   tt-footer   : legend (shipped/eliminated + tier dots) + source
// Args: debateData = { championsCount, summaryRoundsCompleted, winners[],
//                      candidates: [{ opId, trackAssignment, feasibility,
//                        evidenceTier, expectedE2ePct, weightedTotal, isWinner }] }
function _buildDebateTooltipBody({ roundId, debateData, metaSuffix }) {
    const dd = debateData || {};
    const championsCount = dd.championsCount || (Array.isArray(dd.candidates) ? dd.candidates.length : null);
    const roundsDone = dd.summaryRoundsCompleted || null;
    const winnersCount = Array.isArray(dd.winners) ? dd.winners.length : null;
    const titleLabel = `DEBATE · R${roundId} — SCOREBOARD`;
    const metaBits = [];
    if (championsCount != null) metaBits.push(`${championsCount} champions`);
    if (roundsDone != null) metaBits.push(`${roundsDone} rounds`);
    if (winnersCount != null) metaBits.push(`${winnersCount} picked`);
    const meta = metaBits.length
        ? `<span class="meta">${_esc(metaBits.join(' · '))}</span>` : '';
    let html = `<div class="cb2-tt-title"><span>${_esc(titleLabel)}</span>${meta}</div>`;

    html += `<div class="cb2-tt-note">` +
        `Ranked by <span class="hl">weighted_total</span> (feasibility × evidence × expected E2E). ` +
        `Threshold for ship: <span class="hl">≥0.60</span>.` +
        `</div>`;

    // Scoreboard table — if we have candidates, render full; else render a
    // degraded row per winner so the tooltip still offers useful info.
    const candidates = Array.isArray(dd.candidates) ? dd.candidates.slice() : null;
    if (candidates && candidates.length) {
        // Sort by weighted_total desc (nulls last).
        candidates.sort((a, b) => {
            const av = (a.weightedTotal != null) ? a.weightedTotal : -Infinity;
            const bv = (b.weightedTotal != null) ? b.weightedTotal : -Infinity;
            return bv - av;
        });
        html += `<table class="cb2-tt-scoreboard"><thead><tr>` +
            `<th></th><th>CHAMPION</th><th class="num">E2E %</th>` +
            `<th>TIER</th><th class="num">SCORE</th>` +
            `</tr></thead><tbody>`;
        candidates.forEach(c => {
            const rowCls = c.isWinner ? 'winner' : 'loser';
            const verdict = c.isWinner
                ? `<span class="verdict win">✓</span>`
                : `<span class="verdict elim">✗</span>`;
            const rawName = c.displayName || c.opId || '—';
            const name = rawName.length > 24 ? rawName.slice(0, 22) + '…' : rawName;
            const e2e = (typeof c.expectedE2ePct === 'number')
                ? (c.expectedE2ePct >= 0 ? '+' : '') + c.expectedE2ePct.toFixed(1) : '—';
            const tier = c.evidenceTier || null;
            const tierHtml = tier
                ? `<span class="${tier === 'tier_1' ? 't1' : tier === 'tier_2' ? 't2' : 't3'}">` +
                  (tier === 'tier_1' ? 'TIER-1' : tier === 'tier_2' ? 'TIER-2' : 'TIER-3') +
                  `</span>`
                : '—';
            const score = (typeof c.weightedTotal === 'number') ? c.weightedTotal.toFixed(2) : '—';
            html += `<tr class="${rowCls}">` +
                `<td>${verdict}</td>` +
                `<td>${_esc(name)}</td>` +
                `<td class="num">${e2e}</td>` +
                `<td class="tier">${tierHtml}</td>` +
                `<td class="num score">${score}</td>` +
                `</tr>`;
        });
        html += `</tbody></table>`;
    } else if (Array.isArray(dd.winners) && dd.winners.length) {
        // Degraded path: only winners are known (no score_breakdown yet).
        html += `<table class="cb2-tt-scoreboard"><thead><tr>` +
            `<th></th><th>CHAMPION</th><th class="num">E2E %</th>` +
            `<th>TIER</th><th class="num">SCORE</th>` +
            `</tr></thead><tbody>`;
        dd.winners.forEach(name => {
            const shown = name.length > 24 ? name.slice(0, 22) + '…' : name;
            html += `<tr class="winner">` +
                `<td><span class="verdict win">✓</span></td>` +
                `<td>${_esc(shown)}</td>` +
                `<td class="num">—</td>` +
                `<td class="tier">—</td>` +
                `<td class="num score">—</td>` +
                `</tr>`;
        });
        html += `<tr><td colspan="5" style="color:rgba(232,232,240,0.30); font-style: italic;">` +
            `(per-candidate score breakdown not emitted for this round)` +
            `</td></tr>`;
        html += `</tbody></table>`;
    } else {
        // No winners, no candidates — show a placeholder line so the hover
        // still carries forward the narrative rather than going blank.
        html += `<div class="cb2-tt-note" style="color:rgba(232,232,240,0.45)">` +
            `No candidates scored yet — debate in progress.` +
            `</div>`;
    }

    html += `<div class="cb2-tt-footer">` +
        `<span class="legend"><span class="dot win"></span> shipped</span>` +
        `<span class="legend"><span class="dot elim"></span> eliminated</span>` +
        `<span class="legend"><span class="dot tier1"></span> T1 evidence</span>` +
        `<span class="legend"><span class="dot tier2"></span> T2</span>` +
        `<span class="legend"><span class="dot tier3"></span> T3</span>` +
        `<div class="src">Source: debate_rationale + debate.selected_candidates · R${roundId}</div>` +
        `</div>`;
    return html;
}

// Public: build the baseline tooltip HTML for a given round.
function _baselineTooltipHtml(state, roundId) {
    const by = state?._catalog?.baselineByRound || {};
    const b = by[roundId] ?? by[String(roundId)];
    if (!b) {
        return roundId > 1 ? `No re-profile captured for round ${roundId}.` : '';
    }
    if (b.source === 'integration_opt_s' && !(b.batchSizes && Object.keys(b.batchSizes).length)) {
        const lines = [
            `<span class="cb2-tt-dim">Derived from round ${b.derivedFromRound} integration opt_s.</span>`,
            `Latency: <span class="cb2-tt-val">${_fmtLatency(b.primaryBsLatencyMs / 1000)}</span>`,
        ];
        if (b.primaryBucket) lines.push(`Bucket: <span class="cb2-tt-val">${_esc(b.primaryBucket.label)}</span>`);
        else if (b.primaryBs != null) lines.push(`Batch size: <span class="cb2-tt-val">${b.primaryBs}</span>`);
        return lines.join('<br>');
    }
    const hasMap = b.batchSizes && Object.keys(b.batchSizes).length > 0;
    if (!hasMap) {
        // Scalar-only legacy entry
        return `Latency: <span class="cb2-tt-val">${_fmtLatency(b.primaryBsLatencyMs / 1000)}</span>`;
    }
    // Detect whether batchSizes holds v4.0 entry-shaped values (.avg, .p50 ...).
    const firstEntry = b.batchSizes[Object.keys(b.batchSizes)[0]];
    const looksLikeV4 = firstEntry && typeof firstEntry === 'object' && typeof firstEntry.avg === 'number';
    if (looksLikeV4) {
        let html = `<div class="cb2-tt-note">chip shows <span class="hl">AVG</span> (seconds)</div>`;
        html += _renderPercentileTable(b.batchSizes, b.perBsVerdict);
        const agg = _verdictAggregate(b.perBsVerdict);
        if (agg !== 'NONE') {
            html += `<div class="cb2-tt-footer">`;
            html += `<span class="legend"><span class="dot pass"></span> PASS</span>`;
            html += `<span class="legend"><span class="dot noise"></span> NOISE</span>`;
            html += `<span class="legend"><span class="dot regress"></span> REGRESSED</span>`;
            if (roundId > 1) {
                html += `<div class="src">vs round ${roundId - 1} baseline</div>`;
            }
            html += `</div>`;
        }
        return html;
    }
    // Legacy sidecar shape: batchSizes[tag] = {baseline_avg_s, baseline_p50_s, ...}
    const lines = [];
    for (const bucket of _bucketRecords(b.batchSizes)) {
        const data = b.batchSizes[bucket.tag];
        const avg = data.baseline_avg_s || data.avg_s;
        const p50 = data.baseline_p50_s || data.p50_s;
        if (avg) {
            let line = `${_esc(bucket.label)}: <span class="cb2-tt-val">${_fmtLatency(avg)}</span>`;
            if (p50) line += ` (p50: ${_fmtLatency(p50)})`;
            lines.push(line);
        }
    }
    return lines.join('<br>');
}

// ── Status helpers ─────────────────────────────────────────────────────────────

function nodeStatusClass(status) {
    const s = (status || '').toLowerCase();
    if (s === 'shipped' || s === 'passed')      return 'shipped';
    if (s === 'failed')                          return 'failed';
    if (s === 'active' || s === 'in_progress')  return 'active';
    return 'ghost';
}

function nodeStatusColor(status) {
    return CB.STATUS_COLOR[status] || CB.STATUS_COLOR[nodeStatusClass(status)] || 'var(--ghost)';
}

function traceClass(status) {
    const s = nodeStatusClass(status);
    if (s === 'shipped') return 'trace-shipped';
    if (s === 'failed')  return 'trace-failed';
    if (s === 'active')  return 'trace-active';
    return 'trace-ghost';
}

// ── Parse state.json into renderable rounds ────────────────────────────────────

function parseRounds(state) {
    const campaign  = state.campaign || {};
    const currentRoundId = campaign.current_round || 1;
    const rounds = campaign.rounds || [];
    const shippedOps = new Set((campaign.shipped_optimizations || [])
        .map(s => typeof s === 'string' ? s : (s && s.op_id) || null)
        .filter(Boolean));
    const currentStageIdx = CB.STAGE_ORDER[currentStage(state)] ?? 0;
    const result = [];

    // Dedup rounds by round_id (or fallback `round`), keeping entry with most
    // data. Shared with auditGateStates so a row and its audit gates always
    // come from the same entry.
    const dedupedRounds = dedupRoundsById(rounds);

    for (const r of dedupedRounds) {
        const rid  = r.round_id;
        const isCurrent = rid === currentRoundId;
        const stageNodes = CB.STAGE_COLS.map((_, colIdx) => {
            if (isCurrent) {
                return _currentRoundNodeStatus(state, colIdx, currentStageIdx, shippedOps);
            } else {
                return _pastRoundNodeStatus(r, colIdx, shippedOps);
            }
        });

        const rTracks = (r.parallel_tracks && r.parallel_tracks.tracks) || {};
        const selected = r.debate?.selected_winners || r.selected_candidates || [];
        result.push({
            roundId: rid, isCurrent, stageNodes,
            speedupAfter: r.cumulative_speedup_after ?? r.combined_e2e_speedup_x ?? null,
            shipped: r.shipped || [],
            selectedCandidates: selected,
            archivedTracks: rTracks,
            archivedIntegration: r.integration || null,
            roundData: r,
        });
    }

    // If no row was flagged as current, create a synthetic current-round row
    if (!result.some(r => r.isCurrent)) {
        const stageNodes = CB.STAGE_COLS.map((_, colIdx) =>
            _currentRoundNodeStatus(state, colIdx, currentStageIdx, shippedOps)
        );
        result.push({
            roundId: currentRoundId, isCurrent: true, stageNodes,
            speedupAfter: campaign.cumulative_e2e_speedup ?? 1.0,
            shipped: [...shippedOps],
            selectedCandidates: Object.keys(currentTracks(state)),
        });
    }

    return result;
}

function _currentRoundNodeStatus(state, colIdx, currentStageIdx, shippedOps) {
    const tracks = currentTracks(state);
    const trackCount = Object.keys(tracks).length;

    if (colIdx > currentStageIdx) {
        return { status: 'pending', label: CB.STAGE_COLS[colIdx], detail: '' };
    }
    if (colIdx === currentStageIdx) {
        return { status: 'active', label: CB.STAGE_COLS[colIdx], detail: `${trackCount} ops` };
    }

    let shipped = 0, failed = 0;
    for (const [opId, t] of Object.entries(tracks)) {
        if (shippedOps.has(opId)) shipped++;
        else if ((t.status || '').toUpperCase() === 'FAILED') failed++;
    }

    if (shipped > 0 && failed === 0) {
        return { status: 'shipped', label: CB.STAGE_COLS[colIdx], detail: `${shipped} shipped` };
    }
    if (failed > 0) {
        return { status: 'failed', label: CB.STAGE_COLS[colIdx], detail: `${failed} failed` };
    }
    return { status: 'shipped', label: CB.STAGE_COLS[colIdx], detail: 'done' };
}

function _pastRoundNodeStatus(round, colIdx, shippedOps) {
    const shipped = (round.shipped || []).length;
    const archivedTracks = (round.parallel_tracks && round.parallel_tracks.tracks) || {};
    const archivedCount = Object.keys(archivedTracks).length;
    const selected = round.debate?.selected_winners || round.selected_candidates || [];
    const candidates = archivedCount || selected.length;

    if (colIdx === 5) {
        if (shipped > 0) return { status: 'shipped', label: CB.STAGE_COLS[colIdx], detail: `+${shipped}` };
        return { status: 'failed', label: CB.STAGE_COLS[colIdx], detail: 'none shipped' };
    }
    if (colIdx >= 3) {
        if (shipped > 0) return { status: 'shipped', label: CB.STAGE_COLS[colIdx], detail: `${shipped}/${candidates}` };
        if (candidates > 0) return { status: 'failed', label: CB.STAGE_COLS[colIdx], detail: `0/${candidates}` };
        return { status: 'ghost', label: CB.STAGE_COLS[colIdx], detail: '' };
    }
    return { status: 'shipped', label: CB.STAGE_COLS[colIdx], detail: 'done' };
}

// ── Layout constants (HUD hybrid) ────────────────────────────────────────────
const L2 = {
    X_START: 110,
    STAGE_W: 200,
    STAGE_H: 100,
    // Inter-chip gaps sized for the audit-gate fuse hex + its "AUD SXX"
    // micro-label sitting mid-trace (widened from 32/28, which left the
    // label ~8px from both chip edges).
    STAGE_GAP: 56,
    TRACK_W: 260,
    TRACK_H: 100,
    INTEG_W: 180,
    INTEG_H: 90,
    EVAL_SIZE: 32,
    ROW_SPACING: 190,
    BRANCH_GAP: 40,
    TRACE_WEIGHT: 2.5,
    VIA_R: 5,
    LOOP_X: 44,
    SVG_W: 1600,
};

// ── Tooltip system ────────────────────────────────────────────────────────────
let _cb2Tip = null;
let _lastMouseMoveHandler = null;

function _initTooltip(mount) {
    if (_cb2Tip && _cb2Tip.el && _cb2Tip.el.parentNode) {
        _cb2Tip.el.parentNode.removeChild(_cb2Tip.el);
    }
    const tip = document.createElement('div');
    tip.className = 'cb2-tooltip';
    tip.innerHTML = '<div class="cb2-tt-label"></div><div class="cb2-tt-body"></div>';
    (mount || document.body).appendChild(tip);
    const labelEl = tip.querySelector('.cb2-tt-label');
    const bodyEl = tip.querySelector('.cb2-tt-body');

    function positionTip(evt) {
        const pad = 16;
        // A FocusEvent (or any non-pointer event) has no clientX/clientY, and
        // undefined + pad is NaN, which the CSSOM discards and leaves the
        // tooltip wherever it last was. Fall back to the target's own box.
        const cx = Number.isFinite(evt && evt.clientX) ? evt.clientX : null;
        const cy = Number.isFinite(evt && evt.clientY) ? evt.clientY : null;
        if (cx === null || cy === null) {
            const el = evt && evt.target;
            const box = el && typeof el.getBoundingClientRect === 'function'
                ? el.getBoundingClientRect() : null;
            evt = {
                clientX: box ? box.left + box.width / 2 : 0,
                clientY: box ? box.top + box.height / 2 : 0,
            };
        }
        let x = evt.clientX + pad, y = evt.clientY + pad;
        const tw = tip.offsetWidth, th = tip.offsetHeight;
        if (x + tw > window.innerWidth - 10) x = evt.clientX - tw - pad;
        if (y + th > window.innerHeight - 10) y = evt.clientY - th - pad;
        tip.style.left = x + 'px';
        tip.style.top = y + 'px';
    }

    // Track of the last-applied variant class so we can swap it cleanly
    // (e.g. tt-mining vs tt-debate d1) when the tooltip moves between chips.
    let _variantClasses = [];
    _cb2Tip = {
        el: tip,
        show(evt, label, body, variantClasses) {
            if (label) {
                labelEl.textContent = label;
                labelEl.style.display = '';
            } else {
                labelEl.textContent = '';
                labelEl.style.display = 'none';
            }
            bodyEl.innerHTML = body;
            // Swap variant classes (if provided). Pass e.g. ['tt-mining'] or
            // ['tt-debate','d1']; pass empty/null to reset back to default.
            if (_variantClasses.length) {
                tip.classList.remove(..._variantClasses);
                _variantClasses = [];
            }
            if (Array.isArray(variantClasses) && variantClasses.length) {
                tip.classList.add(...variantClasses);
                _variantClasses = variantClasses.slice();
            }
            tip.classList.add('visible');
            positionTip(evt);
        },
        hide() { tip.classList.remove('visible'); },
        position: positionTip,
    };

    if (_lastMouseMoveHandler) {
        document.removeEventListener('mousemove', _lastMouseMoveHandler);
    }
    _lastMouseMoveHandler = (e) => {
        if (tip.classList.contains('visible')) positionTip(e);
    };
    document.addEventListener('mousemove', _lastMouseMoveHandler);
}

// ── Data mapping: parseRounds() -> mockup format ─────────────────────────────

function mapRoundsToMockup(state, rounds) {
    const campaign = state.campaign || {};
    const opNames = opDisplayNames(state);
    // op_id list → display-name list for detail/secondary lines.
    const _named = ids => (ids || []).map(id => opNames.get(id) || id);
    const currentStageIdx = CB.STAGE_ORDER[currentStage(state)] ?? 0;
    const auditByRound = auditGateStates(state);
    const baselineByRound = state._catalog?.baselineByRound || {};
    const miningByRound   = state._catalog?.miningByRound   || {};
    const debateByRound   = state._catalog?.debateByRound   || {};
    const curDebate = currentDebate(state);
    const liveDebateRoundsCompleted = curDebate.rounds_completed || 0;
    // team_name lives at round level in v2, not inside debate.
    const liveDebateTeamActive = !!(_currentRoundObj(state).team_name);

    function _baselineHero(roundId) {
        const b = baselineByRound[roundId];
        if (!b || !b.primaryBsLatencyMs) return '';
        const ms = b.primaryBsLatencyMs;
        const derived = b.source === 'integration_opt_s';
        // Prepend ⟲ marker when the latency is derived from the prior round's
        // integration opt_s (no dedicated re-profile sweep ran). Signals the user
        // that it's the carry-forward measurement, not a fresh profile.
        const val = ms >= 1000 ? (ms / 1000).toFixed(2) + 's' : ms.toFixed(1) + 'ms';
        return derived ? '\u27F2 ' + val : val;
    }
    function _baselineSecondary(roundId) {
        const b = baselineByRound[roundId];
        if (!b) return '';
        if (b.primaryBucket) return b.primaryBucket.label;
        return b.primaryBs == null ? '' : `BS=${b.primaryBs}`;
    }
    function _baselineTooltip(roundId) {
        const b = baselineByRound[roundId];
        if (b && b.source === 'integration_opt_s') {
            // Derived re-profile — show provenance so the user can trace it.
            const lines = [
                `<span class="cb2-tt-dim">Derived from round ${b.derivedFromRound} integration opt_s.</span>`,
                `Latency: <span class="cb2-tt-val">${(b.primaryBsLatencyMs/1000).toFixed(3)}s</span>`,
            ];
            if (b.primaryBucket) lines.push(`Bucket: <span class="cb2-tt-val">${_esc(b.primaryBucket.label)}</span>`);
            else if (b.primaryBs != null) lines.push(`Batch size: <span class="cb2-tt-val">${b.primaryBs}</span>`);
            return lines.join('<br>');
        }
        if (!b || !b.batchSizes) {
            return roundId > 1
                ? `No re-profile captured for round ${roundId}.`
                : '';
        }
        const lines = [];
        for (const bucket of _bucketRecords(b.batchSizes)) {
            const data = b.batchSizes[bucket.tag];
            const avg = data.baseline_avg_s || data.avg_s;
            const p50 = data.baseline_p50_s || data.p50_s;
            if (avg) {
                const avgMs = avg * 1000;
                let line = `${_esc(bucket.label)}: <span class="cb2-tt-val">${avgMs.toFixed(1)}ms</span>`;
                if (p50) line += ` (p50: ${(p50 * 1000).toFixed(1)}ms)`;
                lines.push(line);
            }
        }
        return lines.join('<br>');
    }
    function _miningHero(roundId, bottleneckPctOverride) {
        const m = miningByRound[roundId];
        if (!m) {
            return bottleneckPctOverride ? `${Math.round(bottleneckPctOverride)}%` : '';
        }
        const comp = m.component;
        const pct = bottleneckPctOverride || m.pct;
        if (comp && pct) {
            const short = comp.length > 18 ? comp.slice(0, 16) + '\u2026' : comp;
            return `${short} ${Math.round(pct)}%`;
        }
        if (pct) return `${Math.round(pct)}%`;
        return '';
    }
    function _miningSecondary(roundId) {
        const m = miningByRound[roundId];
        if (!m?.amdahlCeiling) return '';
        return `ceiling: ${m.amdahlCeiling.toFixed(1)}\u00d7`;
    }
    function _miningTooltip(roundId) {
        const m = miningByRound[roundId];
        if (!m) return '';
        const lines = [];
        if (m.amdahlCeiling) lines.push(`Amdahl ceiling: <span class="cb2-tt-val">${m.amdahlCeiling.toFixed(2)}\u00d7</span>`);
        if (m.component) lines.push(`Component: <span class="cb2-tt-val">${_esc(m.component)}</span>`);
        return lines.join('<br>');
    }
    // Debate hero — hard rule:
    //   "N ROUNDS"  ⇐ currentDebate(state).rounds_completed (current round
    //                 only; the debate team state is scoped to the current round).
    //   "M CHAMPIONS" ⇐ round-scoped debate_rationale champion_id set.
    // Past rounds: show winners if a summary sidecar exists, else champion count.
    // If nothing known for this round, return '' — no cross-round leakage.
    function _debateHero(roundId, isCurrent, shippedCount, selectedCount) {
        const d = debateByRound[roundId];
        if (isCurrent && liveDebateTeamActive) {
            if (liveDebateRoundsCompleted > 0) {
                return liveDebateRoundsCompleted === 1
                    ? `1 ROUND`
                    : `${liveDebateRoundsCompleted} ROUNDS`;
            }
            if (d?.championsCount) return `${d.championsCount} CHAMPIONS`;
            return 'DEBATE STARTING';
        }
        // Past rounds: prefer outcome-oriented hero — "SHIPPED" count is the
        // single most informative number once the round is over. Fall back to
        // debate-team sidecars only if shipped count isn't known.
        if (Number.isFinite(shippedCount) && shippedCount > 0) {
            if (Number.isFinite(selectedCount) && selectedCount > 0) {
                return `${shippedCount}/${selectedCount} SHIPPED`;
            }
            return `${shippedCount} SHIPPED`;
        }
        if (d?.summaryRoundsCompleted && d.summaryRoundsCompleted > 1) return `${d.summaryRoundsCompleted} ROUNDS`;
        if (d?.winners?.length) return `${d.winners.length} WINNERS`;
        if (d?.championsCount) return `${d.championsCount} CHAMPIONS`;
        return '';
    }
    function _debateSecondary(roundId, isCurrent, shippedOps) {
        const d = debateByRound[roundId];
        if (isCurrent && liveDebateTeamActive) {
            // Primary slot already shows rounds/champions; secondary shows the
            // OTHER of those two when both known. Rationale count is intentionally
            // omitted here — it's already surfaced by the "RATIONALE ×N" stack
            // label below the chip, and rendering it inline was clipping the chip.
            if (liveDebateRoundsCompleted > 0 && d?.championsCount) return `${d.championsCount} CHAMPIONS`;
            return '';
        }
        // Past rounds: if the round shipped ops, surface their names as the
        // secondary line — that's the concrete outcome readers want to see.
        if (Array.isArray(shippedOps) && shippedOps.length > 0) {
            const joined = _named(shippedOps).join(', ');
            return joined.length > 30 ? joined.slice(0, 28) + '\u2026' : joined;
        }
        if (d?.winners?.length) {
            const joined = _named(d.winners).join(', ');
            return joined.length > 30 ? joined.slice(0, 28) + '\u2026' : joined;
        }
        if (d?.championsCount && d?.summaryRoundsCompleted && d.summaryRoundsCompleted > 1) {
            return `${d.championsCount} CHAMPIONS`;
        }
        return '';
    }
    function _debateTooltip(roundId, isCurrent) {
        const d = debateByRound[roundId];
        const lines = [];
        if (isCurrent) {
            if (liveDebateRoundsCompleted) lines.push(`Rounds completed: <span class="cb2-tt-val">${liveDebateRoundsCompleted}</span>`);
        } else if (d?.summaryRoundsCompleted) {
            lines.push(`Rounds: <span class="cb2-tt-val">${d.summaryRoundsCompleted}</span>`);
        }
        if (d?.championsCount) lines.push(`Champions: <span class="cb2-tt-val">${d.championsCount}</span>`);
        if (d?.rationalesCount) lines.push(`Rationale artifacts: <span class="cb2-tt-val">${d.rationalesCount}</span>`);
        if (d?.winners?.length) lines.push(`Winners: ${_named(d.winners).map(_esc).join(', ')}`);
        if (!lines.length) return '';
        return lines.join('<br>');
    }

    const roundSpeedups = {};
    for (const r of rounds) {
        if (r.speedupAfter) roundSpeedups[r.roundId] = r.speedupAfter;
    }

    // Live-slot attribution. currentTracks(state)/currentIntegration
    // dereference campaign.rounds[current_round-1] — the live round's state.
    const liveTracks = currentTracks(state);
    const liveInteg  = currentIntegration(state);
    const curRid = campaign.current_round || 1;
    const curRoundMeta = (campaign.rounds || []).find(rd => rd.round_id === curRid);
    const curSelected = curRoundMeta?.debate?.selected_winners || curRoundMeta?.selected_candidates;
    const curHasOwnCandidates = !!(curSelected && curSelected.length);
    // Schema-aligned integration.status enum (10 values):
    //   pending, in_progress, validated, single_pass, combined, gated_pass,
    //   completed, exhausted, failed, skipped.
    // Terminal (integration work is done): all except pending/in_progress.
    // 'complete' (without the -d) is a legacy alias kept for backwards-compat.
    const liveIntegStatusRaw = String(liveInteg.status || '').toLowerCase();
    const INTEG_TERMINAL = new Set([
        'complete', 'completed', 'validated', 'single_pass', 'combined',
        'gated_pass', 'exhausted', 'failed', 'skipped',
    ]);
    const liveIntegDone = INTEG_TERMINAL.has(liveIntegStatusRaw);
    let liveSlotsOwnerRoundId = curRid;
    if (curRid > 1 && !curHasOwnCandidates && Object.keys(liveTracks).length > 0 && liveIntegDone) {
        const priorRid = curRid - 1;
        const priorRound = rounds.find(r => r.roundId === priorRid);
        const priorArchived = (priorRound?.roundData?.parallel_tracks && priorRound.roundData.parallel_tracks.tracks) || null;
        if (priorRound && (!priorArchived || Object.keys(priorArchived).length === 0)) {
            liveSlotsOwnerRoundId = priorRid;
        }
    }

    return rounds.map((r, ri) => {
        const isCurrent = r.isCurrent;
        const mockRound = { id: r.roundId, stages: [], tracks: [], integration: null, eval: null, active: isCurrent };
        mockRound.auditGates = auditByRound[r.roundId] || null;

        if (!isCurrent) {
            // Post-T16 pipeline. R1 has a fresh BASELINE measurement; R2+
            // emits the same chip but relabelled "RE-PROFILE" — same colIdx
            // 0, same data pipeline, verdict-driven coloring. Per-round
            // stage list:
            //   R1                              : BASELINE → MINING → DEBATE
            //   R2+ after prev SHIPPED          : RE-PROFILE → MINING → DEBATE
            //   R2+ after prev EXHAUSTED/FAILED : RE-PROFILE →           DEBATE
            // The prev-round lookup uses the iteration neighbour (rounds are
            // already sorted ascending by round_id in ingestCampaignState).
            const prevRound = ri > 0 ? rounds[ri - 1] : null;
            const prevStatus = String(prevRound?.roundData?.status || '').toUpperCase();
            const prevExhausted = prevStatus === 'EXHAUSTED' || prevStatus === 'FAILED';
            const includeBaseline = true;  // every round shows the col-0 baseline chip
            const includeMining   = r.roundId === 1 || (r.roundId > 1 && !prevExhausted);

            if (includeBaseline) {
                const baseElapsed = _stageElapsed(state, '1_baseline', r.roundId);
                const baseHero = _baselineHero(r.roundId);
                const baseRound = baselineByRound[r.roundId];
                // v4.0 dual-hero inputs: pass the full latency map + per-bucket verdict
                // down to makeStageChip so it can render best→worst + traffic-light.
                // batchSizes on baselineByRound is populated from baseline.e2e_latency
                // (or integration.e2e_latency_combined via carry-forward).
                const baseLatencyMap = (baseRound && baseRound.batchSizes
                    && typeof baseRound.batchSizes === 'object'
                    && Object.keys(baseRound.batchSizes).length > 0)
                        ? baseRound.batchSizes : null;
                // R2+ verdict: prefer server-emitted baseline.per_bs_verdict;
                // otherwise synthesize by comparing this round's baseline map
                // against prev round's. Re-profile is a pure measurement, so
                // PASS/REGRESSED is purely "faster vs prev?".
                let basePerBsVerdict = baseRound?.perBsVerdict || null;
                if (r.roundId > 1 && !basePerBsVerdict) {
                    const prevBaseRound = baselineByRound[r.roundId - 1];
                    const prevMap = prevBaseRound?.batchSizes || null;
                    basePerBsVerdict = _synthesizeReprofileVerdict(baseLatencyMap, prevMap);
                }
                const integDisabled = _prevIntegrationDisabled(rounds, ri);
                mockRound.stages.push({
                    name: 'BASELINE',
                    colIdx: 0,
                    detail: r.roundId === 1 ? 'Initial model profiling' : 'Re-profile after integration',
                    value: baseHero,
                    latencyMap: baseLatencyMap,
                    perBsVerdict: basePerBsVerdict,
                    secondaryValue: _baselineSecondary(r.roundId),
                    secondaryColor: 'var(--dim)', elapsed: baseElapsed,
                    tooltipExtras: _baselineTooltip(r.roundId),
                    designation: `U${(r.roundId - 1) * 3 + 1}`,
                    reprofileRound: r.roundId > 1,
                    integrationDisabled: integDisabled,
                });
            }
            if (includeMining) {
                const bottleneckPct = r.roundData?.bottleneck_mining?.top_bottleneck_share_pct;
                const mineElapsed = _stageElapsed(state, '2_bottleneck_mining', r.roundId);
                const mineData = state._catalog?.miningByRound?.[r.roundId];
                mockRound.stages.push({
                    name: 'MINING',
                    colIdx: 1,
                    detail: mineData?.component ? `${mineData.component} bottleneck` : 'Bottleneck analysis',
                    value: _miningHero(r.roundId, bottleneckPct), secondaryValue: _miningSecondary(r.roundId),
                    secondaryColor: 'var(--dim)', elapsed: mineElapsed,
                    tooltipExtras: _miningTooltip(r.roundId), designation: `U${(r.roundId - 1) * 3 + 2}`,
                    miningData: mineData || null,
                    miningPctOverride: bottleneckPct,
                });
            }
            const debElapsed = _stageElapsed(state, '3_debate', r.roundId);
            const debData = state._catalog?.debateByRound?.[r.roundId];
            const pastShippedOps = Array.isArray(r.shipped) ? r.shipped : [];
            const pastSelectedCount = Array.isArray(r.selectedCandidates) ? r.selectedCandidates.length : null;
            mockRound.stages.push({
                name: 'DEBATE',
                colIdx: 2,
                detail: debData?.winners?.length ? _named(debData.winners).join(', ') : 'Strategy selection',
                value: _debateHero(r.roundId, false, pastShippedOps.length, pastSelectedCount),
                secondaryValue: _debateSecondary(r.roundId, false, pastShippedOps),
                secondaryColor: 'var(--dim)', elapsed: debElapsed,
                tooltipExtras: _debateTooltip(r.roundId, false), designation: `U${(r.roundId - 1) * 3 + 3}`,
                debateData: debData || null,
                debateShippedOps: pastShippedOps,
            });

            // Past-round data sources, in order of preference:
            //   1. round.parallel_tracks.tracks (authoritative per-round record)
            //   2. r.shipped thin summary (last-resort fallback)
            //   3. Live current-round tracks when this row owns the live slots
            //      (orchestrator scaffolded the next round's debate without the
            //      prior round's tracks being marked complete — see
            //      liveSlotsOwnerRoundId).
            let archivedTracks = (r.roundData?.parallel_tracks && r.roundData.parallel_tracks.tracks) || null;
            let archivedInteg  = r.roundData?.integration || null;
            const ownsLiveSlots = liveSlotsOwnerRoundId === r.roundId;
            if (ownsLiveSlots) {
                if (!archivedTracks || Object.keys(archivedTracks).length === 0) {
                    archivedTracks = { ...liveTracks };
                }
                if (!archivedInteg && liveInteg && (liveInteg.status || liveInteg.passing_candidates)) {
                    archivedInteg = { ...liveInteg };
                }
            }
            const shipped = new Set(r.shipped || []);
            if (ownsLiveSlots && shipped.size === 0) {
                for (const s of (campaign.shipped_optimizations || [])) {
                    const opId = typeof s === 'string' ? s : s?.op_id;
                    if (opId) shipped.add(opId);
                }
            }
            const shippedMeta = new Map();
            for (const s of (state.campaign?.shipped_optimizations || [])) {
                if (s && s.op_id && (s.round === r.roundId || s.round == null)) {
                    shippedMeta.set(s.op_id, s);
                }
            }

            if (archivedTracks && Object.keys(archivedTracks).length > 0) {
                for (const [opId, t] of Object.entries(archivedTracks)) {
                    const verdict = String(t.verdict || t.status || '').toUpperCase();
                    const isShipped = shipped.has(opId);
                    let status;
                    if (isShipped) status = 'shipped';
                    else if (verdict === 'GATED_PASS' || verdict === 'GATED-PASS') status = 'validated';
                    else if (verdict === 'GPU_BLOCKED') status = 'blocked';
                    else if (verdict === 'GATING_REQUIRED') status = 'gating';
                    else if (verdict === 'FAIL' || verdict === 'FAILED') status = 'failed';
                    else if (verdict === 'PASS' || verdict === 'PASSED') status = 'validated';
                    else status = 'shipped';
                    const kSpeedScalar = _kernelSpeedupFromTrack(t);
                    const eSpeedScalar = _e2eSpeedupScalar(t.e2e_speedup);
                    const amdahlPp = _amdahlPredictionPp(t.e2e_speedup);
                    const heroSpeedup = kSpeedScalar != null ? kSpeedScalar : (eSpeedScalar != null ? eSpeedScalar : 0);
                    mockRound.tracks.push({
                        name: opId,
                        displayName: opNames.get(opId) || null,
                        status,
                        speedup: heroSpeedup ? heroSpeedup.toFixed(2) : '0',
                        detail: t.fail_reason || t.failure_reason || opNames.get(opId) || opId.replace(/_/g, ' '),
                        failReason: t.failure_reason || t.fail_reason || null,
                        lossy: !!(t.classification && t.classification.toLowerCase() === 'lossy'),
                        kernelSpeedup: kSpeedScalar,
                        e2eSpeedup: eSpeedScalar,
                        amdahlPredictionPp: amdahlPp,
                        perBsVerdict: t.per_bs_verdict || null,
                        e2eLatencyOpt: t.e2e_latency_opt || null,
                        commitSha: t.commit_sha || null,
                    });
                }
            } else {
                // Tracks archive missing — fall back to the shipped_optimizations
                // metadata so past-round rows still render a mint chip per shipped op.
                const allOps = new Set([...shipped, ...(r.selectedCandidates || [])]);
                for (const opId of allOps) {
                    const meta = shippedMeta.get(opId) || {};
                    const isShipped = shipped.has(opId);
                    const metaE2EPp = meta.individual_e2e_delta_pp ?? meta.individual_e2e_delta_pp_amdahl ?? null;
                    const metaSpeedup = metaE2EPp != null ? 1 + (metaE2EPp / 100) : null;
                    const heroSpeedup = metaSpeedup != null ? metaSpeedup : 0;
                    const classification = (meta.classification || '').toString();
                    mockRound.tracks.push({
                        name: opId,
                        displayName: opNames.get(opId) || null,
                        status: isShipped ? 'shipped' : 'shipped',
                        speedup: heroSpeedup ? heroSpeedup.toFixed(2) : '0',
                        detail: opNames.get(opId) || opId.replace(/_/g, ' '),
                        failReason: null,
                        lossy: !!(classification && classification.toLowerCase() === 'lossy'),
                        kernelSpeedup: null,
                        e2eSpeedup: null,
                        amdahlPredictionPp: metaE2EPp,
                        commitSha: null,
                    });
                }
            }

            const combinedSpeedup = r.roundData?.combined_e2e_speedup_x;
            const cumulativeAfter = r.speedupAfter ?? combinedSpeedup ?? null;
            // v4.0 per-bucket verdict + this-round speedup. `archivedInteg` is the
            // round's integration object; its `per_bs_verdict` drives the chip
            // traffic light. `thisRoundSpeedup` compares the first exact shared
            // canonical workload tag; same-BS heterogeneous slices never alias.
            const pastPerBsVerdict = archivedInteg?.per_bs_verdict || null;
            const pastIntegMap = archivedInteg?.e2e_latency_combined || null;
            const prevRoundCarry = rounds[ri - 1];
            const prevBaseMap = prevRoundCarry?.roundData?.baseline?.e2e_latency
                || prevRoundCarry?.roundData?.integration?.e2e_latency_combined
                || null;
            let pastThisRoundSpeedup = null;
            if (pastIntegMap && typeof pastIntegMap === 'object' && r.roundId > 1) {
                pastThisRoundSpeedup = _computeSpeedupFromMaps(prevBaseMap, pastIntegMap);
            }
            if (shipped.size > 0 && cumulativeAfter && cumulativeAfter > 1) {
                const prevRound = rounds[ri - 1];
                const prevSpeedup = prevRound?.speedupAfter || 1.0;
                // Delta is a percentage-points number. Prefer the server-emitted
                // combined_e2e_delta_pp when present (authoritative); otherwise
                // derive from (this_x - prev_x) * 100. NEVER store a multiplier
                // delta and NEVER render it with "×".
                const serverDeltaPp = r.roundData?.combined_e2e_delta_pp;
                const deltaPp = typeof serverDeltaPp === 'number' && Number.isFinite(serverDeltaPp)
                    ? serverDeltaPp
                    : (cumulativeAfter - prevSpeedup) * 100;
                const integElapsed = _stageElapsed(state, '6_integration', r.roundId);
                mockRound.integration = {
                    speedup: cumulativeAfter.toFixed(2),
                    deltaPp: deltaPp > 0 ? deltaPp : null,
                    thisRoundSpeedup: pastThisRoundSpeedup
                        || (r.roundId === 1 ? 1.0 : ((deltaPp / 100) + 1)),
                    cumulativeSpeedup: cumulativeAfter,
                    perBsVerdict: pastPerBsVerdict,
                    combinedMap: pastIntegMap,
                    mergedOps: archivedInteg?.passing_candidates || r.shipped || [],
                    elapsed: integElapsed,
                };
            } else if (archivedInteg && String(archivedInteg.status || '').toLowerCase() !== 'pending') {
                const integElapsed = _stageElapsed(state, '6_integration', r.roundId);
                const integStatus = String(archivedInteg.status || '').toLowerCase();
                mockRound.integration = {
                    speedup: (cumulativeAfter || 1).toFixed(2),
                    deltaPp: null,
                    thisRoundSpeedup: pastThisRoundSpeedup || (r.roundId === 1 ? 1.0 : null),
                    cumulativeSpeedup: cumulativeAfter,
                    perBsVerdict: pastPerBsVerdict,
                    combinedMap: pastIntegMap,
                    mergedOps: archivedInteg.passing_candidates || [],
                    failedOps: archivedInteg.failed_candidates || [],
                    status: integStatus === 'complete' ? 'completed' : integStatus,
                    elapsed: integElapsed,
                };
            }

            const roundStatus = r.roundData?.status;
            const stopStatuses = new Set(['EXHAUSTED', 'FAILED']);
            mockRound.eval = !roundStatus ? 'CONTINUE' : stopStatuses.has(roundStatus) ? 'STOP' : 'CONTINUE';
            mockRound.evalNote = r.roundData?.note || null;
        } else {
            // Post-T16 pipeline. R1 starts with BASELINE; R2+ shows the
            // re-profile chip at col 0 (carry-forward latency from prev
            // round's integration.e2e_latency_combined). Stage list:
            //   R1                              : BASELINE → MINING → DEBATE
            //   R2+ after prev SHIPPED          : BASELINE → MINING → DEBATE
            //   R2+ after prev EXHAUSTED/FAILED : BASELINE →           DEBATE
            const prevRoundForLive = ri > 0 ? rounds[ri - 1] : null;
            const prevLiveStatus = String(prevRoundForLive?.roundData?.status || '').toUpperCase();
            const prevLiveExhausted = prevLiveStatus === 'EXHAUSTED' || prevLiveStatus === 'FAILED';

            const stages = [];
            let stageNames, stageKeys, stageMap;
            if (r.roundId === 1) {
                stageNames = ['BASELINE', 'MINING', 'DEBATE'];
                stageKeys  = ['1_baseline', '2_bottleneck_mining', '3_debate'];
                stageMap   = [0, 1, 2];
            } else if (prevLiveExhausted) {
                stageNames = ['BASELINE', 'DEBATE'];
                stageKeys  = ['1_baseline', '3_debate'];
                stageMap   = [0, 2];
            } else {
                stageNames = ['BASELINE', 'MINING', 'DEBATE'];
                stageKeys  = ['1_baseline', '2_bottleneck_mining', '3_debate'];
                stageMap   = [0, 1, 2];
            }

            const curDebData = state._catalog?.debateByRound?.[r.roundId];
            const curMineData = state._catalog?.miningByRound?.[r.roundId];
            const liveIntegDisabled = _prevIntegrationDisabled(rounds, ri);
            for (let si = 0; si < stageNames.length; si++) {
                const stageIdx = stageMap[si];
                const isComplete = stageIdx < currentStageIdx;
                const isActive = stageIdx === currentStageIdx;
                if (isComplete || isActive || stageIdx <= currentStageIdx) {
                    let value = '', secondaryValue = '', secondaryColor = 'var(--dim)';
                    let detail = isActive ? 'Currently running...' : 'Completed';
                    let tooltipExtras = '';
                    let baseLatencyMap = null, basePerBsVerdict = null;
                    const nm = stageNames[si];
                    if (nm === 'BASELINE' && !isActive) {
                        value = _baselineHero(r.roundId);
                        secondaryValue = _baselineSecondary(r.roundId);
                        tooltipExtras = _baselineTooltip(r.roundId);
                        const bRound = baselineByRound[r.roundId];
                        // v4.0 dual-hero inputs (map + verdicts). Only populate
                        // when batchSizes carries the v4 entry shape (keyed by bucket tag).
                        if (bRound && bRound.batchSizes
                            && typeof bRound.batchSizes === 'object'
                            && Object.keys(bRound.batchSizes).length > 0) {
                            baseLatencyMap = bRound.batchSizes;
                            basePerBsVerdict = bRound.perBsVerdict || null;
                        }
                        // R2+ verdict synthesis: compare cur vs prev round
                        // baseline maps when no server verdict is present.
                        if (r.roundId > 1 && !basePerBsVerdict) {
                            const prevBaseRound = baselineByRound[r.roundId - 1];
                            const prevMap = prevBaseRound?.batchSizes || null;
                            basePerBsVerdict = _synthesizeReprofileVerdict(baseLatencyMap, prevMap);
                        }
                        if (r.roundId > 1) detail = 'Re-profile after integration';
                    } else if (nm === 'MINING' && !isActive) {
                        value = _miningHero(r.roundId); secondaryValue = _miningSecondary(r.roundId);
                        if (curMineData?.component) detail = `${curMineData.component} bottleneck`;
                        tooltipExtras = _miningTooltip(r.roundId);
                    } else if (nm === 'DEBATE' && !isActive) {
                        value = _debateHero(r.roundId, true);
                        secondaryValue = _debateSecondary(r.roundId, true);
                        if (curDebData?.winners?.length) detail = _named(curDebData.winners).join(', ');
                        tooltipExtras = _debateTooltip(r.roundId, true);
                    } else if (nm === 'DEBATE' && isActive) {
                        // Show live progress even while the debate stage is active.
                        value = _debateHero(r.roundId, true);
                        secondaryValue = _debateSecondary(r.roundId, true);
                        tooltipExtras = _debateTooltip(r.roundId, true);
                        if (liveDebateRoundsCompleted === 0) {
                            detail = 'Champions debating...';
                        } else {
                            detail = `Round ${liveDebateRoundsCompleted} complete — advancing`;
                        }
                    }
                    stages.push({
                        name: nm, colIdx: stageIdx, detail, value, secondaryValue, secondaryColor,
                        tooltipExtras, active: isActive,
                        latencyMap: baseLatencyMap, perBsVerdict: basePerBsVerdict,
                        elapsed: _stageElapsed(state, stageKeys[si], r.roundId),
                        designation: `U${(r.roundId - 1) * 3 + si + 1}`,
                        miningData: nm === 'MINING' ? (curMineData || null) : null,
                        debateData: nm === 'DEBATE' ? (curDebData || null) : null,
                        debateActive: nm === 'DEBATE' && (isActive || liveDebateTeamActive),
                        debateShippedOps: nm === 'DEBATE' ? [...(campaign.shipped_optimizations || [])]
                            .map(s => typeof s === 'string' ? s : s?.op_id).filter(Boolean) : null,
                        reprofileRound: nm === 'BASELINE' && r.roundId > 1,
                        integrationDisabled: nm === 'BASELINE' && liveIntegDisabled,
                    });
                }
            }
            if (stages.length < stageNames.length && currentStageIdx >= 3) {
                while (stages.length < stageNames.length) {
                    const si = stages.length;
                    const nm = stageNames[si];
                    const isBaselineNm = nm === 'BASELINE' || nm === 'RE-PROFILE';
                    const bRoundFallback = isBaselineNm ? baselineByRound[r.roundId] : null;
                    const fallbackMap = (bRoundFallback && bRoundFallback.batchSizes
                        && typeof bRoundFallback.batchSizes === 'object'
                        && Object.keys(bRoundFallback.batchSizes).length > 0)
                            ? bRoundFallback.batchSizes : null;
                    // R2+ verdict synthesis for fallback path too.
                    let fallbackVerdict = bRoundFallback?.perBsVerdict || null;
                    if (isBaselineNm && r.roundId > 1 && !fallbackVerdict) {
                        const prevBaseRound = baselineByRound[r.roundId - 1];
                        const prevMap = prevBaseRound?.batchSizes || null;
                        fallbackVerdict = _synthesizeReprofileVerdict(fallbackMap, prevMap);
                    }
                    stages.push({
                        name: nm || 'DEBATE',
                        colIdx: stageMap[si] ?? si,
                        detail: isBaselineNm && r.roundId > 1 ? 'Re-profile after integration' : 'Completed',
                        value: isBaselineNm ? _baselineHero(r.roundId)
                            : nm === 'MINING' ? _miningHero(r.roundId) : _debateHero(r.roundId, true),
                        secondaryValue: isBaselineNm ? _baselineSecondary(r.roundId)
                            : nm === 'MINING' ? _miningSecondary(r.roundId) : _debateSecondary(r.roundId, true),
                        secondaryColor: 'var(--dim)',
                        tooltipExtras: isBaselineNm ? _baselineTooltip(r.roundId)
                            : nm === 'MINING' ? _miningTooltip(r.roundId) : _debateTooltip(r.roundId, true),
                        latencyMap: fallbackMap,
                        perBsVerdict: fallbackVerdict,
                        elapsed: _stageElapsed(state, stageKeys[si], r.roundId),
                        designation: `U${(r.roundId - 1) * 3 + si + 1}`,
                        miningData: nm === 'MINING' ? (curMineData || null) : null,
                        debateData: nm === 'DEBATE' ? (curDebData || null) : null,
                        debateActive: false,
                        debateShippedOps: nm === 'DEBATE' ? [...(campaign.shipped_optimizations || [])]
                            .map(s => typeof s === 'string' ? s : s?.op_id).filter(Boolean) : null,
                        reprofileRound: isBaselineNm && r.roundId > 1,
                        integrationDisabled: isBaselineNm && liveIntegDisabled,
                    });
                }
            }
            mockRound.stages = stages;

            const allTracks = currentTracks(state);
            const shippedOps = new Set((campaign.shipped_optimizations || [])
                .map(s => typeof s === 'string' ? s : (s && s.op_id) || null)
                .filter(Boolean));
            // Filter tracks to current round only
            const currentRoundId = campaign.current_round || 1;
            const currentRoundData = (campaign.rounds || [])
                .find(rd => rd.round_id === currentRoundId) || null;
            const curSelectedWinners = currentRoundData?.debate?.selected_winners
                || currentRoundData?.selected_candidates;
            let currentRoundTrackIds;
            if (liveSlotsOwnerRoundId !== r.roundId) {
                // Live tracks belong to a prior round (scaffolded-next-round case).
                // Leave this round's tracks empty to avoid duplicating the past row.
                currentRoundTrackIds = new Set();
            } else if (curSelectedWinners?.length) {
                currentRoundTrackIds = new Set(curSelectedWinners);
            } else {
                // Fallback: use all tracks under the current round's parallel_tracks.tracks.
                currentRoundTrackIds = new Set(Object.keys(allTracks));
            }
            const tracks = {};
            for (const [k, v] of Object.entries(allTracks)) {
                if (currentRoundTrackIds.has(k)) tracks[k] = v;
            }
            for (const [opId, t] of Object.entries(tracks)) {
                const verdict = String(t.verdict || t.status || '').toUpperCase();
                const isShipped = shippedOps.has(opId);
                let status;
                if (isShipped) status = 'shipped';
                else if (verdict === 'GATED_PASS' || verdict === 'GATED-PASS') status = 'validated';
                else if (verdict === 'GPU_BLOCKED') status = 'blocked';
                else if (verdict === 'GATING_REQUIRED') status = 'gating';
                else if (verdict === 'FAIL' || verdict === 'FAILED') status = 'failed';
                else if (verdict === 'PASS' || verdict === 'PASSED') status = 'validated';
                else status = 'active';
                const kSpeedScalar = _kernelSpeedupFromTrack(t);
                const eSpeedScalar = _e2eSpeedupScalar(t.e2e_speedup);
                const amdahlPp = _amdahlPredictionPp(t.e2e_speedup);
                const heroSpeedup = kSpeedScalar != null ? kSpeedScalar : (eSpeedScalar != null ? eSpeedScalar : 0);
                mockRound.tracks.push({
                    name: opId,
                    displayName: opNames.get(opId) || null,
                    status,
                    speedup: heroSpeedup ? heroSpeedup.toFixed(2) : '0',
                    detail: t.fail_reason || t.failure_reason || opNames.get(opId) || opId.replace(/_/g, ' '),
                    failReason: t.failure_reason || t.fail_reason || null,
                    lossy: !!(t.classification && t.classification.toLowerCase() === 'lossy'),
                    kernelSpeedup: kSpeedScalar,
                    e2eSpeedup: eSpeedScalar,
                    amdahlPredictionPp: amdahlPp,
                    perBsVerdict: t.per_bs_verdict || null,
                    e2eLatencyOpt: t.e2e_latency_opt || null,
                    commitSha: t.commit_sha || null,
                    subStage: t.current_stage || t.sub_stage || null,
                });
            }

            const integ = liveSlotsOwnerRoundId === r.roundId ? currentIntegration(state) : {};
            // Schema-aligned integration.status enum (10 values):
            //   pending, in_progress, validated, single_pass, combined, gated_pass,
            //   completed, exhausted, failed, skipped.
            // Legacy alias 'complete' (no -d) is folded into 'completed'.
            const integStatusRaw = String(integ.status || '').toLowerCase();
            const integStatus = integStatusRaw === 'complete' ? 'completed' : integStatusRaw;
            // Success-flavored terminals — integration produced a usable result.
            const INTEG_SUCCESS = new Set(['completed', 'validated', 'single_pass', 'combined', 'gated_pass']);
            // Stop-worthy terminals — integration done but no forward progress.
            const INTEG_STOP    = new Set(['failed', 'exhausted', 'skipped']);
            const integSuccess  = INTEG_SUCCESS.has(integStatus);
            const integRunning  = integStatus === 'in_progress' || integStatus === 'running';
            const integStopped  = INTEG_STOP.has(integStatus);
            const integTerminal = integSuccess || integStopped;
            if (integTerminal || integRunning) {
                const prevRound = rounds[ri - 1];
                const prevSpeedup = prevRound?.speedupAfter || 1.0;
                const cumulative = campaign.cumulative_e2e_speedup ?? 1;
                // Delta in percentage points — prefer v4.0 client-side computation
                // from the map-based baseline + integration latency fields. Fall
                // back to the legacy v3 scalar on combined_e2e_result only when
                // the maps aren't present (transitional states).
                const baseMap = r.roundData?.baseline?.e2e_latency;
                const combMap = integ.e2e_latency_combined;
                let deltaPp = _computeDeltaPpFromMaps(baseMap, combMap);
                if (deltaPp == null) {
                    const serverDeltaPp = integ.combined_e2e_result?.delta_pp;
                    deltaPp = typeof serverDeltaPp === 'number' && Number.isFinite(serverDeltaPp)
                        ? serverDeltaPp
                        : (cumulative - prevSpeedup) * 100;
                }
                // v4.0 this-round speedup: previous baseline / this integration
                // at the first exact shared workload tag. Round 1 has no prev.
                let thisRoundSpeedup = null;
                if (combMap && typeof combMap === 'object') {
                    const prevBaseMap = prevRound?.roundData?.baseline?.e2e_latency
                        || prevRound?.roundData?.integration?.e2e_latency_combined
                        || baseMap  // round 1 fallback: current baseline
                        || null;
                    thisRoundSpeedup = _computeSpeedupFromMaps(prevBaseMap, combMap);
                }
                if (thisRoundSpeedup == null && r.roundId === 1) thisRoundSpeedup = 1.0;
                const integElapsed = _stageElapsed(state, '6_integration', r.roundId);
                const passing = (integ.passing_candidates || []).length;
                const failedN = (integ.failed_candidates || []).length;
                const total = passing + failedN;
                const progressText = integRunning
                    ? (total > 0 ? `${passing}/${total} MERGING` : 'MERGING')
                    : integStatus === 'failed'
                        ? 'MERGE FAILED'
                        : integStatus === 'exhausted'
                            ? 'EXHAUSTED'
                            : integStatus === 'skipped'
                                ? 'SKIPPED'
                                : integStatus === 'gated_pass'
                                    ? 'GATED PASS'
                                    : integStatus === 'single_pass'
                                        ? 'SINGLE PASS'
                                        : integStatus === 'combined'
                                            ? 'COMBINED'
                                            : integStatus === 'validated'
                                                ? 'VALIDATED'
                                                : null;
                mockRound.integration = {
                    speedup: cumulative.toFixed(2),
                    // Show the positive delta only for success-terminals — for
                    // in_progress / failed / exhausted / skipped we want the secondary
                    // slot to carry the status text instead.
                    deltaPp: integSuccess && deltaPp > 0 ? deltaPp : null,
                    thisRoundSpeedup,
                    cumulativeSpeedup: cumulative,
                    perBsVerdict: integ.per_bs_verdict || null,
                    combinedMap: combMap || null,
                    mergedOps: integ.passing_candidates || [],
                    failedOps: integ.failed_candidates || [],
                    status: integStatus,  // schema enum or 'running'
                    progressText,
                    elapsed: integElapsed,
                };
            }

            // ── Current-round eval diamond ──
            // Always render a diamond on the current row for visual consistency.
            // Variant the text by stage:
            //   stages 1-5 (baseline → parallel_tracks)    → ACTIVE (pulsing, no text)
            //   stage 6 (integration)                       → ACTIVE until terminal; GO/STOP once resolved
            //   stage 7 (campaign_eval/7b_report)            → GO unless campaign.status ∈ {campaign_complete, campaign_exhausted} (then STOP)
            // Integration verdicts are read from the current round's integration
            // ONLY when current_stage==='6_integration' (for earlier stages, the
            // current round's integration object is still pending).
            // Diamonds render ONLY at 6_integration (terminal verdict), 7_campaign_eval,
            // or 7b_report. Stages 1-5 and in-progress integration get no diamond —
            // the trace still runs to EVAL_CX but terminates with a small tick.
            const passingVerdicts = new Set(['PASS', 'PASSED', 'GATED_PASS', 'GATED-PASS']);
            const sv = currentStage(state);
            const cs = (campaign.status || '').toLowerCase();
            if (sv === '7_campaign_eval' || sv === '7b_report') {
                const campaignDone = cs === 'campaign_complete' || cs === 'campaign_exhausted';
                mockRound.eval = campaignDone ? 'STOP' : 'CONTINUE';
                mockRound.evalNote = campaignDone
                    ? (cs === 'campaign_exhausted'
                        ? 'Campaign exhausted — no further rounds'
                        : 'Campaign converged — no further rounds')
                    : 'Round complete — advancing to next round';
            } else if (sv === '6_integration') {
                // Re-read THIS round's integration — not the stale campaign.integration field,
                // which for an in-flight round 2 might still carry round 1 data.
                if (integSuccess) {
                    const passingCount = (integ.passing_candidates || []).length;
                    const trackPassed = Object.values(allTracks).some(t => passingVerdicts.has(String(t.verdict || t.status || '').toUpperCase()));
                    mockRound.eval = (passingCount > 0 || trackPassed) ? 'CONTINUE' : 'STOP';
                    mockRound.evalNote = mockRound.eval === 'CONTINUE'
                        ? 'Integration passed — advancing to next round'
                        : 'No ops passed integration — campaign halt';
                } else if (integStopped) {
                    mockRound.eval = 'STOP';
                    mockRound.evalNote = integStatus === 'exhausted'
                        ? 'Integration exhausted — campaign halt'
                        : integStatus === 'skipped'
                            ? 'Integration skipped — campaign halt'
                            : 'Integration failed — campaign halt';
                } else {
                    // Integration in-flight — no diamond yet.
                    mockRound.eval = null;
                    mockRound.evalNote = null;
                }
            } else {
                // Stages 1-5 — pre-decision. No diamond.
                mockRound.eval = null;
                mockRound.evalNote = null;
            }
        }

        // ── Attach sidecar label-kinds to this round ──
        const allRationales = state._catalog?.rationales || [];
        mockRound.rationales = allRationales.filter(rat =>
            rat.round === r.roundId || (rat.round == null && r.roundId === 1)
        );
        const allResolutions = state._catalog?.resolutions || [];
        // Resolutions now carry round metadata when available; fall back to
        // pinning unlabeled resolutions onto the current round's integration.
        const _roundId = r.roundId;
        mockRound.resolutions = allResolutions.filter(res => {
            if (res.round != null) return res.round === _roundId;
            return isCurrent;
        });
        if (mockRound.resolutions.length > 0 && !mockRound.integration) {
            // Synthesize a minimal integration chip so the badge has a host
            mockRound.integration = {
                speedup: (campaign.cumulative_e2e_speedup ?? 1).toFixed(2),
                deltaPp: null,
                mergedOps: [],
                elapsed: _stageElapsed(state, '6_integration', r.roundId),
                synthetic: true,
            };
        }

        return mockRound;
    });
}

// ── SVG helpers (traces only) ─────────────────────────────────────────────────

function _svgEl(tag, attrs, parent) {
    const e = document.createElementNS(CB.SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
}

function buildPathD(pts) {
    if (pts.length < 2) return '';
    let d = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 1; i < pts.length; i++) d += ` L${pts[i][0]},${pts[i][1]}`;
    return d;
}

function estimatePathLen(pts) {
    let len = 0;
    for (let i = 1; i < pts.length; i++) {
        len += Math.abs(pts[i][0] - pts[i-1][0]) + Math.abs(pts[i][1] - pts[i-1][1]);
    }
    return len;
}

function drawTrace(svgLayer, points, color, animate, animDelay, filterAttr) {
    const d = buildPathD(points);
    const len = estimatePathLen(points);
    // Dim base rail (static, visible before draw-in)
    _svgEl('path', { d, fill: 'none', stroke: color, 'stroke-width': L2.TRACE_WEIGHT, opacity: 0.08, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'pointer-events': 'none' }, svgLayer);
    // Main animated trace
    const trace = _svgEl('path', { d, fill: 'none', stroke: color, 'stroke-width': L2.TRACE_WEIGHT, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-dasharray': len, 'stroke-dashoffset': len, 'pointer-events': 'none' }, svgLayer);
    const drawDuration = Math.max(0.4, len / 500);
    trace.style.cssText = `--tl:${len};animation:cb2DrawTrace ${drawDuration}s ease-out ${animDelay}s forwards;`;
    if (filterAttr) trace.setAttribute('filter', filterAttr);
    // Glow underlay
    const glowTrace = _svgEl('path', { d, fill: 'none', stroke: color, 'stroke-width': 6, 'stroke-linecap': 'round', opacity: 0.12, 'stroke-dasharray': len, 'stroke-dashoffset': len, 'pointer-events': 'none' }, svgLayer);
    glowTrace.style.cssText = `--tl:${len};animation:cb2DrawTrace ${drawDuration}s ease-out ${animDelay}s forwards;`;
}

// Small vertical tick used as trace terminus when no decision diamond is rendered.
// Signals "trace continues to the eval column but no verdict yet" without
// implying the decision has been made.
function drawTerminusTick(svgLayer, x, y, color, delay) {
    const tickH = 10;
    const tick = _svgEl('line', { x1: x, y1: y - tickH / 2, x2: x, y2: y + tickH / 2,
        stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round',
        opacity: 0, 'pointer-events': 'none' }, svgLayer);
    setTimeout(() => { tick.style.transition = 'opacity 0.3s'; tick.style.opacity = '0.55'; }, delay * 1000);
}

function drawVia(svgLayer, x, y, color, delay) {
    const outerVia = _svgEl('circle', { cx: x, cy: y, r: L2.VIA_R + 2, fill: 'none', stroke: `${color}30`, 'stroke-width': 1, opacity: 0, 'pointer-events': 'none' }, svgLayer);
    const via = _svgEl('circle', { cx: x, cy: y, r: L2.VIA_R, fill: '#05050a', stroke: color, 'stroke-width': 2, opacity: 0, 'pointer-events': 'none' }, svgLayer);
    const dot = _svgEl('circle', { cx: x, cy: y, r: 2, fill: color, opacity: 0, 'pointer-events': 'none' }, svgLayer);
    setTimeout(() => {
        [outerVia, via, dot].forEach(e => { e.style.transition = 'opacity 0.3s'; e.style.opacity = '1'; });
    }, delay * 1000);
}

// Short local-time string for audit-gate tooltip rows ("14:32:05" or "—").
function _auditShortTime(iso) {
    if (!iso) return null;
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleTimeString();
}

// T_AUDIT_S{1,2,45,67} designator label per gate key, matched to the
// canonical stage names used in the auditor dispatch prompt.
const AUDIT_GATE_LABEL = {
    stage_1: 'T_AUDIT_S1',
    stage_2: 'T_AUDIT_S2',
    stage_45: 'T_AUDIT_S45',
    stage_67: 'T_AUDIT_S67',
};

// Draws an audit-gate "fuse hex" — a ~16px flat-top hexagon sitting on the
// trace where a plain via would render today, plus a micro-designator label
// underneath. gate = { state, started_at, passed_at, cycle, verdict_file, note }
// (see auditGateStates). bypass/pending gates render as a plain via with no
// interactivity; running/passed/escalated render the hex and are clickable.
function drawAuditHex(canvas, svgLayer, x, y, gate, gateKey, roundId, delay, onAuditOpen) {
    const st = (gate && gate.state) || 'bypass';
    // A gate with no dispatch yet keeps the plain via the board has always
    // drawn at this junction. A hollow hex would delete an opaque trace marker
    // and replace it with a near-invisible shape carrying no tooltip or click.
    if (st === 'bypass' || st === 'pending') {
        drawVia(svgLayer, x, y, '#00ffb2', delay);
        return;
    }

    const COLORS = {
        running: '#ffaa00',
        passed: '#00ffb2',
        escalated: '#ff3355',
    };
    const color = COLORS[st] || COLORS.running;
    const filterAttr = st === 'passed' ? 'url(#cb2-gM)'
        : st === 'escalated' ? 'url(#cb2-gR)'
        : st === 'running' ? 'url(#cb2-gA)'
        : null;

    const size = 16;
    // Flat-top hexagon clip-path (percentages of the element box).
    const clip = 'polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%)';

    const wrap = _h('div', 'cb2-audit-hex-wrap');
    wrap.style.cssText = `position:absolute;left:${x - size / 2}px;top:${y - size / 2}px;width:${size}px;height:${size}px;z-index:3;opacity:0;transition:opacity 0.3s;`;

    const hex = _h('div', `cb2-audit-hex cb2-audit-hex--${st}`);
    hex.style.cssText = `width:100%;height:100%;clip-path:${clip};background:${color};border:1.5px solid ${color};box-sizing:border-box;` + (filterAttr ? `filter:drop-shadow(0 0 3px ${color});` : '');
    wrap.appendChild(hex);

    if (st === 'passed') {
        const check = _h('div', 'cb2-audit-hex-check');
        check.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:8px;color:#05050a;pointer-events:none;';
        check.textContent = '✓';
        wrap.appendChild(check);
    }

    const label = _h('div', 'cb2-audit-hex-label');
    label.style.cssText = `position:absolute;left:50%;top:${size + 3}px;transform:translateX(-50%);white-space:nowrap;font-family:'Share Tech Mono',monospace;font-size:7px;letter-spacing:0.5px;color:${color};pointer-events:none;`;
    label.textContent = 'AUD ' + gateKey.replace('stage_', 'S').toUpperCase();
    wrap.appendChild(label);

    // Every state that reaches here (running/passed/escalated) is interactive;
    // pending and bypass returned above as a plain via.
    {
        wrap.style.cursor = 'pointer';
        wrap.tabIndex = 0;
        wrap.setAttribute('role', 'button');
        wrap.setAttribute('aria-label', AUDIT_GATE_LABEL[gateKey] + ' round ' + roundId);
    }

    if (_cb2Tip) {
        const gateLabel = AUDIT_GATE_LABEL[gateKey] || gateKey.toUpperCase();
        const titleLabel = `AUDIT GATE · ${_esc(gateLabel)} · R${roundId}`;
        const metaParts = [];
        if (gate.cycle != null) metaParts.push('CYCLE ' + gate.cycle);
        metaParts.push(st.toUpperCase());
        const meta = `<span class="meta">${_esc(metaParts.join(' · '))}</span>`;
        const show = (e) => {
            let body = `<div class="cb2-tt-title"><span>${titleLabel}</span>${meta}</div>`;
            const startedStr = _auditShortTime(gate.started_at);
            const passedStr = _auditShortTime(gate.passed_at);
            if (startedStr) body += `Started: <span class="cb2-tt-val">${_esc(startedStr)}</span><br>`;
            if (passedStr) body += `Passed: <span class="cb2-tt-val">${_esc(passedStr)}</span><br>`;
            if (gate.verdict_file) body += `<span class="cb2-tt-dim">Verdict: ${_esc(gate.verdict_file)}</span><br>`;
            if (gate.note) body += `<span class="cb2-tt-dim">${_esc(gate.note)}</span>`;
            _cb2Tip.show(e, null, body, null);
        };
        wrap.addEventListener('mouseenter', show);
        wrap.addEventListener('mouseleave', () => _cb2Tip.hide());
        // A FocusEvent has no clientX/clientY, so positionTip would compute NaN
        // and the tooltip would land nowhere near the focused hex. Synthesize
        // the coordinates from the element's own box for the keyboard path.
        wrap.addEventListener('focusin', () => {
            const box = typeof wrap.getBoundingClientRect === 'function'
                ? wrap.getBoundingClientRect() : null;
            show(box
                ? { clientX: box.left + box.width / 2, clientY: box.top + box.height / 2 }
                : { clientX: 0, clientY: 0 });
        });
        wrap.addEventListener('focusout', () => _cb2Tip.hide());
    }

    if (typeof onAuditOpen === 'function') {
        const openIt = () => onAuditOpen(roundId, gateKey, gate);
        wrap.addEventListener('click', openIt);
        wrap.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openIt(); }
        });
    }

    canvas.appendChild(wrap);
    setTimeout(() => { wrap.style.opacity = '1'; }, delay * 1000);
}

// ── SVG Defs (glow filters only — no PCB patterns) ─────────────────────────

function _buildSvgDefs() {
    const defs = document.createElementNS(CB.SVG_NS, 'defs');
    defs.innerHTML = `
      <filter id="cb2-gM" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="4" result="b"/><feFlood flood-color="#00ffb2" flood-opacity="0.35" result="c"/>
        <feComposite in="c" in2="b" operator="in" result="g"/><feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <filter id="cb2-gC" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="4" result="b"/><feFlood flood-color="#00f3ff" flood-opacity="0.35" result="c"/>
        <feComposite in="c" in2="b" operator="in" result="g"/><feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <filter id="cb2-gR" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="4" result="b"/><feFlood flood-color="#ff3355" flood-opacity="0.35" result="c"/>
        <feComposite in="c" in2="b" operator="in" result="g"/><feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <filter id="cb2-gA" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="4" result="b"/><feFlood flood-color="#ffaa00" flood-opacity="0.35" result="c"/>
        <feComposite in="c" in2="b" operator="in" result="g"/><feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <marker id="cb2-arrowMint" markerWidth="10" markerHeight="8" refX="10" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8" fill="none" stroke="#00ffb2" stroke-width="1.5"/>
      </marker>
      <marker id="cb2-arrowCyan" markerWidth="10" markerHeight="8" refX="10" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8" fill="none" stroke="#00f3ff" stroke-width="1.5"/>
      </marker>
    `;
    return defs;
}

// ── HTML chip builders ────────────────────────────────────────────────────────

function _h(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html) e.innerHTML = html;
    return e;
}

function makeStageChip(canvas, x, y, stage, roundActive, delay, roundId, colIdx, onClick, reachable) {
    const isActive = stage.active;
    const isReachable = typeof reachable === 'function' ? reachable(roundId, colIdx) : true;
    // Baseline chips with a map-based latency field activate the dual-hero
    // path and a verdict-driven state modifier. Both R1 BASELINE and R2+
    // RE-PROFILE go through this path — R2+ is the same chip with a
    // relabelled header and verdict colors driven by latency comparison
    // against prev round's baseline. The chip-test-harness also constructs
    // RE-PROFILE chips directly to exercise verdict CSS variants in
    // isolation, so the literal name is honored independently of the
    // round mapper's `reprofileRound` flag.
    const isBaselineKind = !isActive && (stage.name === 'BASELINE' || stage.name === 'RE-PROFILE');
    // Re-profile presentation is triggered either by the round mapper
    // (`reprofileRound: true` for R2+) or by an explicit RE-PROFILE name
    // from the chip-test-harness, which constructs chips directly without
    // going through the round mapper. R1 BASELINE keeps its original label.
    const isReprofile = isBaselineKind && (stage.reprofileRound || stage.name === 'RE-PROFILE');
    const integDisabled = !!(stage.integrationDisabled && isReprofile);
    const isMiningKind   = stage.name === 'MINING';
    const isDebateKind   = stage.name === 'DEBATE';
    const bsRange = isBaselineKind ? _bsRange(stage.latencyMap) : null;
    const verdictAgg = isBaselineKind ? _verdictAggregate(stage.perBsVerdict) : 'NONE';
    let stateCls;
    if (isMiningKind) {
        // MINING chips always use the violet stage style — no active/pending
        // state variants exist today (mining is a single one-shot stage).
        stateCls = 'cb2-stage-mining' + (isActive ? ' cb2-stage-active' : '');
    } else if (isDebateKind) {
        // DEBATE chips use the cyan stage style; active state is layered on
        // top so the in-progress glow still renders over the debate envelope.
        stateCls = 'cb2-stage-debate' + (isActive ? ' cb2-stage-active' : '');
    } else if (isActive) {
        stateCls = 'cb2-stage-active';
    } else if (integDisabled) {
        // Integration failed/skipped: re-profile is just a restatement of
        // the prior round's baseline. Render neutral done envelope; the
        // cb2-chip-disabled modifier added below drains color/glow.
        stateCls = 'cb2-stage-done';
    } else if (isBaselineKind && verdictAgg === 'RED') {
        stateCls = 'cb2-stage-done cb2-stage-reprofile cb2-stage-fail';
    } else if (isBaselineKind && verdictAgg === 'AMBER') {
        stateCls = 'cb2-stage-done cb2-stage-reprofile';
    } else {
        stateCls = 'cb2-stage-done';
    }
    const pastCls = !roundActive ? ' cb2-row-past' : '';
    // Reachability OR integrationDisabled both gray the chip out — same
    // visual treatment, different semantic causes.
    const disabledCls = (!isReachable || integDisabled) ? ' cb2-chip-disabled' : '';
    const chip = _h('div', `cb2-hud ${stateCls}${isActive ? '' : ' cb2-row-anim'}${pastCls}${disabledCls}`);
    chip.style.cssText = `left:${x}px;top:${y}px;width:${L2.STAGE_W}px;height:${L2.STAGE_H}px;${isActive ? '' : 'animation-delay:' + delay + 's;'}`;
    chip.setAttribute('data-stage-name', stage.name || '');
    if (isReprofile) chip.setAttribute('data-reprofile', '1');
    if (integDisabled) chip.setAttribute('data-integration-disabled', '1');

    // Label format per approved mockup: `BASELINE \u00B7 R{n}` for R1, or
    // `RE-PROFILE \u00B7 R{n}` for R2+ (same chip, same data pipeline \u2014
    // only the label + verdict-color treatment changes). Prepend the
    // \u27F2 glyph when integration failed/skipped to signal "this is a
    // carry-forward of prev round's baseline, not a fresh measurement".
    // Designation (U1/U4/U7) is dropped on baseline/reprofile chips per the
    // approved mockup.
    const reprofileGlyph = integDisabled ? '\u27F2 ' : '';
    const displayName = isReprofile ? 'RE-PROFILE' : stage.name;
    const baselineLabel = isBaselineKind
        ? `${reprofileGlyph}${displayName} \u00B7 R${roundId}`
        : _esc(stage.name);

    // Dual-hero path: first/last deterministically sorted workload buckets.
    if (isBaselineKind && bsRange && bsRange.count >= 2 && stage.latencyMap) {
        const firstBucket = bsRange.first;
        const lastBucket = bsRange.last;
        const bestAvg = stage.latencyMap[firstBucket.tag]?.avg;
        const worstAvg = stage.latencyMap[lastBucket.tag]?.avg;
        const bestStr = typeof bestAvg === 'number' ? _fmtLatency(bestAvg) : '\u2014';
        const worstStr = typeof worstAvg === 'number' ? _fmtLatency(worstAvg) : '\u2014';
        const rangeLabel = bsRange.heterogeneous
            ? `${firstBucket.compactLabel} \u2192 ${lastBucket.compactLabel} \u00b7 ${bsRange.count} buckets`
            : `BS ${bsRange.min} \u2014 ${bsRange.max} \u00B7 ${bsRange.count} sizes`;
        chip.innerHTML = `<div class="cb2-ctl"></div><div class="cb2-cbr"></div>
          <div class="cb2-label">${baselineLabel}</div>
          <div class="cb2-dual">
            <div class="cb2-best">${_esc(bestStr)}</div>
            <div class="cb2-sep">\u2192</div>
            <div class="cb2-worst">${_esc(worstStr)}</div>
          </div>
          <div class="cb2-range">${_esc(rangeLabel)}</div>
          <div class="cb2-metric-tag">\u00B7 AVG \u00B7</div>`;
    } else if (isBaselineKind && bsRange && bsRange.count === 1 && stage.latencyMap) {
        // Single-bucket fallback: centered hero + full bucket identity.
        // `cb2-hero-single` modifier centers the hero horizontally + adds
        // vertical breathing room (mockup line 403).
        const bucket = bsRange.first;
        const singleAvg = stage.latencyMap[bucket.tag]?.avg;
        const singleStr = typeof singleAvg === 'number' ? _fmtLatency(singleAvg) : (stage.value || '\u2014');
        const rangeLabel = bucket.label;
        chip.innerHTML = `<div class="cb2-ctl"></div><div class="cb2-cbr"></div>
          <div class="cb2-label">${baselineLabel}</div>
          <div class="cb2-hero cb2-hero-single">${_esc(singleStr)}</div>
          <div class="cb2-range">${_esc(rangeLabel)}</div>
          <div class="cb2-metric-tag">\u00B7 AVG \u00B7</div>`;
    } else if (isMiningKind) {
        // MINING chip \u2014 Option A layout (mockup \u00A71).
        //   label        : MINING \u00B7 R{n}
        //   pct-hero     : {pct}%      (Orbitron 28px violet)
        //   component    : TOP \u00B7 {name}
        //   ceiling      : ceiling: {x}\u00D7
        // Graceful degradation: if pct/component/ceiling absent, each row
        // collapses independently \u2014 the chip still renders with a consistent
        // vertical rhythm via margin: auto on the ceiling footer.
        const md = stage.miningData || {};
        // Prefer the explicit override (state.bottleneck_mining.top_bottleneck_share_pct)
        // passed down from the orchestrator, falling back to the catalog value.
        const rawPct = (typeof stage.miningPctOverride === 'number' ? stage.miningPctOverride : null)
            ?? (typeof md.pct === 'number' ? md.pct : null);
        const pctStr = (rawPct != null && Number.isFinite(rawPct))
            ? String(Math.round(rawPct)) : '\u2014';
        const component = md.component || stage.detail || '';
        // Truncate to fit ~24 chars on the component line (200px chip, mono 11px \u2248 6.3px/char).
        const compDisplay = component.length > 22 ? component.slice(0, 20) + '\u2026' : component;
        const ceilingStr = (md.amdahlCeiling && Number.isFinite(md.amdahlCeiling))
            ? md.amdahlCeiling.toFixed(1) + '\u00D7' : '\u2014';
        const miningLabel = `MINING \u00B7 R${roundId}`;
        chip.innerHTML = `<div class="cb2-ctl"></div><div class="cb2-cbr"></div>
          <div class="cb2-label">${miningLabel}</div>
          <div class="cb2-pct-hero">${_esc(pctStr)}<span class="unit">%</span></div>
          ${component ? `<div class="cb2-component-line"><span class="cb2-comp-tag">TOP \u00B7</span>${_esc(compDisplay)}</div>` : ''}
          <div class="cb2-ceiling">ceiling:<span class="cb2-ceiling-val">${_esc(ceilingStr)}</span></div>`;
    } else if (isDebateKind) {
        // DEBATE chip \u2014 Option A layout (mockup \u00A73).
        //   label        : DEBATE \u00B7 R{n}
        //   winner-label : \u25C6 WINNER(S) | \u25C6 CHAMPIONS | \u25C6 DEBATING\u2026
        //   winner-hero  : {winner names, truncated} [+N more]
        //   right-stack  : counters (champions \u00B7 rounds)
        // Active-state fallback: when debate is running with no winners yet,
        // the hero shows the champion count + the cyan italic "debating\u2026" variant.
        const dd = stage.debateData || {};
        const debateLabel = `DEBATE \u00B7 R${roundId}`;
        const shippedOps = Array.isArray(stage.debateShippedOps) ? stage.debateShippedOps : [];
        const winners = Array.isArray(dd.winners) && dd.winners.length
            ? dd.winners
            : (shippedOps.length ? shippedOps : null);
        const champions = dd.championsCount || null;
        const rounds = dd.summaryRoundsCompleted || null;
        // Chip body branches:
        //   (1) Debate still running, no winners picked \u2192 hero = "debating\u2026"
        //   (2) Winners known \u2192 hero = first 2 names + "+N more" pill when overflow
        //   (3) Champions but no winners \u2192 hero = "{N} CHAMPIONS"
        let winnerLabelText = '\u25C6 WINNER';
        let winnerHeroHtml = '';
        if (isActive && (!winners || !winners.length)) {
            winnerLabelText = '\u25C6 DEBATE';
            winnerHeroHtml = `<div class="cb2-winner-hero">debating\u2026</div>`;
        } else if (winners && winners.length > 0) {
            winnerLabelText = winners.length > 1 ? '\u25C6 WINNERS' : '\u25C6 WINNER';
            // Truncate each winner name individually so the `+N` overflow pill
            // always fits. Budget: ~20 chars of op names total (mono 14px \u2248
            // 8.5px/char \u00D7 20 = 170px, leaving room for the pill and padding).
            // Single winner: 22-char budget (whole line). Two winners: 9 chars
            // each so the `, ` separator + both names + pill fits.
            const shown = winners.slice(0, 2);
            const overflow = winners.length - shown.length;
            const perNameBudget = shown.length === 1 ? 22 : 9;
            const truncName = (n) => n.length > perNameBudget ? n.slice(0, perNameBudget - 1) + '\u2026' : n;
            const heroText = shown.map(truncName).join(', ');
            const morePill = overflow > 0
                ? `<span class="cb2-more-pill">+${overflow}</span>` : '';
            winnerHeroHtml = `<div class="cb2-winner-hero">${_esc(heroText)}${morePill}</div>`;
        } else if (champions) {
            winnerLabelText = '\u25C6 CHAMPIONS';
            winnerHeroHtml = `<div class="cb2-winner-hero">${champions} active</div>`;
        } else {
            winnerHeroHtml = `<div class="cb2-winner-hero" style="color:rgba(232,232,240,0.25)">\u2014</div>`;
        }
        // Right-stack counter cells. Always show champions when known; show
        // rounds when known. Gracefully omit either column if data missing.
        let counterHtml = '';
        if (champions != null) {
            counterHtml += `<div class="cb2-counter"><span class="cb2-counter-v">${champions}</span><span class="cb2-counter-k">champions</span></div>`;
        }
        if (rounds != null) {
            counterHtml += `<div class="cb2-counter"><span class="cb2-counter-v">${rounds}</span><span class="cb2-counter-k">rounds</span></div>`;
        }
        chip.innerHTML = `<div class="cb2-ctl"></div><div class="cb2-cbr"></div>
          <div class="cb2-label">${debateLabel}</div>
          <div class="cb2-winner-label">${_esc(winnerLabelText)}</div>
          ${winnerHeroHtml}
          <div class="cb2-right-stack">${counterHtml}</div>`;
    } else {
        // Legacy single-hero path \u2014 preserved for every non-baseline stage
        // that isn't MINING or DEBATE (e.g. transitional / unknown stages)
        // and for any baseline without a latency map.
        const heroText = stage.value || '';
        const heroCompact = heroText.length > 8 ? ' cb2-hero-compact' : '';
        chip.innerHTML = `<div class="cb2-ctl"></div><div class="cb2-cbr"></div>
          <div class="cb2-desig">${_esc(stage.designation || '')}</div>
          <div class="cb2-label">${isActive ? '\u2B22 ' : ''}${_esc(stage.name)}</div>
          ${stage.value ? `<div class="cb2-hero${heroCompact}">${_esc(stage.value)}</div>` : `<div class="cb2-hero" style="color:rgba(232,232,240,0.08)">\u2014</div>`}
          ${stage.secondaryValue ? `<div class="cb2-sec">${_esc(stage.secondaryValue)}</div>` : ''}`;
    }

    // Tooltip wiring. Four distinct paths:
    //   1. BASELINE with latency map  → rich percentile tooltip
    //   2. MINING                     → §5 profiling breakdown
    //   3. DEBATE                     → D1 scoreboard
    //   4. Other (transitional/unknown) → legacy simple tooltip
    // DEBATE tooltip re-enabled: the D1 scoreboard complements the rationale
    // card reveal group (which stays click/focus-driven) rather than duplicating it.
    if (_cb2Tip) {
        chip.addEventListener('mouseenter', e => {
            if (integDisabled) {
                // R2+ baseline (re-profile) chip but the prior round's
                // integration failed/skipped \u2014 there is no fresh
                // measurement to re-profile against. Surface that fact
                // directly rather than rendering the rich latency table
                // (which would imply a real comparison was run).
                const titleLabel = 'RE-PROFILE \u00b7 R' + roundId;
                const titleHtml = `<div class="cb2-tt-title"><span>${_esc(titleLabel)}</span></div>`;
                const body = `<span class="cb2-tt-dim">Reusing previous round's baseline \u2014 integration did not succeed.</span>`
                    + (stage.value ? `<br>Latency: <span class="cb2-tt-val">${_esc(stage.value)}</span>` : '')
                    + (stage.secondaryValue ? `<br><span class="cb2-tt-dim">${_esc(stage.secondaryValue)}</span>` : '');
                _cb2Tip.show(e, null, titleHtml + body, null);
            } else if (isBaselineKind && stage.latencyMap && Object.keys(stage.latencyMap).length) {
                // Rich tooltip for the baseline / re-profile latency map.
                // R1 reads "BASELINE \u00b7 R1"; R2+ reads "RE-PROFILE \u00b7 RN".
                const titleLabel = (isReprofile ? 'RE-PROFILE' : stage.name) + ' \u00b7 R' + roundId;
                const titleHtml = `<div class="cb2-tt-title"><span>${_esc(titleLabel)}</span></div>`;
                const buckets = _bucketRecords(stage.latencyMap);
                const bucketCount = buckets.length;
                const primaryBucket = buckets[0] || null;
                const summary = [];
                if (primaryBucket && stage.latencyMap[primaryBucket.tag]?.avg != null) {
                    summary.push({ label: primaryBucket.label, value: _fmtLatency(stage.latencyMap[primaryBucket.tag].avg), cls: 'mint' });
                }
                if (bucketCount > 1) summary.push({ label: 'BUCKETS', value: String(bucketCount), cls: 'cyan' });
                const noteText = 'chip shows {AVG} \u00b7 table shows full percentiles';
                const hasVerdicts = verdictAgg !== 'NONE';
                const sourceText = hasVerdicts && roundId > 1 ? 'vs R' + (roundId - 1) + ' baseline' : null;
                const bodyHtml = _buildRichTooltipBody({
                    summary: summary.length ? summary : null,
                    note: noteText,
                    latencyMap: stage.latencyMap,
                    perBsVerdict: stage.perBsVerdict,
                    legend: hasVerdicts,
                    source: sourceText,
                });
                _cb2Tip.show(e, null, titleHtml + bodyHtml, null);
            } else if (isMiningKind) {
                // \u00a75 PROFILING BREAKDOWN \u2014 mining chip hover.
                const body = _buildMiningTooltipBody({
                    roundId,
                    miningData: stage.miningData || {},
                    metaSuffix: null,
                });
                _cb2Tip.show(e, null, body, ['tt-mining']);
            } else if (isDebateKind) {
                // D1 SCOREBOARD \u2014 debate chip hover. Rationale-reveal card group
                // still opens via click/focus; the tooltip shows the ranked
                // leaderboard instead of duplicating the chip-face counters.
                const body = _buildDebateTooltipBody({
                    roundId,
                    debateData: stage.debateData || {},
                    metaSuffix: null,
                });
                _cb2Tip.show(e, null, body, ['tt-debate', 'd1']);
            } else {
                // Legacy simple tooltip for transitional / unknown stages.
                let body = `<span class="cb2-tt-dim">${_esc(stage.detail || '')}</span>`;
                if (stage.value) body += `<br><span class="cb2-tt-val">${_esc(stage.value)}</span>`;
                if (stage.tooltipExtras) body += `<br>${stage.tooltipExtras}`;
                if (stage.elapsed) body += `<br><span class="cb2-tt-dim">Duration: ${_esc(stage.elapsed)}</span>`;
                if (!isReachable) body += `<br><span style="color:rgba(255,170,0,0.85)">Not yet run in round ${roundId}</span>`;
                body += `<br><span class="cb2-tt-dim" style="font-size:9px">${_esc(stage.designation || '')}</span>`;
                _cb2Tip.show(e, _esc(stage.name), body, null);
            }
        });
        chip.addEventListener('mouseleave', () => _cb2Tip.hide());
    }
    if (onClick && isReachable) chip.addEventListener('click', () => onClick(roundId, colIdx, true));

    canvas.appendChild(chip);
    return { cx: x + L2.STAGE_W, cy: y + L2.STAGE_H / 2, left: x, top: y, el: chip };
}

function makeTrackChip(canvas, x, y, track, delay, roundId, onClick, roundActive) {
    const cls = `cb2-track cb2-track-${track.status}`;
    const isActiveTrack = track.status === 'active';
    const pastCls = roundActive === false ? ' cb2-row-past' : '';
    const chip = _h('div', `cb2-hud ${cls}${isActiveTrack ? '' : ' cb2-row-anim'}${pastCls}`);
    // validated/gated are rendered via JS (CSS is owned by the rationale-cards branch);
    // paint the accent + background inline so the chip reads as mint/amber without a CSS rule.
    let extraStyle = '';
    if (track.status === 'validated') {
        extraStyle = 'background:linear-gradient(rgba(0,255,178,0.02),rgba(0,255,178,0.02)) #05050a;border-color:rgba(0,255,178,0.12);border-left:3px solid #00ffb2;--ac:#00ffb2;';
    } else if (track.status === 'gated' || track.status === 'gating') {
        extraStyle = 'background:linear-gradient(rgba(255,170,0,0.025),rgba(255,170,0,0.025)) #05050a;border-color:rgba(255,170,0,0.14);border-left:3px solid #ffaa00;--ac:#ffaa00;';
    } else if (track.status === 'blocked') {
        // Amber-tinted chip — a GPU resource issue is a deferrable failure,
        // distinct from 'failed' (correctness/perf).
        extraStyle = 'background:linear-gradient(rgba(255,170,0,0.03),rgba(255,170,0,0.03)) #05050a;border-color:rgba(255,170,0,0.18);border-left:3px solid #ffaa00;--ac:#ffaa00;';
    }
    chip.style.cssText = `left:${x}px;top:${y}px;width:${L2.TRACK_W}px;height:${L2.TRACK_H}px;animation-delay:${delay}s;${extraStyle}`;
    const sym = track.status === 'shipped' ? '\u25B8'
              : track.status === 'failed'  ? '\u2717'
              : track.status === 'validated' ? '\u2713'
              : track.status === 'gated'   ? '\u25D1'
              : track.status === 'blocked' ? '\u29B8'   /* circled reverse solidus */
              : track.status === 'gating'  ? '\u29D6'   /* white hourglass */
              : '\u25B6';
    // Header row: name (left) + LOSSLESS/LOSSY badge (middle) + status tag (right).
    // Badge only renders for shipped / validated / gated \u2014 those are the terminal
    // verdicts where precision cost matters. Failed / active / blocked / gating
    // don't carry a lossy decision yet.
    const hasLossy = (track.lossy !== undefined && track.lossy !== null)
        && ['shipped', 'validated', 'gated'].includes(track.status);
    const lossyBadgeHtml = hasLossy
        ? `<span class="cb2-badge cb2-badge-head ${track.lossy ? 'cb2-badge-ly' : 'cb2-badge-ll'}">${track.lossy ? 'LOSSY' : 'LOSSLESS'}</span>`
        : '';
    // Visible label prefers the debate-minted display name; track.name (the
    // op_id) stays the click/navigation key.
    const trackLabel = track.displayName || track.name;
    let content = `<div class="cb2-ctl"></div><div class="cb2-cbr"></div>
      <div class="cb2-t-head-row">
        <div class="cb2-t-name">${_esc(trackLabel.length > 20 ? trackLabel.slice(0, 19) + '\u2026' : trackLabel)}</div>
        ${lossyBadgeHtml}
        <div class="cb2-t-tag">${sym} ${_esc(track.status.toUpperCase())}</div>
      </div>`;

    if (track.status === 'shipped') {
        const kSpeed = Number(track.kernelSpeedup);
        const heroVal = Number.isFinite(kSpeed) && kSpeed > 0 ? kSpeed.toFixed(2) : track.speedup;
        // Status line trimmed to `· ACCEPTED` per mockup (line 427) — the header
        // tag already says "▸ SHIPPED", so the status line carries a second
        // signal rather than repeating the first.
        content += `<div class="cb2-t-status" style="color:#00ffb2">· ACCEPTED</div>`;
        // Body row: hero (left) + e2e (right) per approved mockup layout
        const eSpeed = Number(track.e2eSpeedup);
        const amPp = Number(track.amdahlPredictionPp);
        let e2eCell = '';
        if (Number.isFinite(eSpeed) && eSpeed > 0) {
            e2eCell = `<div class="cb2-t-e2e"><span class="cb2-e2e-label">e2e:</span>${_esc(eSpeed.toFixed(2))}\u00d7</div>`;
        } else if (Number.isFinite(amPp)) {
            const sign = amPp >= 0 ? '+' : '';
            e2eCell = `<div class="cb2-t-e2e" style="color:rgba(232,232,240,0.55);font-style:italic"><span class="cb2-e2e-label">amdahl:</span>${sign}${_esc(amPp.toFixed(2))}pp</div>`;
        }
        content += `<div class="cb2-t-body-row"><div class="cb2-t-hero" style="color:#00ffb2;text-shadow:0 0 20px rgba(0,255,178,0.25)">${_esc(heroVal)}\u00d7</div>${e2eCell}</div>`;
    } else if (track.status === 'failed') {
        // Status line swapped to `· VALIDATION ERR` — the header tag already
        // says "✗ FAILED", so repeating "FAILED" on both lines was redundant.
        content += `<div class="cb2-t-status" style="color:#ff3355">· VALIDATION ERR</div>`;
        content += `<div class="cb2-t-body-row"><div class="cb2-t-hero" style="color:#ff3355;font-size:18px">FAILED</div></div>`;
    } else if (track.status === 'validated') {
        const kSpeed = Number(track.kernelSpeedup);
        const heroVal = Number.isFinite(kSpeed) && kSpeed > 0 ? kSpeed.toFixed(2) : track.speedup;
        content += `<div class="cb2-t-status" style="color:#00ffb2">PASS \u2014 AWAITING MERGE</div>`;
        const eSpeed = Number(track.e2eSpeedup);
        const amPp = Number(track.amdahlPredictionPp);
        let e2eCell = '';
        if (Number.isFinite(eSpeed) && eSpeed > 0) {
            e2eCell = `<div class="cb2-t-e2e"><span class="cb2-e2e-label">e2e:</span>${_esc(eSpeed.toFixed(2))}\u00d7</div>`;
        } else if (Number.isFinite(amPp)) {
            const sign = amPp >= 0 ? '+' : '';
            e2eCell = `<div class="cb2-t-e2e" style="color:rgba(232,232,240,0.55);font-style:italic"><span class="cb2-e2e-label">amdahl:</span>${sign}${_esc(amPp.toFixed(2))}pp</div>`;
        }
        content += `<div class="cb2-t-body-row"><div class="cb2-t-hero" style="color:#00ffb2;text-shadow:0 0 18px rgba(0,255,178,0.22)">${_esc(heroVal)}\u00d7</div>${e2eCell}</div>`;
    } else if (track.status === 'gated') {
        const kSpeed = Number(track.kernelSpeedup);
        const heroVal = Number.isFinite(kSpeed) && kSpeed > 0 ? kSpeed.toFixed(2) : track.speedup;
        // Status line trimmed to `\u00b7 KERNEL-ONLY PASS` \u2014 header tag already
        // says "\u25d1 GATED", and LOSSY/LOSSLESS badge moved up to the header row.
        content += `<div class="cb2-t-status" style="color:#ffaa00">\u00b7 KERNEL-ONLY PASS</div>`;
        content += `<div class="cb2-t-hero" style="color:#ffaa00;text-shadow:0 0 18px rgba(255,170,0,0.22)">${_esc(heroVal)}\u00d7</div>`;
        // Gated = kernel-only verdict. Show amdahl prediction (if present) so the
        // reader knows what the ship-gate math predicted, but NEVER reuse the kernel
        // number as an e2e value. Lossy badge is now in the header \u2014 footer
        // line carries only the amdahl / "kernel-only" fallback.
        const amPp = Number(track.amdahlPredictionPp);
        if (Number.isFinite(amPp)) {
            const sign = amPp >= 0 ? '+' : '';
            content += `<div class="cb2-sec" style="margin-top:3px;color:rgba(255,170,0,0.55)">amdahl: ${sign}${_esc(amPp.toFixed(2))}pp</div>`;
        } else {
            content += `<div class="cb2-sec" style="margin-top:3px;color:rgba(255,170,0,0.55)">kernel-only</div>`;
        }
    } else if (track.status === 'blocked') {
        // GPU_BLOCKED — track could not run because no GPU was allocated in time
        // (or the gating sub-agent was starved). Amber pill + amber hero label so the
        // reader can distinguish it from a gated kernel-only pass.
        content += `<div class="cb2-t-status" style="color:#ffaa00;display:flex;align-items:center;gap:6px"><span class="cb2-gated-pill">\u29B8 GPU BLOCKED</span></div>`;
        content += `<div class="cb2-t-hero" style="color:#ffaa00;text-shadow:0 0 14px rgba(255,170,0,0.18);font-size:16px">GPU BLOCKED</div>`;
        const failText = track.failReason
            ? (track.failReason.length > 30 ? track.failReason.slice(0, 28) + '\u2026' : track.failReason)
            : 'gating skipped \u2014 no GPU';
        content += `<div class="cb2-sec" style="margin-top:3px;color:rgba(255,170,0,0.55)">${_esc(failText)}</div>`;
    } else if (track.status === 'gating') {
        // GATING_REQUIRED — validator demanded a gating rerun but the track
        // has not yet resolved. Show an amber "awaiting gating" placeholder.
        content += `<div class="cb2-t-status" style="color:#ffaa00;display:flex;align-items:center;gap:6px"><span class="cb2-gated-pill">\u29D6 GATING REQUIRED</span></div>`;
        content += `<div class="cb2-t-hero" style="color:#ffaa00;text-shadow:0 0 14px rgba(255,170,0,0.18);font-size:16px">AWAITING GATE</div>`;
        if (track.subStage) {
            const ss = track.subStage.length > 28 ? track.subStage.slice(0, 26) + '\u2026' : track.subStage;
            content += `<div class="cb2-sec" style="margin-top:3px;color:rgba(255,170,0,0.55)">${_esc(ss)}</div>`;
        }
    } else if (track.status === 'active') {
        content += `<div class="cb2-t-status" style="color:#00f3ff">IN PROGRESS</div>`;
        if (track.subStage) {
            const ss = track.subStage.length > 28 ? track.subStage.slice(0, 26) + '\u2026' : track.subStage;
            content += `<div class="cb2-sec" style="margin-top:3px">${_esc(ss)}</div>`;
        }
    }

    // Per-bucket verdict pip row. Only shipped / validated tracks carry verdicts;
    // gated / blocked / gating / failed / active keep their existing status-only
    // content. Failed tracks show "correctness: {reason}" in the pip area.
    // Truncate to 3 pips + `+N more` beyond 4 buckets so the row never
    // wraps at the 260px chip width.
    if (track.status === 'failed') {
        const failText = track.failReason
            ? (track.failReason.length > 30 ? track.failReason.slice(0, 28) + '\u2026' : track.failReason)
            : 'correctness failure';
        content += `<div class="cb2-pip-row" style="color:rgba(255,51,85,0.55)"><span>correctness: ${_esc(failText)}</span></div>`;
    } else if ((track.status === 'shipped' || track.status === 'validated') && track.perBsVerdict) {
        const buckets = _bucketRecords(track.perBsVerdict);
        if (buckets.length) {
            const visibleBuckets = buckets.length > 4 ? buckets.slice(0, 3) : buckets;
            const overflow = buckets.length - visibleBuckets.length;
            const pips = visibleBuckets.map(bucket => {
                const verdict = track.perBsVerdict[bucket.tag];
                let cls = 'noise';
                if (verdict === 'PASS') cls = 'pass';
                else if (verdict === 'REGRESSED' || verdict === 'CATASTROPHIC') cls = 'regress';
                else if (verdict === 'NOISE') cls = 'noise';
                return `<span class="cb2-pip ${cls}" title="${_esc(bucket.tag)}"><span class="cb2-dot"></span>${_esc(bucket.compactLabel)}</span>`;
            }).join('');
            const overflowTag = overflow > 0
                ? `<span class="cb2-pip-overflow">+${overflow}</span>`
                : '';
            // LOSSY/LOSSLESS badge lives in the header row now (see cb2-t-head-row).
            // Keeps the pip row uncluttered and lets all 3+ pips fit the 260px width.
            content += `<div class="cb2-pip-row">${pips}${overflowTag}</div>`;
        }
    }

    chip.innerHTML = content;

    // Tooltip — rich version for shipped/validated (with e2e data), simple for others
    if (_cb2Tip) {
        chip.addEventListener('mouseenter', e => {
            const ttTitle = _esc(trackLabel).toUpperCase();
            if (track.status === 'shipped' || track.status === 'validated') {
                // Rich tooltip per approved mockup: title, summary, note, table, legend, source
                const ttKSpeed = Number(track.kernelSpeedup);
                const ttESpeed = Number(track.e2eSpeedup);
                const ttAmPp = Number(track.amdahlPredictionPp);
                const lossyLabel = (track.lossy !== undefined && track.lossy !== null)
                    ? (track.lossy ? 'LOSSY' : 'LOSSLESS') : '';
                // Surface the op_id in the meta pill when the title shows a
                // display name, so the tracking handle stays discoverable.
                const opIdPill = track.displayName ? track.name : '';
                const metaPill = [opIdPill, lossyLabel, track.status.toUpperCase()].filter(Boolean).join(' \u00b7 ');
                const summary = [];
                if (Number.isFinite(ttKSpeed) && ttKSpeed > 0)
                    summary.push({ label: 'KERNEL \u00d7', value: ttKSpeed.toFixed(2) + '\u00d7', cls: 'mint' });
                if (Number.isFinite(ttESpeed) && ttESpeed > 0)
                    summary.push({ label: 'E2E \u00d7', value: ttESpeed.toFixed(2) + '\u00d7', cls: 'mint' });
                else if (Number.isFinite(ttAmPp))
                    summary.push({ label: 'AMDAHL', value: (ttAmPp >= 0 ? '+' : '') + ttAmPp.toFixed(2) + 'pp', cls: 'cyan' });
                const primaryBucket = track.e2eLatencyOpt ? _bucketRecords(track.e2eLatencyOpt)[0] : null;
                const noteText = primaryBucket
                    ? 'E2E LATENCY PER BUCKET \u00b7 chip shows {AVG} of ' + primaryBucket.label
                    : 'chip shows {AVG} \u00b7 table shows full percentiles';
                const latMap = track.e2eLatencyOpt || null;
                const pbv = track.perBsVerdict || null;
                const sourceText = track.amdahlPredictionPp != null
                    ? 'vs baseline \u00b7 amdahl +' + (Number(track.amdahlPredictionPp) || 0).toFixed(1) + 'pp'
                    : 'vs baseline';
                const titleHtml = `<div class="cb2-tt-title"><span>TRACK \u00b7 ${ttTitle}</span><span class="meta">${_esc(metaPill)}</span></div>`;
                const bodyHtml = _buildRichTooltipBody({
                    summary: summary.length ? summary : null,
                    note: noteText,
                    latencyMap: latMap,
                    perBsVerdict: pbv,
                    legend: !!pbv,
                    source: sourceText,
                });
                _cb2Tip.show(e, null, titleHtml + bodyHtml);
            } else {
                // Simple tooltip for gated/failed/blocked/gating/active
                const color = track.status === 'failed' ? '#ff3355'
                            : track.status === 'gated' || track.status === 'blocked' || track.status === 'gating' ? '#ffaa00'
                            : '#00f3ff';
                const statusLabel = track.status === 'gated'     ? 'GATED PASS'
                                  : track.status === 'blocked'   ? 'GPU BLOCKED'
                                  : track.status === 'gating'    ? 'GATING REQUIRED'
                                  : track.status.toUpperCase();
                let body = `<span class="cb2-tt-dim">${_esc(track.detail || '')}</span><br>`;
                body += `Status: <span style="color:${color}">${_esc(statusLabel)}</span><br>`;
                if (track.status !== 'failed') {
                    const ttKSpeed = Number(track.kernelSpeedup);
                    if (Number.isFinite(ttKSpeed) && ttKSpeed > 0) body += `Kernel: <span class="cb2-tt-val">${_esc(ttKSpeed.toFixed(2))}\u00d7</span><br>`;
                }
                if (track.failReason) body += `<span style="color:#ff3355">${_esc(track.failReason)}</span><br>`;
                if (track.lossy) body += `<span style="color:#ffaa00">Lossy (precision trade-off)</span><br>`;
                if (track.commitSha) body += `<span class="cb2-tt-dim">Commit: ${_esc(track.commitSha.slice(0, 9))}</span>`;
                _cb2Tip.show(e, _esc(trackLabel), body);
            }
        });
        chip.addEventListener('mouseleave', () => _cb2Tip.hide());
    }
    if (onClick) chip.addEventListener('click', () => onClick(roundId, track.name, false));

    canvas.appendChild(chip);
    return { right: x + L2.TRACK_W, cy: y + L2.TRACK_H / 2, left: x, top: y };
}

function makeRationaleCard(parent, x, y, width, rat, delay, onOpen) {
    // `lg-rationale-group` gates visibility on DEBATE chip hover/focus.
    // When `x` is null the card is rendered inside a relative wrapper
    // (stacked), so we emit position:relative instead of absolute —
    // this lets the wrapper cap height + scroll for large stacks.
    const card = _h('div', 'lg-rationale-card lg-rationale-group');
    if (x == null) {
        card.style.cssText = `position:relative;width:${width}px;margin-bottom:8px;`;
    } else {
        card.style.cssText = `left:${x}px;top:${y}px;width:${width}px;`;
    }
    const stance = (rat.stance || 'proposed').toLowerCase();
    const championName = rat.championId || 'CHAMPION';
    const desc = _esc(rat.description || 'No description');
    // Champion name is truncated via CSS (text-overflow: ellipsis) at 150px max-width.
    card.innerHTML =
        `<div class="lg-rationale-card__head">
           <span class="lg-rationale-card__champion" title="${_esc(rat.championId || '')}">${_esc(championName)}</span>
           <span class="lg-rationale-card__stance ${_esc(stance)}">${_esc(stance.toUpperCase())}</span>
         </div>
         <div class="lg-rationale-card__desc">${desc}</div>
         <div class="lg-rationale-card__foot">
           <span class="lg-rationale-card__op">${_esc(rat.opId || '')}</span>
           <span class="lg-rationale-card__arrow">OPEN &#8594;</span>
         </div>`;
    if (_cb2Tip) {
        card.addEventListener('mouseenter', e => {
            let body = `<span class="cb2-tt-val">${_esc(rat.championId || 'Champion')}</span><br>`;
            body += `Stance: <span style="color:#00f3ff">${_esc(stance.toUpperCase())}</span><br>`;
            if (rat.opId) body += `Op: <span class="cb2-tt-dim">${_esc(rat.opId)}</span><br>`;
            body += `<span class="cb2-tt-dim">${_esc(rat.description || '')}</span>`;
            _cb2Tip.show(e, 'DEBATE RATIONALE', body);
        });
        card.addEventListener('mouseleave', () => _cb2Tip.hide());
    }
    if (onOpen) card.addEventListener('click', (e) => { e.stopPropagation(); onOpen(rat); });
    // Keyboard actuation: Enter/Space both trigger open, matching click.
    card.setAttribute('tabindex', '0');
    card.setAttribute('role', 'button');
    card.setAttribute('aria-label',
        `Open rationale — ${rat.championId || 'champion'} ${(rat.stance || 'proposed')}`);
    if (onOpen) {
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
                e.preventDefault();
                e.stopPropagation();
                onOpen(rat);
            }
        });
    }
    parent.appendChild(card);
    return card;
}

function makeResolutionBadge(canvas, x, y, width, resolution, delay, onOpen) {
    const badge = _h('div', 'lg-resolution-badge');
    badge.style.cssText = `left:${x}px;top:${y}px;width:${width}px;animation-delay:${delay}s;`;
    const shaShort = resolution.headSha ? resolution.headSha.slice(0, 7) : '';
    const typePretty = (resolution.resolutionType || 'merge')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
    badge.innerHTML =
        `<div class="lg-resolution-badge__head">
           <span class="lg-resolution-badge__tag">&#9670; RESOLUTION MERGED</span>
           ${shaShort ? `<span class="lg-resolution-badge__sha">${_esc(shaShort)}</span>` : ''}
         </div>
         <div class="lg-resolution-badge__type"><strong>${_esc(typePretty)}</strong></div>`;
    if (_cb2Tip) {
        badge.addEventListener('mouseenter', e => {
            let body = `Type: <span class="cb2-tt-val">${_esc(typePretty)}</span><br>`;
            if (resolution.headSha) body += `SHA: <span class="cb2-tt-val">${_esc(resolution.headSha)}</span><br>`;
            if (resolution.description) body += `<span class="cb2-tt-dim">${_esc(resolution.description)}</span>`;
            _cb2Tip.show(e, 'RESOLUTION MERGED', body);
        });
        badge.addEventListener('mouseleave', () => _cb2Tip.hide());
    }
    if (onOpen) badge.addEventListener('click', (e) => { e.stopPropagation(); onOpen(resolution); });
    canvas.appendChild(badge);
    return badge;
}

function makeIntegChip(canvas, x, y, data, delay, roundId, onClick, roundActive, reachable) {
    const isReachable = typeof reachable === 'function' ? reachable(roundId, 5) : true;
    const pastCls = roundActive === false ? ' cb2-row-past' : '';
    const disabledCls = isReachable ? '' : ' cb2-chip-disabled';
    const status = data.status || (data.synthetic ? 'synthetic' : 'completed');
    const isRunning = status === 'in_progress' || status === 'running';
    const isFailed  = status === 'failed';
    // v4.0 traffic-light state from per_bs_verdict. Only applies to terminal
    // (non-running, non-failed) integration chips — running/failed retain their
    // existing cyan-pulse / red-fail treatments.
    const verdictAgg = (!isRunning && !isFailed) ? _verdictAggregate(data.perBsVerdict) : 'NONE';
    let trafficCls = '';
    if (!isRunning && !isFailed) {
        if (verdictAgg === 'GREEN' || verdictAgg === 'NONE') trafficCls = ' cb2-integ-green';
        else if (verdictAgg === 'RED') trafficCls = ' cb2-integ-red';
    }
    const chip = _h('div', `cb2-hud cb2-integ cb2-row-anim${trafficCls}${pastCls}${disabledCls}`);
    // Paint per-status accent inline (CSS owns the base .cb2-integ amber; override
    // for running = cyan pulse, failed = red, completed = keep amber → mint hero).
    let extraStyle = '';
    if (isRunning) {
        // Don't override `animation` — .cb2-row-anim needs cb2FadeSlideIn to bring
        // opacity from 0 → 1. Use box-shadow for the pulse, driven by a pseudo-hover
        // via ::after. Instead, attach the glow via `cb2-integ-pulse` class which
        // layers a box-shadow-only animation on top.
        extraStyle = 'background:linear-gradient(rgba(0,243,255,0.03),rgba(0,243,255,0.03)) #05050a;border-color:rgba(0,243,255,0.25);border-left:3px solid #00f3ff;--ac:#00f3ff;box-shadow:0 0 18px rgba(0,243,255,0.22);';
    } else if (isFailed) {
        extraStyle = 'background:linear-gradient(rgba(255,51,85,0.03),rgba(255,51,85,0.03)) #05050a;border-color:rgba(255,51,85,0.28);border-left:3px solid #ff3355;--ac:#ff3355;';
    }
    chip.style.cssText = `left:${x}px;top:${y}px;width:${L2.INTEG_W}px;height:${L2.INTEG_H}px;animation-delay:${delay}s;${extraStyle}`;
    const labelColor = isRunning ? '#00f3ff' : isFailed ? '#ff3355' : null;
    const heroColor  = isRunning ? '#00f3ff' : isFailed ? '#ff3355' : null;
    const labelText  = isRunning ? 'MERGING' : isFailed ? 'MERGE FAIL' : 'INTEGRATE';

    // Hero = this-round speedup when present (v4.0 chip contract); fall back to
    // the legacy cumulative `data.speedup` for pre-v4.0 paths.
    const __integThisRound = Number(data.thisRoundSpeedup);
    const __integHero = Number.isFinite(__integThisRound) && __integThisRound > 0
        ? __integThisRound.toFixed(2)
        : (data.speedup != null ? String(data.speedup) : '1.00');

    let content = `<div class="cb2-ctl"></div><div class="cb2-cbr"></div>
      <div class="cb2-desig"${labelColor ? ` style="color:${labelColor}"` : ''}>INT</div>
      <div class="cb2-head-row">
        <div class="cb2-label"${labelColor ? ` style="color:${labelColor}"` : ''}>${_esc(labelText)}</div>
        ${(!isRunning && !isFailed) ? `<div class="cb2-e2e-tag">\u00b7 E2E LAT \u00d7</div>` : ''}
      </div>
      <div class="cb2-hero"${heroColor ? ` style="color:${heroColor};text-shadow:0 0 16px ${heroColor}40"` : ''}>${_esc(__integHero)}\u00d7</div>`;

    if (isRunning || isFailed) {
        // Legacy delta/progress path for running / failed chips. A multiplier
        // "×" and percentage-points "pp" are different quantities and must
        // never be conflated (the old "+0.068×" misread fix).
        if (typeof data.deltaPp === 'number' && Number.isFinite(data.deltaPp)) {
            const deltaColor = data.deltaPp > 0 ? '#00ffb2' : '#ff3355';
            const sign = data.deltaPp >= 0 ? '+' : '';
            content += `<div class="cb2-sec" style="color:${deltaColor}">\u0394 ${sign}${_esc(data.deltaPp.toFixed(2))}pp</div>`;
        } else if (data.progressText) {
            const progColor = isRunning ? 'rgba(0,243,255,0.8)' : 'rgba(255,51,85,0.85)';
            content += `<div class="cb2-sec" style="color:${progColor};font-weight:600">${_esc(data.progressText)}</div>`;
        }
    } else {
        // Terminal chip — cum secondary (left) + per-bucket pips (right).
        const __integCum = Number(data.cumulativeSpeedup);
        const __integCumValue = Number.isFinite(__integCum) && __integCum > 0
            ? __integCum.toFixed(2)
            : (data.speedup != null ? String(data.speedup) : __integHero);
        let __integPipsHtml = '';
        if (data.perBsVerdict && typeof data.perBsVerdict === 'object') {
            const buckets = _bucketRecords(data.perBsVerdict);
            if (buckets.length) {
                // Per approved mockup (line 338): each verdict dot carries its
                // full bucket identity directly beneath. Stacked vertically via
                // .cb2-integ-pip wrapper (dot on top, bucket label below).
                __integPipsHtml = `<div class="cb2-pips">` + buckets.map(bucket => {
                    const v = data.perBsVerdict[bucket.tag];
                    let cls = 'noise';
                    if (v === 'PASS') cls = 'pass';
                    else if (v === 'REGRESSED' || v === 'CATASTROPHIC') cls = 'regress';
                    return `<span class="cb2-integ-pip ${cls}" title="${_esc(bucket.tag)}">`
                         + `<span class="cb2-dot ${cls}"></span>`
                         + `<span class="cb2-bs-label">${_esc(bucket.compactLabel)}</span>`
                         + `</span>`;
                }).join('') + `</div>`;
            }
        }
        content += `<div class="cb2-bottom-row">`
                 + `<div class="cb2-cum"><span class="cb2-cum-label">cum</span><span class="cb2-cum-val">${_esc(__integCumValue)}\u00d7</span></div>`
                 + __integPipsHtml
                 + `</div>`;
    }
    chip.innerHTML = content;

    // Tooltip — rich version for completed integration, simple for running/failed
    if (_cb2Tip) {
        chip.addEventListener('mouseenter', e => {
            if (!isRunning && !isFailed) {
                // Rich tooltip per approved mockup
                const __ttThisRound = Number(data.thisRoundSpeedup);
                const __ttCum = Number(data.cumulativeSpeedup);
                const thisRoundVal = Number.isFinite(__ttThisRound) ? __ttThisRound.toFixed(2) + '\u00d7' : (data.speedup || '1.00') + '\u00d7';
                const cumVal = Number.isFinite(__ttCum) ? __ttCum.toFixed(2) + '\u00d7' : (data.speedup || '1.00') + '\u00d7';
                const verdictLabel = verdictAgg === 'GREEN' ? 'ALL PASS'
                                   : verdictAgg === 'RED' ? 'ALL REGRESSED'
                                   : verdictAgg === 'AMBER' ? 'MIXED'
                                   : '';
                const opsCount = data.mergedOps ? data.mergedOps.length : 0;
                const metaPill = [verdictLabel, opsCount > 0 ? 'SHIPPED ' + opsCount + ' OPS' : ''].filter(Boolean).join(' \u00b7 ');
                const titleHtml = `<div class="cb2-tt-title"><span>INTEGRATION \u00b7 R${roundId}</span><span class="meta">${_esc(metaPill)}</span></div>`;
                // Summary: THIS ROUND / CUMULATIVE / delta
                const summary = [
                    { label: 'THIS ROUND \u00d7', value: thisRoundVal, cls: 'mint' },
                    { label: 'CUMULATIVE \u00d7', value: cumVal, cls: 'cyan' },
                ];
                if (typeof data.deltaPp === 'number' && Number.isFinite(data.deltaPp)) {
                    const sign = data.deltaPp >= 0 ? '+' : '';
                    const deltaCls = data.deltaPp < 0 ? 'red' : 'amber';
                    summary.push({ label: '\u0394 vs R' + (roundId > 1 ? roundId - 1 : 1), value: sign + data.deltaPp.toFixed(2) + 'pp', cls: deltaCls });
                }
                const noteText = 'E2E LATENCY COMBINED \u00b7 chip hero = {THIS ROUND} (R' + (roundId > 1 ? roundId - 1 : 1) + ' baseline \u00f7 this integration)';
                const latMap = data.combinedMap || data.e2eLatencyCombined || null;
                const pbv = data.perBsVerdict || null;
                const mergedList = data.mergedOps && data.mergedOps.length
                    ? 'merged: ' + data.mergedOps.join(', ')
                    : '';
                const sourceText = ['vs R' + (roundId > 1 ? roundId - 1 : 1) + ' baseline \u00b7 threshold \u00b15%', mergedList].filter(Boolean).join('\n');
                const bodyHtml = _buildRichTooltipBody({
                    summary,
                    note: noteText,
                    latencyMap: latMap,
                    perBsVerdict: pbv,
                    legend: !!pbv,
                    source: sourceText,
                });
                _cb2Tip.show(e, null, titleHtml + bodyHtml);
            } else {
                // Simple tooltip for running/failed
                const statusLabel = isRunning ? 'IN PROGRESS' : 'FAILED';
                const statusColor = isRunning ? '#00f3ff' : '#ff3355';
                let body = `Status: <span style="color:${statusColor}">${_esc(statusLabel)}</span><br>`;
                if (data.progressText) body += `<span class="cb2-tt-dim">${_esc(data.progressText)}</span><br>`;
                if (data.mergedOps && data.mergedOps.length) body += `<span class="cb2-tt-dim">Merged: ${data.mergedOps.map(_esc).join(', ')}</span><br>`;
                if (data.failedOps && data.failedOps.length) body += `<span style="color:rgba(255,51,85,0.8)">Skipped: ${data.failedOps.map(_esc).join(', ')}</span><br>`;
                if (data.elapsed) body += `<span class="cb2-tt-dim">Duration: ${_esc(data.elapsed)}</span>`;
                _cb2Tip.show(e, 'INTEGRATION', body);
            }
        });
        chip.addEventListener('mouseleave', () => _cb2Tip.hide());
    }
    if (onClick && isReachable) chip.addEventListener('click', () => onClick(roundId, 5, true));

    canvas.appendChild(chip);
    return { right: x + L2.INTEG_W, cy: y + L2.INTEG_H / 2, left: x };
}

function makeEvalDiamond(canvas, cx, cy, decision, delay, roundId, onClick, evalNote, reachable) {
    const isReachable = typeof reachable === 'function' ? reachable(roundId, 6) : true;
    const isActive = decision === 'ACTIVE';
    const color = decision === 'CONTINUE' ? '#00f3ff'
                : decision === 'STOP'     ? '#ff3355'
                : /* ACTIVE */               '#00f3ff';
    const s = L2.EVAL_SIZE;
    const d = _h('div', 'cb2-row-anim');
    d.style.cssText = `position:absolute;z-index:2;left:${cx - s * 0.72}px;top:${cy - s * 0.72}px;width:${s * 1.44}px;height:${s * 1.44}px;animation-delay:${delay}s;display:flex;align-items:center;justify-content:center`;
    // For ACTIVE: reuse the existing cb2GlowCranked keyframe (already defined in
    // lightgrid.css for active track chips) on the outer diamond; inner is a bright
    // dot instead of GO / STOP text. For CONTINUE / STOP: solid border + text.
    const borderStyle = isActive ? 'dashed' : 'solid';
    const shadowIntensity = isActive ? '55' : '40';
    const innerContent = isActive
        ? `<span style="transform:rotate(-45deg);width:${Math.round(s * 0.26)}px;height:${Math.round(s * 0.26)}px;border-radius:50%;background:${color};box-shadow:0 0 10px ${color};"></span>`
        : `<span style="transform:rotate(-45deg);font-family:var(--font-head);font-size:7px;font-weight:700;color:${color};letter-spacing:0.5px">${decision === 'CONTINUE' ? 'GO' : 'STOP'}</span>`;
    // cb2GlowCranked is a 1s box-shadow / border-color pulse already baked into CSS.
    const outerAnim = isActive ? 'animation:cb2GlowCranked 1.6s ease-in-out infinite;' : '';
    d.innerHTML = `<div style="width:${s}px;height:${s}px;transform:rotate(45deg);background:#0c0c18;border:1.5px ${borderStyle} ${color};border-radius:3px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 12px ${color}${shadowIntensity};cursor:pointer;transition:box-shadow 0.2s;${outerAnim}">
      ${innerContent}</div>`;

    // Tooltip
    if (_cb2Tip) {
        const labelMap = { CONTINUE: 'GO', STOP: 'STOP', ACTIVE: 'PENDING' };
        const decisionLabel = labelMap[decision] || decision;
        const fallbackNote = isActive
            ? 'Tracks still in progress — campaign decision pending'
            : 'Evaluating convergence criteria';
        d.addEventListener('mouseenter', e => _cb2Tip.show(e, 'CAMPAIGN EVAL',
            `Decision: <span style="color:${color}">${_esc(decisionLabel)}</span><br>` +
            (evalNote ? `<span class="cb2-tt-dim">${_esc(evalNote)}</span>` : `<span class="cb2-tt-dim">${_esc(fallbackNote)}</span>`)
        ));
        d.addEventListener('mouseleave', () => _cb2Tip.hide());
    }
    if (!isReachable) {
        d.classList.add('cb2-diamond-disabled');
    }
    if (onClick && isReachable) d.addEventListener('click', () => onClick(roundId, 6, true));

    canvas.appendChild(d);
    return { right: cx + s * 0.72, cx, cy, left: cx - s * 0.72 };
}

// ── Tooltip CSS (injected once) ──────────────────────────────────────────────

const CB2_CSS = `
  .cb2-tooltip {
    position: fixed; z-index: 200; pointer-events: none;
    background: rgba(8,8,18,0.96); border: 1px solid rgba(0,243,255,0.35);
    border-radius: 4px; padding: 12px 14px 10px 14px;
    font-family: 'Share Tech Mono', monospace; font-size: 11px; color: #e8e8f0;
    box-shadow: 0 0 24px rgba(0,243,255,0.12), 0 8px 32px rgba(0,0,0,0.7);
    opacity: 0; transition: opacity 0.15s;
    max-width: 380px; line-height: 1.6;
    backdrop-filter: blur(8px);
  }
  .cb2-tooltip.visible { opacity: 1; }
  .cb2-tooltip .cb2-tt-label {
    color: #00f3ff; font-family: 'Orbitron', sans-serif;
    font-size: 9px; letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 4px; padding-bottom: 4px;
    border-bottom: 1px solid rgba(0,243,255,0.1);
  }
  .cb2-tooltip .cb2-tt-val { color: #00ffb2; }
  .cb2-tooltip .cb2-tt-dim { color: rgba(232,232,240,0.4); }
  /* ── Rich tooltip: title bar ── */
  .cb2-tooltip .cb2-tt-title {
    font-family: 'Orbitron', sans-serif; font-size: 10px; letter-spacing: 2px;
    color: #00f3ff; margin-bottom: 6px; padding-bottom: 6px;
    border-bottom: 1px solid rgba(0,243,255,0.12);
    display: flex; justify-content: space-between; align-items: center;
  }
  .cb2-tooltip .cb2-tt-title .meta {
    font-family: 'Share Tech Mono', monospace; font-size: 8px;
    color: rgba(232,232,240,0.55); letter-spacing: 1px; font-weight: normal;
  }
  /* ── Rich tooltip: summary stats row ── */
  .cb2-tooltip .cb2-tt-summary {
    display: flex; gap: 18px; padding: 8px 0 10px 0;
    border-bottom: 1px solid rgba(232,232,240,0.06); margin-bottom: 6px;
  }
  .cb2-tooltip .cb2-tt-summary .stat {
    display: flex; flex-direction: column; gap: 2px;
  }
  .cb2-tooltip .cb2-tt-summary .stat-label {
    font-family: 'Orbitron', sans-serif; font-size: 7px;
    letter-spacing: 1.5px; color: rgba(232,232,240,0.55);
  }
  .cb2-tooltip .cb2-tt-summary .stat-val {
    font-family: 'Orbitron', sans-serif; font-size: 14px; font-weight: 700;
  }
  .cb2-tooltip .cb2-tt-summary .stat-val.mint { color: #00ffb2; text-shadow: 0 0 6px rgba(0,255,178,0.25); }
  .cb2-tooltip .cb2-tt-summary .stat-val.cyan { color: #00f3ff; }
  .cb2-tooltip .cb2-tt-summary .stat-val.amber { color: #ffaa00; }
  .cb2-tooltip .cb2-tt-summary .stat-val.red { color: #ff3355; text-shadow: 0 0 6px rgba(255,51,85,0.25); }
  /* ── Rich tooltip: note row ── */
  .cb2-tooltip .cb2-tt-note {
    font-family: 'Share Tech Mono', monospace; font-size: 9px;
    color: rgba(232,232,240,0.55); padding: 2px 0 6px 0; letter-spacing: 0.5px;
  }
  .cb2-tooltip .cb2-tt-note .hl {
    color: #00f3ff; font-weight: bold;
    background: rgba(0,243,255,0.08); padding: 1px 4px;
  }
  /* ── Rich tooltip: footer (legend + source) ── */
  .cb2-tooltip .cb2-tt-footer {
    margin-top: 8px; padding-top: 6px;
    border-top: 1px solid rgba(232,232,240,0.06);
    font-family: 'Share Tech Mono', monospace; font-size: 9px;
    color: rgba(232,232,240,0.55); letter-spacing: 0.5px; line-height: 1.5;
  }
  .cb2-tooltip .cb2-tt-footer .legend {
    display: inline-flex; align-items: center; gap: 4px; margin-right: 10px;
  }
  .cb2-tooltip .cb2-tt-footer .dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  }
  .cb2-tooltip .cb2-tt-footer .dot.pass { background: #00ffb2; box-shadow: 0 0 4px rgba(0,255,178,0.6); }
  .cb2-tooltip .cb2-tt-footer .dot.noise { background: rgba(232,232,240,0.45); }
  .cb2-tooltip .cb2-tt-footer .dot.regress { background: #ff3355; box-shadow: 0 0 4px rgba(255,51,85,0.6); }
  .cb2-tooltip .cb2-tt-src {
    margin-top: 4px; color: rgba(232,232,240,0.3); font-style: italic;
  }
  /* ── Audit-gate fuse hex ── */
  .cb2-audit-hex-wrap {
    outline: none;
  }
  .cb2-audit-hex-wrap:focus-visible .cb2-audit-hex {
    outline: 2px solid #00f3ff;
    outline-offset: 2px;
  }
  .cb2-audit-hex {
    transition: box-shadow 0.2s, background 0.2s;
  }
  .cb2-audit-hex--running {
    animation: cb2GlowCranked 1.6s ease-in-out infinite;
  }
  .cb2-audit-hex--passed {
    box-shadow: 0 0 6px rgba(0,255,178,0.5);
  }
  .cb2-audit-hex--escalated {
    box-shadow: 0 0 8px rgba(255,51,85,0.65);
  }
  .cb2-audit-hex-label {
    text-align: center;
  }
  @media (prefers-reduced-motion: reduce) {
    .cb2-audit-hex--running {
      animation: none !important;
    }
  }
`;

// ── Rationale reveal (DEBATE chip hover/focus gate) ─────────────────────────
// The DEBATE chip acts as the trigger surface; mousing over the chip
// (or any of its cards) reveals the group, with a ~100ms grace period on
// leave so chip→card traversal doesn't flicker. Keyboard focus also
// reveals (chip is tabbable) — accessibility parity with the pointer path.
function _wireRationaleReveal({ chip, wrap, cards, stackLbl }) {
    if (!chip || !cards || cards.length === 0) return;
    chip.classList.add('cb2-debate-has-rationale');
    if (!chip.hasAttribute('tabindex')) chip.setAttribute('tabindex', '0');
    chip.setAttribute('role', 'button');
    chip.setAttribute('aria-haspopup', 'true');
    chip.setAttribute('aria-expanded', 'false');

    let hideTimer = null;
    let revealed = false;

    const setRevealed = (next) => {
        if (revealed === next) return;
        revealed = next;
        chip.classList.toggle('cb2-debate-revealed', next);
        chip.setAttribute('aria-expanded', next ? 'true' : 'false');
        if (stackLbl) {
            stackLbl.classList.toggle('lg-rationale-stack-label--revealed', next);
        }
        cards.forEach(c => c.classList.toggle('lg-rationale-group--revealed', next));
        if (wrap) wrap.classList.toggle('lg-rationale-group-wrap--revealed', next);
    };

    const isInGroup = (node) => {
        if (!node) return false;
        if (node === chip || node === wrap) return true;
        if (wrap && wrap.contains(node)) return true;
        return cards.some(c => c === node || c.contains(node));
    };

    const show = () => {
        if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
        setRevealed(true);
    };
    const scheduleHide = (e) => {
        // Don't hide when focus/hover moves INTO a card or the wrap —
        // this keeps keyboard tab order chip → card-1 → ... working.
        if (e && isInGroup(e.relatedTarget)) return;
        if (hideTimer) clearTimeout(hideTimer);
        // 100ms grace lets the pointer traverse chip ↔ card without flicker
        hideTimer = setTimeout(() => setRevealed(false), 100);
    };

    chip.addEventListener('mouseenter', show);
    chip.addEventListener('mouseleave', scheduleHide);
    // Use focusin/focusout so `relatedTarget` is populated — this lets
    // us keep the group revealed when focus hops from chip into any card.
    chip.addEventListener('focusin', show);
    chip.addEventListener('focusout', scheduleHide);
    if (wrap) {
        wrap.addEventListener('mouseenter', show);
        wrap.addEventListener('mouseleave', scheduleHide);
        wrap.addEventListener('focusin', show);
        wrap.addEventListener('focusout', scheduleHide);
    }
    cards.forEach(card => {
        card.addEventListener('mouseenter', show);
        card.addEventListener('mouseleave', scheduleHide);
        card.addEventListener('focusin', show);
        card.addEventListener('focusout', scheduleHide);
    });
}

// ── Main hybrid render ──────────────────────────────────────────────────────

function buildBoard(mockupRounds, onNodeClick, onArtifactOpen, reachable, onAuditOpen) {
    const _onClick = typeof onNodeClick === 'function' ? onNodeClick : () => {};
    const _onOpen = typeof onArtifactOpen === 'function' ? onArtifactOpen : () => {};
    // Reachability predicate: (roundId, colIdx) => bool. Default = everything
    // reachable (preserves pre-gating behavior for callers that haven't updated).
    const _reachable = typeof reachable === 'function' ? reachable : () => true;
    const _onAuditOpen = typeof onAuditOpen === 'function' ? onAuditOpen : () => {};

    // Canvas container (relative positioned, HTML chips go here)
    const canvas = document.createElement('div');
    canvas.className = 'cb2-canvas';
    canvas.style.width = L2.SVG_W + 'px';

    // SVG trace layer (absolute overlay)
    const svgEl = document.createElementNS(CB.SVG_NS, 'svg');
    svgEl.classList.add('cb2-trace-layer');
    svgEl.appendChild(_buildSvgDefs());
    canvas.appendChild(svgEl);

    const rowCenters = [];
    const evalPositions = [];
    const rowBottoms = [];  // Lowest y occupied by any chip/badge in each row — used
                            // to route feedback connectors below the track block.
    let cursorY = 30;

    // Single eval-diamond X column, fixed across all rows.
    // A fully populated row layout is:
    //   X_START + 3 stages (STAGE_W each + STAGE_GAP between)
    //     + BRANCH_GAP + BRANCH_GAP + TRACK_W (track block)
    //     + BRANCH_GAP + INTEG_W (integ chip)
    //     + BRANCH_GAP + EVAL_SIZE*0.72
    // This is where a stage-6 row plants its diamond; pinning every row to
    // this same X makes the diamonds line up vertically regardless of
    // whether tracks or integration rendered.
    const EVAL_CX =
        L2.X_START
        + 3 * L2.STAGE_W + 2 * L2.STAGE_GAP
        + L2.BRANCH_GAP        // stage → branchX
        + L2.BRANCH_GAP        // branchX → trackStartX
        + L2.TRACK_W
        + L2.BRANCH_GAP        // → mergeX
        + L2.BRANCH_GAP        // → integX
        + L2.INTEG_W
        + L2.BRANCH_GAP        // → diamond
        + L2.EVAL_SIZE * 0.72 + 4;

    mockupRounds.forEach((round, ri) => {
        const delay = ri * 0.4;

        // Extra top padding for centered multi-track rounds
        if (round.tracks.length > 1) {
            const _ttH = round.tracks.length * L2.TRACK_H + (round.tracks.length - 1) * 16;
            cursorY += Math.max(0, (_ttH - L2.STAGE_H) / 2);
        }


        const rowCY = cursorY + L2.STAGE_H / 2;
        rowCenters.push(rowCY);

        // Round label
        const rl = _h('div', `cb2-round-lbl ${round.active ? 'active' : 'done'}`);
        rl.style.cssText = `left:${L2.LOOP_X - 15}px;top:${rowCY - 8}px;width:30px`;
        rl.textContent = `R${round.id}`;
        canvas.appendChild(rl);

        // ── STAGES ──
        let cx = L2.X_START;
        const stagePositions = [];

        round.stages.forEach((stage, si) => {
            // Prefer the stage's declared colIdx (so R2+ DEBATE reports col=2,
            // not its array position 1 which would mean MINING). Fall back to
            // si for legacy callers that haven't populated colIdx.
            const stageColIdx = (typeof stage.colIdx === 'number') ? stage.colIdx : si;
            const chipInfo = makeStageChip(canvas, cx, cursorY, stage, round.active, delay + si * 0.1, round.id, stageColIdx, _onClick, _reachable);
            stagePositions.push({ x: cx, right: cx + L2.STAGE_W, cy: rowCY, name: stage.name, el: chipInfo.el });

            // Trace between stages. The color follows the INCOMING stage's
            // status, not the round as a whole — on the active row, stages
            // before the currently-running one are already done, so their
            // incoming traces must render green (completed). Only the trace
            // leading into the one active stage gets the cyan "live" glow.
            if (si > 0) {
                const prev = stagePositions[si - 1];
                const stageActive = !!stage.active;
                const traceColor = stageActive ? '#00f3ff' : '#00ffb2';
                const tf = stageActive ? 'url(#cb2-gC)' : 'url(#cb2-gM)';
                drawTrace(svgEl, [[prev.right + 4, rowCY], [cx - 4, rowCY]], traceColor, true, delay + si * 0.2, tf);
                const viaX = prev.right + (cx - prev.right) / 2;
                let interStageGateKey = null;
                if (prev.name === 'BASELINE' && stage.name === 'MINING') interStageGateKey = 'stage_1';
                else if (prev.name === 'MINING' && stage.name === 'DEBATE') interStageGateKey = 'stage_2';
                const interStageGate = interStageGateKey && round.auditGates ? round.auditGates[interStageGateKey] : null;
                // bypass and pending keep the plain via below, drawn in the
                // stage's own trace colour rather than the hex's fixed green.
                if (interStageGate && interStageGate.state !== 'bypass' && interStageGate.state !== 'pending') {
                    drawAuditHex(canvas, svgEl, viaX, rowCY, interStageGate, interStageGateKey, round.id, delay + si * 0.2, _onAuditOpen);
                } else {
                    drawVia(svgEl, viaX, rowCY, traceColor, delay + si * 0.2);
                }
            }
            cx += L2.STAGE_W + L2.STAGE_GAP;
        });

        // ── Rationale satellite cards below the DEBATE stage chip ──
        if (round.rationales && round.rationales.length > 0) {
            const debIdx = stagePositions.findIndex(p => p.name === 'DEBATE');
            const debPos = debIdx >= 0 ? stagePositions[debIdx] : null;
            // The DEBATE chip node — we attach this round's rationale
            // reveal listeners here so the chip is the only trigger
            // surface (rather than each card being independently hovered).
            const debateChip = debPos && debPos.el ? debPos.el : null;
            if (debPos) {
                const cardW = L2.STAGE_W;
                const cardX = debPos.x;
                // Drop the card ~14px farther below the DEBATE → track trace so
                // the horizontal wire at rowCY passes cleanly above the label.
                const cardsStartY = cursorY + L2.STAGE_H + 32;
                // Stack label sits just above the cards
                const stackLbl = _h('div', 'lg-rationale-stack-label');
                stackLbl.style.cssText = `left:${cardX}px;top:${cursorY + L2.STAGE_H + 18}px;width:${cardW}px;text-align:center;`;
                stackLbl.textContent = `RATIONALE \u00d7${round.rationales.length}`;
                canvas.appendChild(stackLbl);
                // Wrap holds the full rationale stack — no scroll, all N
                // cards visible on reveal. Cards are hidden (opacity:0)
                // at rest so the visual footprint only appears on hover.
                const wrap = _h('div', 'lg-rationale-group-wrap');
                wrap.style.cssText =
                    `position:absolute;z-index:5;left:${cardX}px;top:${cardsStartY}px;` +
                    `width:${cardW}px;`;
                canvas.appendChild(wrap);
                const cards = round.rationales.map((rat, i) => {
                    const card = makeRationaleCard(wrap, null, null, cardW, rat,
                        delay + 0.6 + i * 0.1, (r) => _onOpen(r.path));
                    // Staggered cascade on reveal only (25ms per card).
                    // Inline `--lg-stagger` is read by the --revealed rule.
                    card.style.setProperty('--lg-stagger', `${i * 25}ms`);
                    return card;
                });

                _wireRationaleReveal({
                    chip: debateChip,
                    wrap,
                    cards,
                    stackLbl,
                });
            }
        }

        // ── BRANCH + TRACKS ──
        const lastStage = stagePositions[stagePositions.length - 1];
        if (!lastStage) { cursorY += L2.ROW_SPACING; return; }
        const branchX = lastStage.right + L2.BRANCH_GAP;

        if (round.tracks.length === 0 && round.eval) {
            // Pre-track stages (1-3) with a verdict already decided (e.g. a
            // pre-emptive stop). Trace from last stage → diamond at EVAL_CX.
            const evalCX = EVAL_CX;
            drawTrace(svgEl, [[lastStage.right + 4, rowCY], [evalCX - L2.EVAL_SIZE * 0.72 - 4, rowCY]],
                '#00f3ff', true, delay + 0.8, 'url(#cb2-gC)');
            makeEvalDiamond(canvas, evalCX, rowCY, round.eval, delay + 0.5, round.id, _onClick, round.evalNote, _reachable);
            evalPositions[ri] = evalCX + L2.EVAL_SIZE * 0.72;
        } else if (round.tracks.length === 0) {
            // No tracks yet AND no verdict — the row is still being built.
            // Leave the trace terminated at the last stage chip; do NOT run a
            // dead line across empty canvas to the eval column.
        } else if (round.tracks.length > 0) {
            const trackStartX = branchX + L2.BRANCH_GAP;
            const trackCount = round.tracks.length;
            const trackVGap = 16;

            // Center the track stack around rowCY
            const totalTrackH = trackCount * L2.TRACK_H + Math.max(0, trackCount - 1) * trackVGap;
            const trackBlockTop = trackCount > 1
                ? rowCY - totalTrackH / 2
                : cursorY + (L2.STAGE_H - L2.TRACK_H) / 2;

            const trackCYs = [];
            round.tracks.forEach((track, ti) => {
                const trackY = trackBlockTop + ti * (L2.TRACK_H + trackVGap);
                const trackCY = trackY + L2.TRACK_H / 2;
                trackCYs.push(trackCY);
                makeTrackChip(canvas, trackStartX, trackY, track, delay + 0.3 + ti * 0.1, round.id, _onClick, round.active);
            });

            // Fan-out/fan-in origin = rowCY (center of stack = center of stages)
            const stackCY = rowCY;

            // Trunk: stage → branchX (horizontal at rowCY, then vertical jog to stackCY)
            if (stackCY !== rowCY) {
                drawTrace(svgEl, [[lastStage.right + 4, rowCY], [branchX, rowCY], [branchX, stackCY]], '#00ffb2', true, delay + round.stages.length * 0.2, 'url(#cb2-gM)');
            } else {
                drawTrace(svgEl, [[lastStage.right + 4, rowCY], [branchX, rowCY]], '#00ffb2', true, delay + round.stages.length * 0.2, 'url(#cb2-gM)');
            }
            drawVia(svgEl, branchX, stackCY, round.active ? '#00f3ff' : '#00ffb2', delay + 0.5);

            // Fan-out traces from stackCY to each track. End ~10px before the
            // chip's left edge so the trace+glow terminates cleanly and doesn't
            // bleed behind the chip's semi-transparent background.
            const TRACK_TRACE_GAP = 10;
            round.tracks.forEach((track, ti) => {
                const trackCY = trackCYs[ti];
                const tColor = track.status === 'failed' ? '#ff3355' : track.status === 'active' ? '#00f3ff' : '#00ffb2';
                const tFilter = track.status === 'failed' ? 'url(#cb2-gR)' : 'url(#cb2-gM)';
                if (trackCY !== stackCY) {
                    drawTrace(svgEl, [[branchX, stackCY], [branchX, trackCY], [trackStartX - TRACK_TRACE_GAP, trackCY]], tColor, true, delay + 0.6 + ti * 0.12, tFilter);
                } else {
                    drawTrace(svgEl, [[branchX, stackCY], [trackStartX - TRACK_TRACE_GAP, trackCY]], tColor, true, delay + 0.6 + ti * 0.12, tFilter);
                }
            });

            // ── MERGE + INTEGRATION ──
            const mergeX = trackStartX + L2.TRACK_W + L2.BRANCH_GAP;

            // The S45 hex renders whenever the round has tracks, NOT only when
            // integration exists. T_AUDIT_S45 fires once every track is
            // terminal and before integration starts, so gating it on
            // round.integration would hide the audit for its whole live window.
            const mergeGate = round.auditGates ? round.auditGates.stage_45 : null;
            const mergeGateLive = !!mergeGate && mergeGate.state !== 'bypass' && mergeGate.state !== 'pending';
            if (mergeGateLive) {
                drawAuditHex(canvas, svgEl, mergeX, stackCY, mergeGate, 'stage_45', round.id, delay + 0.9, _onAuditOpen);
            } else if (round.integration) {
                drawVia(svgEl, mergeX, stackCY, '#00ffb2', delay + 0.9);
            }

            if (round.integration) {
                // Fan-in traces from each shipped track to stackCY. Leave a
                // matching 10px gap on the chip's right edge to mirror the
                // inbound trace termination and avoid under-chip glow bleed.
                const TRACK_OUT_GAP = 10;
                round.tracks.forEach((track, ti) => {
                    if (track.status === 'shipped') {
                        const trackCY = trackCYs[ti];
                        if (trackCY !== stackCY) {
                            drawTrace(svgEl, [[trackStartX + L2.TRACK_W + TRACK_OUT_GAP, trackCY], [mergeX, trackCY], [mergeX, stackCY]], '#00ffb2', true, delay + 0.8, 'url(#cb2-gM)');
                        } else {
                            drawTrace(svgEl, [[trackStartX + L2.TRACK_W + TRACK_OUT_GAP, trackCY], [mergeX, stackCY]], '#00ffb2', true, delay + 0.8, 'url(#cb2-gM)');
                        }
                    }
                });

                // Merge → Integration (vertical jog from stackCY to rowCY, then horizontal)
                const integX = mergeX + L2.BRANCH_GAP;
                const integY = cursorY + (L2.STAGE_H - L2.INTEG_H) / 2;
                makeIntegChip(canvas, integX, integY, round.integration, delay + 0.4, round.id, _onClick, round.active, _reachable);

                // Resolution badges below integration chip
                if (round.resolutions && round.resolutions.length > 0) {
                    const badgeW = 200;
                    const badgeX = integX + (L2.INTEG_W - badgeW) / 2;
                    const badgeStartY = integY + L2.INTEG_H + 14;
                    round.resolutions.slice(0, 3).forEach((res, i) => {
                        makeResolutionBadge(canvas, badgeX, badgeStartY + i * 52, badgeW, res,
                            delay + 0.9 + i * 0.1, (r) => _onOpen(r.path));
                    });
                }
                if (stackCY !== rowCY) {
                    drawTrace(svgEl, [[mergeX, stackCY], [mergeX, rowCY], [integX - 4, rowCY]], '#ffaa00', true, delay + 1.0, 'url(#cb2-gA)');
                } else {
                    drawTrace(svgEl, [[mergeX, stackCY], [integX - 4, rowCY]], '#ffaa00', true, delay + 1.0, 'url(#cb2-gA)');
                }

                // Always run the trace from integration chip → EVAL_CX; diamond
                // renders only when a terminal verdict is present.
                const evalCX = EVAL_CX;
                drawTrace(svgEl, [[integX + L2.INTEG_W + 4, rowCY], [evalCX - L2.EVAL_SIZE * 0.72 - 4, rowCY]], '#00f3ff', true, delay + 1.2, 'url(#cb2-gC)');
                const integEvalGate = round.auditGates ? round.auditGates.stage_67 : null;
                if (integEvalGate && integEvalGate.state !== 'bypass' && integEvalGate.state !== 'pending') {
                    const gateMidX = (integX + L2.INTEG_W + 4 + (evalCX - L2.EVAL_SIZE * 0.72 - 4)) / 2;
                    drawAuditHex(canvas, svgEl, gateMidX, rowCY, integEvalGate, 'stage_67', round.id, delay + 1.2, _onAuditOpen);
                }
                if (round.eval) {
                    makeEvalDiamond(canvas, evalCX, rowCY, round.eval, delay + 0.5, round.id, _onClick, round.evalNote, _reachable);
                    evalPositions[ri] = evalCX + L2.EVAL_SIZE * 0.72;
                } else {
                    drawTerminusTick(svgEl, evalCX - L2.EVAL_SIZE * 0.72 - 4, rowCY, '#00f3ff', delay + 1.3);
                }
            } else if (round.eval) {
                // Tracks present, no integration, but an eval verdict is
                // already decided — run the trace from the track block's
                // right edge to the eval diamond at EVAL_CX.
                const evalCX = EVAL_CX;
                const traceColor = '#ff3355';
                const traceFilter = 'url(#cb2-gR)';
                drawTrace(svgEl, [[trackStartX + L2.TRACK_W + 4, rowCY], [evalCX - L2.EVAL_SIZE * 0.72 - 4, rowCY]], traceColor, true, delay + 0.9, traceFilter);
                makeEvalDiamond(canvas, evalCX, rowCY, round.eval, delay + 0.5, round.id, _onClick, null, _reachable);
                evalPositions[ri] = evalCX + L2.EVAL_SIZE * 0.72;
            }
            // Otherwise: tracks are still running AND there is no integration
            // AND no eval verdict — the campaign hasn't reached the IMPL→EVAL
            // transition yet. Terminate the row at the track block's right
            // edge rather than drawing a premature connector across the
            // unreached EVAL and INTEGRATION columns. Fan-out traces already
            // terminate at each track chip; no trailing trace is needed.
        }

        // Dynamic row spacing: account for centered track block
        const _tc = round.tracks.length;
        // Row bottom = the lowest pixel occupied by this row's chips. Feedback
        // connectors must route BELOW this y to avoid slicing through a
        // track chip body when the row has multi-track centering.
        let _rowBottom = cursorY + L2.STAGE_H;
        if (_tc > 1) {
            const _totalTH = _tc * L2.TRACK_H + (_tc - 1) * 16;
            const _extraBelow = Math.max(0, (_totalTH - L2.STAGE_H) / 2);
            _rowBottom = rowCY + _totalTH / 2;
            cursorY += L2.ROW_SPACING + _extraBelow;
        } else if (_tc === 1) {
            _rowBottom = cursorY + (L2.STAGE_H + L2.TRACK_H) / 2;
            cursorY += L2.ROW_SPACING;
        } else {
            cursorY += L2.ROW_SPACING;
        }
        rowBottoms.push(_rowBottom);
    });

    // ── LOOP CONNECTORS (vertical left side) ──
    for (let i = 0; i < rowCenters.length - 1; i++) {
        const fromY = rowCenters[i];
        const toY = rowCenters[i + 1];
        const isToActive = mockupRounds[i + 1] && mockupRounds[i + 1].active;
        const color = isToActive ? '#00f3ff' : 'rgba(0,243,255,0.3)';

        const d = `M${L2.LOOP_X},${fromY + 20} L${L2.LOOP_X},${toY - 20}`;
        _svgEl('path', { d, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-dasharray': '4 8', opacity: 0.3, 'stroke-linecap': 'round', 'pointer-events': 'none' }, svgEl);
        const fp = _svgEl('path', { d, fill: 'none', stroke: color, 'stroke-width': 1.5, 'stroke-dasharray': '3 9', opacity: 0.6, 'stroke-linecap': 'round', 'pointer-events': 'none' }, svgEl);
        fp.style.animation = 'cb2FlowDashDown 1.2s linear infinite';
        // Arrow
        const arrowY = toY - 22;
        _svgEl('path', { d: `M${L2.LOOP_X - 5},${arrowY} L${L2.LOOP_X},${arrowY + 8} L${L2.LOOP_X + 5},${arrowY}`, fill: 'none', stroke: color, 'stroke-width': 1.5, 'stroke-linecap': 'round', 'pointer-events': 'none' }, svgEl);
        drawVia(svgEl, L2.LOOP_X, fromY + 20, color, 2.2 + i * 0.2);
        drawVia(svgEl, L2.LOOP_X, toY - 20, color, 2.2 + i * 0.2);
        // Label on first connector
        if (i === 0) {
            const lbl = _svgEl('text', { x: L2.LOOP_X, y: (fromY + toY) / 2, 'text-anchor': 'middle', fill: 'rgba(0,243,255,0.15)', 'font-family': "'Orbitron', sans-serif", 'font-size': '8', 'letter-spacing': '2', 'font-weight': '600', transform: `rotate(-90 ${L2.LOOP_X} ${(fromY + toY) / 2})`, 'pointer-events': 'none' }, svgEl);
            lbl.textContent = 'OPTIMIZATION LOOP';
        }
    }

    // ── Feedback connectors from eval diamonds ──
    // The horizontal leg must route BELOW the row's track block — otherwise
    // a row with 3 centered tracks will see this line slice through a track
    // chip body (because the ROW_SPACING midpoint falls inside the lower
    // track's vertical band). `rowBot` is the authoritative bottom of the
    // track block; park the horizontal 14px below it.
    for (let i = 0; i < mockupRounds.length - 1; i++) {
        if (!evalPositions[i]) continue;
        const rowCY = rowCenters[i];
        const nextRCY = rowCenters[i + 1];
        const rowBot = rowBottoms[i] ?? rowCY + L2.STAGE_H / 2;
        const nextTop = nextRCY - L2.STAGE_H / 2;  // approx top of next row's stage chips
        // Target 14px below this row's bottom. If that would crash into the
        // next row's chips, split the difference between rowBot and nextTop.
        const desired = rowBot + 14;
        const midY = desired < nextTop - 10 ? desired : (rowBot + nextTop) / 2;
        const evalRightX = evalPositions[i] + 6;
        const dropX = Math.min(evalRightX + 20, L2.SVG_W - 30);
        drawTrace(svgEl, [[evalRightX, rowCY], [dropX, rowCY], [dropX, midY], [L2.LOOP_X, midY]], 'rgba(0,243,255,0.2)', true, 2 + i * 0.25, null);
        drawVia(svgEl, dropX, rowCY, 'rgba(0,243,255,0.2)', 2.2 + i * 0.25);
        drawVia(svgEl, dropX, midY, 'rgba(0,243,255,0.2)', 2.3 + i * 0.25);
    }

    // Set final dimensions
    const totalH = cursorY + 40;
    canvas.style.height = totalH + 'px';
    svgEl.setAttribute('width', L2.SVG_W);
    svgEl.setAttribute('height', totalH);
    svgEl.setAttribute('viewBox', `0 0 ${L2.SVG_W} ${totalH}`);

    return canvas;
}

// ── Sidebar (kept as no-op) ──────────────────────────────────────────────────

function buildSidebar(state, rounds) {
    const div = document.createElement('div');
    div.className = 'cb-sidebar';
    div.style.display = 'none';
    return div;
}

function _esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function _statusColor(status) {
    const s = (status || '').toLowerCase();
    if (s === 'active') return 'var(--cyan)';
    if (s === 'complete' || s === 'campaign_complete') return 'var(--mint)';
    if (s === 'failed' || s === 'campaign_exhausted' || s === 'exhausted') return 'var(--red)';
    return 'var(--dim)';
}

// ── Catalog enrichment ────────────────────────────────────────────────────────

function _enrichFromCatalog(state, tree) {
    state._catalog = {};

    // Sidecar-removal (2026-05-27): this function previously consumed a flat
    // path → sidecar-entry dict (with metrics/labels/refs blobs). Now the
    // server only ships a tree manifest (`{root, files: [path, ...]}`) plus
    // the orchestrator-enriched `state.json`. Per-bucket sourcing:
    //
    //   baselineByRound  ← state.campaign.rounds[N].baseline.e2e_latency
    //                       (+ legacy integration.e2e_latency_combined / opt_s
    //                        carry-forward for re-profile chips)
    //   miningByRound    ← state.campaign.rounds[N].bottleneck_mining
    //   debateByRound    ← state.campaign.rounds[N].debate (winners,
    //                       champions_count, selected_candidates, rounds_completed)
    //                       + tree.files scan for rationale-count tiebreak
    //   rationales       ← tree.files: rounds/{N}/debate/round_*/<champ>_<stance>*.md
    //   resolutions      ← tree.files: rounds/{N}/integration/diff*.{patch|diff}
    //   hasReport        ← tree.files contains rounds/{N}/REPORT.md or report/*

    // Tree files (paths only — no metadata).
    let files = [];
    if (tree && Array.isArray(tree.files)) {
        files = tree.files.filter(p => typeof p === 'string');
    } else if (Array.isArray(tree)) {
        files = tree.filter(p => typeof p === 'string');
    } else if (tree && typeof tree === 'object') {
        // Legacy sidecar dict — keys are paths. Used for backward-compat
        // until all callers pass tree shape.
        files = Object.keys(tree);
    }
    files = files.filter(p =>
        p && !p.startsWith('_archive/') && !p.includes('/__pycache__/') && !p.endsWith('.metrics.json'),
    );

    // Path-derived metadata helper (shared with campaign-app.js so labels stay
    // in sync). Used for debate-rationale enumeration where we need stance +
    // champion-id from the file name.
    const _PA = (typeof window !== 'undefined' && window.LG_HELPERS && window.LG_HELPERS.parseArtifactPath)
        ? window.LG_HELPERS.parseArtifactPath
        : null;
    const _parseFor = (k) => {
        if (_PA) return _PA(k);
        return { round: null, stage: null, track_id: null, kind: null, op_id: null, champion_id: null, stance: null };
    };

    // ── Baseline / re-profile latency, indexed by round ──────────────────────
    // state.campaign.rounds[N].baseline.e2e_latency is the authoritative
    // {bucket_tag: {avg, p50, p10, ...}} map. Legacy v3 carry-forward paths kept for
    // re-profile chips on R(N+1) when the next round hasn't re-profiled yet.
    state._catalog.baselineByRound = {};

    const campaignRoundsForBaseline = ((state.campaign || {}).rounds) || [];

    // (1) Primary — baseline.e2e_latency map
    for (const rr of campaignRoundsForBaseline) {
        const rid = Number(rr.round_id ?? rr.round);
        if (!Number.isFinite(rid)) continue;
        const lat = rr?.baseline?.e2e_latency;
        if (!lat || typeof lat !== 'object') continue;
        const primaryBucket = _bucketRecords(lat)[0];
        if (!primaryBucket) continue;
        const primaryEntry = lat[primaryBucket.tag];
        if (!primaryEntry || typeof primaryEntry.avg !== 'number' || primaryEntry.avg <= 0) continue;
        state._catalog.baselineByRound[rid] = {
            primaryBsLatencyMs: primaryEntry.avg * 1000,
            primaryBs: primaryBucket.batchSize,
            primaryBucket,
            batchSizes: lat,
            perBsVerdict: rr?.baseline?.per_bs_verdict || null,
            emittedAt: null,
            source: 'baseline_e2e_latency',
        };
    }

    // (1b) Past-round carry-forward — for any R(N) that lacks a primary
    //      baseline entry but where R(N-1)'s integration produced an
    //      e2e_latency_combined map, the R(N) re-profile chip displays
    //      the carried-forward latency. Walk in ascending round_id order so
    //      "previous round" is always already populated.
    const sortedRoundsAsc = [...campaignRoundsForBaseline]
        .map(rr => ({ rr, rid: Number(rr.round_id ?? rr.round) }))
        .filter(x => Number.isFinite(x.rid))
        .sort((a, b) => a.rid - b.rid);
    for (const { rr, rid } of sortedRoundsAsc) {
        if (rid <= 1) continue;
        if (state._catalog.baselineByRound[rid]) continue;
        const priorRound = sortedRoundsAsc.find(x => x.rid === rid - 1)?.rr;
        if (!priorRound) continue;
        const integ = priorRound.integration || {};
        const elc = integ.e2e_latency_combined;
        if (elc && typeof elc === 'object') {
            const primaryBucket = _bucketRecords(elc)[0];
            const entry = primaryBucket ? elc[primaryBucket.tag] : null;
            if (entry && typeof entry.avg === 'number' && entry.avg > 0) {
                state._catalog.baselineByRound[rid] = {
                    primaryBsLatencyMs: entry.avg * 1000,
                    primaryBs: primaryBucket.batchSize,
                    primaryBucket,
                    batchSizes: elc,
                    perBsVerdict: integ.per_bs_verdict || null,
                    emittedAt: null,
                    source: 'integration_e2e_latency_combined',
                    derivedFromRound: rid - 1,
                };
            }
        }
    }

    // (2) Carry-forward — integration.e2e_latency_combined for R(N) becomes
    //     the R(N+1) baseline when R(N+1) hasn't been re-profiled yet.
    //     Skip when R(N+1) already has a primary entry.
    const seenRids = new Set(campaignRoundsForBaseline.map(rr => Number(rr.round_id ?? rr.round)).filter(Number.isFinite));
    const maxRid = seenRids.size ? Math.max(...seenRids) : null;
    if (maxRid != null) {
        const carryRid = maxRid + 1;
        if (!state._catalog.baselineByRound[carryRid]) {
            const priorRound = campaignRoundsForBaseline.find(
                rr => Number(rr.round_id ?? rr.round) === maxRid,
            );
            const integ = priorRound?.integration || {};
            const elc = integ.e2e_latency_combined;
            if (elc && typeof elc === 'object') {
                const primaryBucket = _bucketRecords(elc)[0];
                const entry = primaryBucket ? elc[primaryBucket.tag] : null;
                if (entry && typeof entry.avg === 'number' && entry.avg > 0) {
                    state._catalog.baselineByRound[carryRid] = {
                        primaryBsLatencyMs: entry.avg * 1000,
                        primaryBs: primaryBucket.batchSize,
                        primaryBucket,
                        batchSizes: elc,
                        perBsVerdict: integ.per_bs_verdict || null,
                        emittedAt: null,
                        source: 'integration_e2e_latency_combined',
                        derivedFromRound: maxRid,
                    };
                }
            } else {
                // (3) Legacy scalar — combined_e2e_result.opt_s / latency_baseline_s
                const cer = integ.combined_e2e_result;
                const scalar = cer?.opt_s ?? cer?.latency_baseline_s;
                if (typeof scalar === 'number' && Number.isFinite(scalar) && scalar > 0) {
                    const perBs = cer?.per_bs_verdict || {};
                    const primaryBucket = _bucketRecords(perBs)[0] || null;
                    state._catalog.baselineByRound[carryRid] = {
                        primaryBsLatencyMs: scalar * 1000,
                        primaryBs: primaryBucket?.batchSize ?? null,
                        primaryBucket,
                        batchSizes: {},
                        perBsVerdict: perBs,
                        emittedAt: null,
                        source: 'integration_opt_s',
                        derivedFromRound: maxRid,
                    };
                }
            }
        }
    }

    // (3) Legacy fallback in the same round — v3 campaigns pre-normalizer
    //     have integration.combined_e2e_result.latency_baseline_s but no
    //     baseline.e2e_latency. The backend normalizer synthesizes the map,
    //     but some pre-render paths bypass it — guard here too.
    for (const rr of campaignRoundsForBaseline) {
        const rid = Number(rr.round_id ?? rr.round);
        if (!Number.isFinite(rid)) continue;
        if (state._catalog.baselineByRound[rid]) continue;
        const cer = rr?.integration?.combined_e2e_result;
        const latS = cer?.latency_baseline_s;
        if (typeof latS !== 'number' || !Number.isFinite(latS) || latS <= 0) continue;
        const perBs = cer?.per_bs_verdict || {};
        const primaryBucket = _bucketRecords(perBs)[0] || null;
        state._catalog.baselineByRound[rid] = {
            primaryBsLatencyMs: latS * 1000,
            primaryBs: primaryBucket?.batchSize ?? null,
            primaryBucket,
            batchSizes: {},
            perBsVerdict: perBs,
            emittedAt: null,
            source: 'compat_fallback',
        };
    }

    // ── Mining (bottleneck), indexed by round ─────────────────────────────────
    // Read directly from state.campaign.rounds[N].bottleneck_mining. Track 1
    // expanded the schema to accept top_component / top_f_decode_pct /
    // amdahl_ceiling so the orchestrator writes them post-T2; this layer just
    // reads what's there.
    state._catalog.miningByRound = {};
    for (const rr of campaignRoundsForBaseline) {
        const rid = Number(rr.round_id ?? rr.round);
        if (!Number.isFinite(rid)) continue;
        const bm = rr.bottleneck_mining;
        if (!bm || typeof bm !== 'object') continue;
        if (!(bm.top_component || bm.top_f_decode_pct || bm.top_bottleneck_share_pct)) continue;
        // Coerce component_breakdown entries to {name, pct}. Accept either
        // explicit {name, pct} or positional [name, pct] tuples.
        let breakdown = null;
        const rawBreakdown = bm.component_breakdown || bm.components || null;
        if (Array.isArray(rawBreakdown)) {
            breakdown = rawBreakdown.map(e => {
                if (Array.isArray(e) && e.length >= 2) return { name: String(e[0]), pct: Number(e[1]) };
                if (e && typeof e === 'object') {
                    const name = e.name || e.component || e.kernel || null;
                    const pct  = e.pct ?? e.share ?? e.share_pct ?? e.percent ?? null;
                    if (name && Number.isFinite(Number(pct))) return { name: String(name), pct: Number(pct) };
                }
                return null;
            }).filter(Boolean);
            if (!breakdown.length) breakdown = null;
        }
        state._catalog.miningByRound[rid] = {
            component: bm.top_component || null,
            pct: bm.top_f_decode_pct ?? bm.top_bottleneck_share_pct ?? null,
            amdahlCeiling: bm.amdahl_ceiling ?? null,
            decodeFrac: typeof bm.decode_frac === 'number' ? bm.decode_frac : null,
            componentBreakdown: breakdown,
            emitter: bm.emitter || null,
            sourcePath: `rounds/${rid}/mining/bottleneck_analysis.md`,
        };
    }

    // ── Debate, per campaign round ────────────────────────────────────────────
    // State is the source of truth: state.campaign.rounds[N].debate provides
    //   selected_winners / winners
    //   champions_count
    //   selected_candidates[] (with score_breakdown for D1 scoreboard tooltip)
    //   rounds_completed
    //   result
    //
    // The tree is scanned only to compute rationalesCount (artifact volume —
    // a tiebreak signal showing how much debate happened beyond what state
    // captures). Track 1 added the Track-1 schema so champions_count / winners
    // / result land directly in state.json.
    state._catalog.debateByRound = {};

    // Tree-derived rationale count per round.
    const _rationaleCountByRound = {};
    const _championsSeenByRound = {};
    for (const path of files) {
        // rounds/{N}/debate/round_*/<champ>_<stance>*.md → debate_rationale
        const m = path.match(/^rounds\/(\d+)\/debate\/round_\d+\//);
        if (!m) continue;
        const parsed = _parseFor(path);
        if (parsed.kind !== 'debate_rationale') continue;
        const rid = Number(m[1]);
        if (!Number.isFinite(rid)) continue;
        _rationaleCountByRound[rid] = (_rationaleCountByRound[rid] || 0) + 1;
        if (parsed.champion_id) {
            const set = _championsSeenByRound[rid] = _championsSeenByRound[rid] || new Set();
            set.add(parsed.champion_id);
        }
    }

    for (const rr of campaignRoundsForBaseline) {
        const rid = Number(rr.round_id ?? rr.round);
        if (!Number.isFinite(rid)) continue;
        const deb = rr.debate || {};
        const winners = Array.isArray(deb.winners) && deb.winners.length
            ? deb.winners
            : (Array.isArray(deb.selected_winners) ? deb.selected_winners : []);
        // Track 1 schema: deb.champions_count is canonical. Fall back to
        // path-derived champion set when state hasn't been written yet.
        const championsCountFromState = typeof deb.champions_count === 'number'
            ? deb.champions_count : null;
        const championsCountFromTree = _championsSeenByRound[rid]
            ? _championsSeenByRound[rid].size : null;
        const championsCount = championsCountFromState ?? championsCountFromTree ?? null;
        const rationalesCount = _rationaleCountByRound[rid] || 0;
        const summaryRoundsCompleted = (typeof deb.rounds_completed === 'number')
            ? deb.rounds_completed : null;

        // Skip rounds with no debate signal at all (rationales count tells us
        // the orchestrator hasn't reached debate yet on this round).
        if (!winners.length && championsCount == null && rationalesCount === 0
            && !Array.isArray(deb.selected_candidates)) {
            continue;
        }

        const bucket = {
            championsCount,
            rationalesCount,
            winners: winners.slice(),
            summaryRoundsCompleted,
        };

        // Per-candidate scoreboard for the D1 tooltip — read directly from
        // state.campaign.rounds[N].debate.selected_candidates[].
        const sel = deb.selected_candidates;
        if (Array.isArray(sel) && sel.length) {
            const winnerSet = new Set(winners);
            bucket.candidates = sel.map(c => {
                const sb = c.score_breakdown || {};
                return {
                    opId: c.op_id || null,
                    displayName: (typeof c.name === 'string' && c.name.trim()) ? c.name.trim() : null,
                    trackAssignment: c.track_assignment || null,
                    feasibility: typeof sb.feasibility === 'number' ? sb.feasibility : null,
                    evidenceTier: sb.evidence_tier || null,
                    expectedE2ePct: typeof sb.expected_e2e_pct === 'number' ? sb.expected_e2e_pct : null,
                    weightedTotal: typeof sb.weighted_total === 'number' ? sb.weighted_total : null,
                    isWinner: c.op_id ? winnerSet.has(c.op_id) : false,
                };
            });
        }

        // Optional: surface scoreboard from Track 1 schema field (free-form
        // object shaped by the orchestrator). Only attached when present so
        // existing tooltip code that branches on `bucket.scoreboard` keeps
        // working.
        if (deb.scoreboard && typeof deb.scoreboard === 'object') {
            bucket.scoreboard = deb.scoreboard;
        }
        if (typeof deb.result === 'string' && deb.result) {
            bucket.result = deb.result;
        }

        state._catalog.debateByRound[rid] = bucket;
    }

    // ── Rationales / Resolutions / Report — tree-only enumeration ─────────────
    // Sidecar removal: the orchestrator no longer ships description blobs;
    // we just emit the path so the renderer can fetch markdown on click.
    state._catalog.rationales = []; // [{round, opId, stance, championId, description, path}]
    state._catalog.resolutions = []; // [{resolutionType, headSha, description, path, round}]
    state._catalog.hasReport = false;

    for (const path of files) {
        const parsed = _parseFor(path);
        const kind = parsed.kind;
        if (!kind) {
            // Even unparsed paths can satisfy hasReport (defensive — REPORT.md
            // sits at rounds/{N}/REPORT.md or report/*; parseArtifactPath
            // already maps both to kind=report_section, so this branch is rare).
            continue;
        }
        if (kind === 'debate_rationale') {
            state._catalog.rationales.push({
                round: parsed.round ?? null,
                opId: parsed.op_id || parsed.track_id || null,
                stance: (parsed.stance || 'proposed').toLowerCase(),
                championId: parsed.champion_id || null,
                description: '',  // body fetched on click; sidecar metadata removed
                path,
            });
        } else if (kind === 'diff' && !parsed.op_id) {
            // Track diffs (per-track impl) carry op_id; only resolver diffs
            // (round-level integration/diff*.patch) are true resolutions.
            state._catalog.resolutions.push({
                resolutionType: 'merge',
                headSha: null,
                description: '',
                path,
                round: parsed.round ?? null,
            });
        } else if (kind === 'report_section') {
            state._catalog.hasReport = true;
        }
    }
}

// Stage duration for a specific round. Reads per-round stage sub-objects
// (campaign.rounds[roundId-1].{stage}.{started_at,completed_at}) via
// STAGE_KEY_MAP. Returns `null` when either timestamp is missing.
function _stageElapsed(state, stageKey, roundId) {
    const campaign = (state && state.campaign) || {};
    const rounds = campaign.rounds || [];
    const idx = ((roundId != null) ? roundId : (campaign.current_round || 1)) - 1;
    if (idx < 0 || idx >= rounds.length) return null;
    const round = rounds[idx] || {};
    const subKey = CB.STAGE_KEY_MAP[stageKey];
    if (!subKey) return null;
    const entry = round[subKey];
    if (!entry || !entry.started_at || !entry.completed_at) return null;
    const ms = new Date(entry.completed_at) - new Date(entry.started_at);
    if (isNaN(ms) || ms < 0) return null;
    const secs = Math.round(ms / 1000);
    if (secs < 60) return secs + 's';
    const mins = Math.floor(secs / 60);
    if (mins < 60) {
        const rem = secs % 60;
        return mins + 'm ' + String(rem).padStart(2, '0') + 's';
    }
    const hrs = Math.floor(mins / 60);
    const remMins = mins % 60;
    return hrs + 'h ' + String(remMins).padStart(2, '0') + 'm';
}

// ── Public API ─────────────────────────────────────────────────────────────────

function renderCircuitBoard(container, state, onNodeClick, catalog, onArtifactOpen, reachable, onAuditOpen) {
    container.innerHTML = '';
    container.style.cssText = 'flex:1;overflow:auto;position:relative;scrollbar-width:thin;scrollbar-color:rgba(0,243,255,0.2) transparent;';

    const rounds = parseRounds(state);
    if (rounds.length === 0) {
        container.innerHTML = `
          <div style="display:flex;flex:1;align-items:center;justify-content:center;
                      flex-direction:column;gap:12px;color:var(--ghost);min-height:200px;">
            <div style="font-family:var(--font-head);font-size:10px;letter-spacing:2px;">NO CAMPAIGN DATA</div>
          </div>
        `;
        return;
    }

    _enrichFromCatalog(state, catalog);
    const mockupData = mapRoundsToMockup(state, rounds);
    _initTooltip(container);

    // Inject tooltip CSS
    const styleTag = document.createElement('style');
    styleTag.textContent = CB2_CSS;
    container.appendChild(styleTag);

    // Build hybrid board (HTML chips + SVG traces)
    const board = buildBoard(mockupData, onNodeClick || (() => {}), onArtifactOpen, reachable, onAuditOpen);
    container.appendChild(board);
}

// Export for campaign-app.js
window.renderCircuitBoard = renderCircuitBoard;

// ── buildL2Nodes: expose node list for testing + Alpine templates ─────────────

function buildL2Nodes(state) {
    const campaign = state.campaign || {};
    const opNames  = opDisplayNames(state);
    const debate   = currentDebate(state);
    const tracks   = currentTracks(state);
    const integration = currentIntegration(state);
    const shippedOps = new Set((campaign.shipped_optimizations || [])
        .map(s => typeof s === 'string' ? s : (s && s.op_id) || null)
        .filter(Boolean));
    const nodes = [];

    nodes.push({ id: 'baseline', tier: 0, label: 'BASELINE', status: 'active', detail: null });
    nodes.push({ id: 'mining',   tier: 1, label: 'MINING',   status: 'active', detail: null });

    const candidates = debate.candidates && debate.candidates.length > 0
        ? debate.candidates
        : Object.keys(tracks);
    if (candidates.length > 0) {
        const winners = new Set(debate.selected_winners || []);
        candidates.forEach(raw => {
            const cand = typeof raw === 'string' ? raw : raw.op_id;
            nodes.push({
                id: `debate_${cand}`, tier: 2, label: opNames.get(cand) || cand,
                status: winners.has(cand) ? 'active' : 'pending', detail: null,
            });
        });
    } else {
        nodes.push({ id: 'debate', tier: 2, label: 'DEBATE', status: 'pending', detail: null });
    }

    const trackIds = Object.keys(tracks);
    if (trackIds.length > 0) {
        trackIds.forEach(opId => {
            const track = tracks[opId];
            let st;
            if (shippedOps.has(opId))                              st = 'shipped';
            else if ((track.status || '').toUpperCase() === 'FAILED')  st = 'failed';
            else                                                    st = 'active';
            const kScalar = _kernelSpeedupFromTrack(track);
            const detail = kScalar != null
                ? `${kScalar.toFixed(2)}x`
                : (track.fail_reason ? track.fail_reason.slice(0, 30) : null);
            nodes.push({ id: opId, tier: 3, label: opNames.get(opId) || opId, status: st, detail });
        });
    } else {
        nodes.push({ id: 'tracks', tier: 3, label: 'TRACKS', status: 'pending', detail: null });
    }

    const intSt = integration.status === 'completed' ? 'shipped'
                : integration.status === 'failed'    ? 'failed' : 'pending';
    nodes.push({ id: 'integration', tier: 5, label: 'INTEGRATION', status: intSt, detail: null });

    return nodes;
}

function nodeStatusClassFull(opId, track, shippedOps) {
    if (shippedOps && shippedOps.includes(opId)) return 'shipped';
    const s = (track && track.status) ? track.status.toUpperCase() : '';
    if (s === 'FAILED') return 'failed';
    if (s === 'IN_PROGRESS' || s === 'COMPLETED') return 'active';
    return nodeStatusClass(track && track.status || '');
}

// ── Global export for tests ───────────────────────────────────────────────────
const CircuitBoard = {
    tierY: rowY,
    trackX,
    rightAngleTrace,
    buildL2Nodes,
    auditGateStates,
    dedupRoundsById,
    parseRounds,
    nodeStatusClass,
    nodeStatusClassFull,
    nodeStatusColor,
    renderCircuitBoard,
    enrichFromCatalog:   _enrichFromCatalog,
    kernelSpeedupScalar: _kernelSpeedupScalar,
    kernelSpeedupFromTrack: _kernelSpeedupFromTrack,
    e2eSpeedupScalar:    _e2eSpeedupScalar,
    amdahlPredictionPp:  _amdahlPredictionPp,
    // v4.0 e2e-latency helpers (schema map-based)
    primaryBsKey:             _primaryBsKey,
    bsRange:                  _bsRange,
    verdictAggregate:         _verdictAggregate,
    synthesizeReprofileVerdict: _synthesizeReprofileVerdict,
    prevIntegrationDisabled:  _prevIntegrationDisabled,
    fmtLatency:               _fmtLatency,
    computeDeltaPpFromMaps:   _computeDeltaPpFromMaps,
    renderPercentileTable:    _renderPercentileTable,
    buildRichTooltipBody:     _buildRichTooltipBody,
    buildMiningTooltipBody:   _buildMiningTooltipBody,
    buildDebateTooltipBody:   _buildDebateTooltipBody,
    baselineTooltipHtml:      _baselineTooltipHtml,
    // v2 round accessors
    currentTracks, currentIntegration, currentDebate, currentStage,
    stageElapsed: _stageElapsed,
    CB,
    // Test harness hooks — wrap the chip builders with a capturing canvas so
    // headless Node.js tests can introspect the resulting chip element. Not
    // used by production renders.
    __testInitTooltip(mount) {
        // Inject CB2_CSS once and initialise the tooltip singleton so hover
        // handlers on test-only chips render the same rich body they do in
        // the live app. Safe to call multiple times.
        if (!document.querySelector('style[data-cb2-test-css]')) {
            const st = document.createElement('style');
            st.setAttribute('data-cb2-test-css', '1');
            st.textContent = CB2_CSS;
            (mount || document.body).appendChild(st);
        }
        _initTooltip(mount || document.body);
    },
    __testMakeStageChip(canvas, x, y, stage, roundActive, delay, roundId, colIdx, onClick, reachable) {
        const fakeCanvas = canvas && typeof canvas.appendChild === 'function'
            ? canvas : { children: [], appendChild(c) { this.children.push(c); } };
        let capturedEl = null;
        const captureCanvas = {
            appendChild(c) { capturedEl = c; fakeCanvas.appendChild(c); return c; },
        };
        const rv = makeStageChip(
            captureCanvas, x ?? 0, y ?? 0, stage, roundActive ?? true,
            delay ?? 0, roundId ?? 1, colIdx ?? 0,
            onClick ?? null, reachable ?? (() => true),
        );
        return { ...rv, el: capturedEl };
    },
    __testMakeTrackChip(canvas, x, y, track, delay, roundId, onClick, roundActive) {
        const fakeCanvas = canvas && typeof canvas.appendChild === 'function'
            ? canvas : { children: [], appendChild(c) { this.children.push(c); } };
        let capturedEl = null;
        const captureCanvas = {
            appendChild(c) { capturedEl = c; fakeCanvas.appendChild(c); return c; },
        };
        const rv = makeTrackChip(
            captureCanvas, x ?? 0, y ?? 0, track, delay ?? 0,
            roundId ?? 1, onClick ?? null, roundActive ?? true,
        );
        return { ...rv, el: capturedEl };
    },
    __testMakeIntegChip(canvas, x, y, data, delay, roundId, onClick, roundActive, reachable) {
        const fakeCanvas = canvas && typeof canvas.appendChild === 'function'
            ? canvas : { children: [], appendChild(c) { this.children.push(c); } };
        let capturedEl = null;
        const captureCanvas = {
            appendChild(c) { capturedEl = c; fakeCanvas.appendChild(c); return c; },
        };
        const rv = makeIntegChip(
            captureCanvas, x ?? 0, y ?? 0, data, delay ?? 0,
            roundId ?? 1, onClick ?? null, roundActive ?? true,
            reachable ?? (() => true),
        );
        return { ...rv, el: capturedEl };
    },
};
// Make helpers reachable from Alpine x-text expressions (module-scoped const
// bindings are not auto-attached to window).
if (typeof window !== 'undefined') window.CircuitBoard = CircuitBoard;

function trackX(idx, count, centerX) {
    const TRACK_W = 200, H_GAP = 40;
    const total = count * TRACK_W + (count - 1) * H_GAP;
    return centerX - total / 2 + idx * (TRACK_W + H_GAP);
}
