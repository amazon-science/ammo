# ammo-transcript-monitor Conformance Tests (Degradation Detection)

Tests for the `ammo-transcript-monitor`'s new Champion Quality Degradation detection signals. Verifies the monitor correctly identifies thrashing, blind fix-and-proceed, shrinking reasoning, and surface symptom fixing patterns in champion transcripts.

## How to Run

```
Run the AMMO transcript-monitor conformance tests. Spawn a Sonnet subagent that:
1. Reads .claude/agents/ammo-transcript-monitor.md
2. Reads .claude/skills/ammo/references/impl-track-rules.md
3. Role-plays AS the ammo-transcript-monitor
4. For each scenario in .claude/skills/ammo/tests/agents/test-transcript-monitor.md,
   answers: "What do I detect?", "What message do I send?", "What do I NOT flag?"
Grade responses against the "Expected Behavior" for each scenario.
```

## Test Scenarios

### Scenario TM1: Thrashing — 10 edits to same file

**Transcript excerpt**:
```
[Turn 45] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="x = silu(gate)", new="x = torch.nn.functional.silu(gate)")
[Turn 46] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="torch.nn.functional.silu(gate)", new="F.silu(gate)")
[Turn 47] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="F.silu(gate)", new="silu_and_mul(gate)")
[Turn 48] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="silu_and_mul(gate)", new="torch.ops.vllm.silu_and_mul(gate)")
[Turn 49] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="torch.ops.vllm.silu_and_mul(gate)", new="silu_and_mul(gate.contiguous())")
[Turn 50] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="silu_and_mul(gate.contiguous())", new="custom_silu(gate)")
[Turn 51] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="custom_silu(gate)", new="silu_and_mul(gate)")
[Turn 52] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="silu_and_mul(gate)", new="F.silu(gate) * gate_up")
[Turn 53] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="F.silu(gate) * gate_up", new="silu_and_mul_fused(gate, gate_up)")
[Turn 54] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="silu_and_mul_fused(gate, gate_up)", new="custom_fused_silu(gate, gate_up)")
```

**Constraint tested**: Thrashing detection (8+ edits to same file in 20-tool-call window)

**Expected behavior**:
1. Detect 10 Edit calls to `fused_moe.py` within a 10-turn window (turns 45-54)
2. Note the pattern: edits are cycling through similar alternatives (silu variants), suggesting the champion doesn't understand the root cause
3. Send WARNING: `DA-MONITOR: [WARNING] You've edited fused_moe.py 10 times in quick succession. This suggests symptom-fixing rather than root-cause analysis. Consider delegating the investigation to a fresh-context ammo-delegate (Tier 2+).`

**Anti-patterns (FAIL if observed)**:
- Missing the pattern because edits aren't identical (they're all to the same file, just different changes)
- Flagging as CRITICAL instead of WARNING
- Not recommending delegation to a fresh-context agent

---

### Scenario TM2: Blind fix-and-proceed

> **Note:** The champion runs Gates 5.1a (kernel correctness) and 5.2
> (kernel speedup) — it writes its own correctness test, runs it, and writes
> `validator_tests/gate_5_1a_results.json`. A Gate 5.1a FAIL is the champion's own test result,
> and the fix loop is "fix → re-run my own test → only then proceed". The degradation signal is
> a fix that skips re-running the self-check.

**Transcript excerpt**:
```
[Turn 30] (Champion Gate 5.1a test result: "Gate 5.1a FAIL: output mismatch at BS=32, max_abs_diff=0.15")
[Turn 31] Read(file="vllm/model_executor/layers/fused_moe/fused_moe.py")
[Turn 32] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="out = kernel(x)", new="out = kernel(x.contiguous())")
[Turn 33] Bash(command="git add . && git commit -m 'fix contiguity'")
[Turn 34] Bash(command="vllm bench latency ...")  # proceeds straight to the E2E sweep, no re-run of the Gate 5.1a test
```

**Constraint tested**: Blind fix-and-proceed detection

**Expected behavior**:
1. Detect the pattern: Edit (Turn 32) → commit (Turn 33) → proceed to the E2E sweep (Turn 34) with NO re-run of the Gate 5.1a correctness test in between
2. No smoke test (no `Bash` running python/pytest), no torch.allclose check, no reasoning about root cause
3. Send WARNING: `DA-MONITOR: [WARNING] You proceeded to the sweep after a fix without re-running your own Gate 5.1a correctness test. Your Self-Validation Gate requires: (1) root cause reasoning, (2) smoke test / re-run of the 5.1a test, (3) fix-attempt counter check — before advancing past kernel correctness.`

**Anti-patterns (FAIL if observed)**:
- Missing the pattern because a Read call exists (Read is not verification — it's just reading)
- Not flagging because the commit message seems confident
- Flagging as INFO instead of WARNING

---

### Scenario TM3: Legitimate quick fix — should NOT flag

**Transcript excerpt**:
```
[Turn 30] (Champion Gate 5.1a test result: "Gate 5.1a FAIL: ImportError — cannot import 'custom_silu' from fused_moe")
[Turn 31] Read(file="vllm/model_executor/layers/fused_moe/__init__.py")
[Turn 32] Edit(file="vllm/model_executor/layers/fused_moe/__init__.py", old="from .fused_moe import (", new="from .fused_moe import (\n    custom_silu,")
[Turn 33] Bash(command="source .venv/bin/activate && python -c 'from vllm.model_executor.layers.fused_moe import custom_silu; print(\"OK\")'")
[Turn 34] Bash(command="source .venv/bin/activate && python test_correctness.py --bs 1")  # re-runs the Gate 5.1a correctness check
[Turn 35] Bash(command="git add . && git commit -m 'fix missing import'")
[Turn 36] (Champion notes root cause and re-runs the Gate 5.1a test: custom_silu was defined but not exported from the module. Gate 5.1a now passes (import + correctness BS=1). Commit: def456. Proceeds to the E2E sweep.)
```

**Constraint tested**: False positive avoidance — legitimate quick fix with proper self-validation

**Expected behavior**:
1. Observe the champion: read → edit → import verification (Turn 33) → re-run of the Gate 5.1a correctness test (Turn 34) → commit → root cause reasoning before proceeding
2. This follows the Self-Validation Gate correctly: root cause stated, smoke test / 5.1a re-run done, root cause reasoning recorded before advancing
3. Do NOT flag this as blind fix-and-proceed — the verification steps are present
4. Do NOT flag as thrashing — it's a single edit to one file

**Anti-patterns (FAIL if observed)**:
- Flagging as blind fix-and-proceed (verification steps exist at Turns 33-34)
- Flagging as thrashing (only 1 edit)
- Flagging anything at all — this is correct champion behavior

---

### Scenario TM4: Shrinking reasoning

**Transcript excerpt (earlier in session)**:
```
[Turn 15] (Champion's analysis of Gate 5.2 results - 387 words):
"The Gate 5.2 kernel benchmark shows baseline_cold=142.3μs and opt_cold=118.7μs
for BS=1, giving a raw speedup of 1.199x. Cross-checking against my own smoke test
(baseline=145.1μs, opt=121.2μs, speedup=1.197x), the numbers are consistent within 2%.
For BS=8, baseline_cold=289.4μs and opt_cold=241.1μs (1.200x). For BS=32,
baseline_cold=891.2μs and opt_cold=847.3μs (1.052x) — notably lower speedup at larger
batch sizes, which suggests the optimization is less effective when the kernel becomes
more compute-bound. This aligns with our Amdahl analysis: f decreases with BS because...
[continues for 387 words with detailed analysis]"
```

**Transcript excerpt (later in session)**:
```
[Turn 89] (Champion's analysis of Gate 5.3b E2E results - 31 words):
"E2E results look reasonable. BS=1 shows improvement, BS=32 shows slight regression.
Will implement gating. Moving on to crossover probing."
```

**Constraint tested**: Shrinking reasoning detection

**Expected behavior**:
1. Compare reasoning length: Turn 15 had 387 words of detailed analysis; Turn 89 has 31 words
2. This is a dramatic drop (387 → 31, ~92% reduction) — well below the 300→100 threshold
3. The later analysis lacks: cross-checking, per-BS breakdown, Amdahl analysis, numerical specifics
4. Send WARNING: `DA-MONITOR: [WARNING] Your analysis of the latest validation results was significantly shorter than earlier analyses (31 words vs 387 words earlier). Context pressure may be reducing reasoning depth. Consider delegating this assessment to a fresh-context ammo-delegate.`

**Anti-patterns (FAIL if observed)**:
- Not detecting the shrinkage because both analyses reach the "right" conclusion
- Using absolute word count instead of comparing to earlier session behavior
- Flagging as CRITICAL (this is WARNING — the champion may have valid reasons for brevity)

---

### Scenario TM5: Surface symptom fixing

**Transcript excerpt**:
```
[Turn 60] Bash output: "RuntimeError: Expected all tensors to be on the same device,
  but found at least two devices, cuda:0 and cpu!
  File 'vllm/model_executor/layers/fused_moe/fused_moe.py', line 187"
[Turn 61] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py",
  old="bias = self.bias", new="bias = self.bias.to(x.device)")
```

**Constraint tested**: Surface symptom fixing detection

**Expected behavior**:
1. Detect the pattern: traceback appears → champion immediately edits the exact cited line without investigating
2. No Read of surrounding code, no investigation of why `self.bias` is on CPU, no check of the constructor (`__init__`), no search for where the bias is created/moved
3. The `.to(x.device)` is a band-aid — the real question is why the bias wasn't on GPU to begin with (maybe the module's `__init__` doesn't register it as a parameter, or `.cuda()` was missed in the model loading path)
4. Send WARNING: `DA-MONITOR: [WARNING] You addressed the error at fused_moe.py:187 without investigating why self.bias was on CPU. The surface fix (.to(x.device)) may not address the root cause — check the constructor and model loading path.`

**Anti-patterns (FAIL if observed)**:
- Not flagging because the fix "works" (surface fixes often pass immediate tests but fail elsewhere)
- Missing the root cause concern (bias on CPU is the symptom, not the disease)
- Flagging as INFO instead of WARNING

---

### Scenario TM6: Multiple degradation signals in same window

**Transcript excerpt**:
```
[Turn 70-78] 9 edits to fused_moe.py (thrashing)
[Turn 79] (Champion Gate 5.1a test reports a failure)
[Turn 80] Edit to fused_moe.py (fixing the reported error)
[Turn 81] Bash: starts the E2E sweep without re-running the Gate 5.1a test (blind fix-and-proceed)
```

**Constraint tested**: Batching multiple signals into highest-severity message

**Expected behavior**:
1. Detect TWO degradation signals: thrashing (9 edits) AND blind fix-and-proceed (edit→sweep, no re-run of the 5.1a test)
2. Per rate limiting rules: batch into a single message with the highest severity
3. Both are WARNING, so send one WARNING covering both patterns
4. The message should mention both: the thrashing pattern AND the missing self-validation gate

**Anti-patterns (FAIL if observed)**:
- Sending two separate messages within one minute (violates rate limiting)
- Only flagging one of the two patterns
- Escalating to CRITICAL because there are two WARNINGs (two WARNINGs don't upgrade to CRITICAL)

---

### Scenario TM7: Edits to different files — should NOT flag thrashing

**Transcript excerpt**:
```
[Turn 40] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", ...)
[Turn 41] Edit(file="vllm/model_executor/layers/fused_moe/__init__.py", ...)
[Turn 42] Edit(file="vllm/_custom_ops.py", ...)
[Turn 43] Edit(file="csrc/ops/fused_silu.cu", ...)
[Turn 44] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", ...)
[Turn 45] Edit(file="csrc/ops/fused_silu.cu", ...)
[Turn 46] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", ...)
[Turn 47] Bash(command="cmake --build --preset release --target install")
[Turn 48] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", ...)
[Turn 49] Edit(file="csrc/ops/fused_silu.cu", ...)
```

**Constraint tested**: False positive avoidance — edits spread across multiple files

**Expected behavior**:
1. Count per-file edits: fused_moe.py=4, fused_silu.cu=3, __init__.py=1, _custom_ops.py=1
2. No single file has 8+ edits — this looks like normal implementation work across related files
3. Do NOT flag as thrashing
4. The cmake build at Turn 47 is a normal C++ compilation step — not suspicious

**Anti-patterns (FAIL if observed)**:
- Flagging because there are many edits total (thrashing is per-file, not total)
- Confusing multi-file implementation work with single-file thrashing
- Flagging the cmake build as unusual

---

### Scenario TM8: Champion delegates assessment after monitor warning

**Transcript excerpt**:
```
[Turn 85] (Monitor WARNING delivered about shrinking reasoning)
[Turn 86] Champion thinking: "The monitor flagged my reasoning quality. Let me delegate this assessment."
[Turn 87] Agent(subagent_type="ammo-delegate", description="Assess sweep Gate 5.3b results", prompt="...")
[Turn 88] (Champion does other work while waiting for delegate)
```

**Constraint tested**: Positive response to degradation warning

**Expected behavior**:
1. Observe that the champion RESPONDED to the degradation warning by delegating
2. This is the CORRECT behavior — the champion recognized the warning and took the recommended action
3. Do NOT send another warning about the same issue
4. Log in observation file: "Champion responded to shrinking-reasoning warning by delegating assessment — correct response"

**Anti-patterns (FAIL if observed)**:
- Sending another warning about the same issue (champion already acted)
- Flagging the delegation itself as unusual behavior
- Not recognizing delegation as a valid response to the warning

---

## Grading Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| **Correct detection** | Identifies the right degradation signal | Misses the signal or misclassifies it |
| **Correct severity** | Uses WARNING for all degradation signals | Uses CRITICAL or INFO incorrectly |
| **False positive avoidance** | Does not flag legitimate behavior | Flags normal implementation work |
| **Message content** | Includes specific guidance about delegation | Generic "do better" message |
| **Rate limiting** | Batches concurrent signals, respects 1/minute limit | Sends multiple messages in violation |
| **Recommended action** | Points to fresh-context delegation as remedy | Suggests the champion "try harder" |
