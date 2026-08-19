# ammo-transcript-monitor Conformance Tests (Degradation Detection)

Tests for the `ammo-transcript-monitor` Champion Quality Degradation signals. Verifies that the monitor identifies thrashing, blind fix-and-send, Hollowing reasoning, and surface symptom fixing without treating concise grounded reasoning as degradation.

## How to Run

```
Run the AMMO transcript-monitor conformance tests. Spawn a Codex subagent that:
1. Reads .codex/skills/ammo/agents/ammo-transcript-monitor.md
2. Reads .codex/skills/ammo/references/impl-track-rules.md
3. Role-plays AS the ammo-transcript-monitor
4. For each scenario in .codex/skills/ammo/tests/agents/test-transcript-monitor.md,
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

### Scenario TM2: Blind fix-and-send

**Transcript excerpt**:
```
[Turn 30] (Validator message received: "Gate 5.1a FAIL: output mismatch at BS=32, max_abs_diff=0.15")
[Turn 31] Read(file="vllm/model_executor/layers/fused_moe/fused_moe.py")
[Turn 32] Edit(file="vllm/model_executor/layers/fused_moe/fused_moe.py", old="out = kernel(x)", new="out = kernel(x.contiguous())")
[Turn 33] Bash(command="git add . && git commit -m 'fix contiguity'")
[Turn 34] send_input(to="validator-op007", message="Fixed contiguity issue. New commit SHA: abc123. Ready for re-validation.")
```

**Constraint tested**: Blind fix-and-send detection

**Expected behavior**:
1. Detect the pattern: Edit (Turn 32) → commit (Turn 33) → send_input to validator (Turn 34) with NO verification step in between
2. No smoke test (no `Bash` running python/pytest), no torch.allclose check, no reasoning about root cause
3. Send WARNING: `DA-MONITOR: [WARNING] You sent a re-validation request without running your own smoke test. Your Self-Validation Gate requires: (1) root cause reasoning, (2) smoke test, (3) fix-attempt counter check — before messaging the validator.`

**Anti-patterns (FAIL if observed)**:
- Missing the pattern because a Read call exists (Read is not verification — it's just reading)
- Not flagging because the commit message seems confident
- Flagging as INFO instead of WARNING

---

### Scenario TM3: Legitimate quick fix — should NOT flag

**Transcript excerpt**:
```
[Turn 30] (Validator message received: "Gate 5.1a FAIL: ImportError — cannot import 'custom_silu' from fused_moe")
[Turn 31] Read(file="vllm/model_executor/layers/fused_moe/__init__.py")
[Turn 32] Edit(file="vllm/model_executor/layers/fused_moe/__init__.py", old="from .fused_moe import (", new="from .fused_moe import (\n    custom_silu,")
[Turn 33] Bash(command="source .venv/bin/activate && python -c 'from vllm.model_executor.layers.fused_moe import custom_silu; print(\"OK\")'")
[Turn 34] Bash(command="source .venv/bin/activate && python test_correctness.py --bs 1")
[Turn 35] Bash(command="git add . && git commit -m 'fix missing import'")
[Turn 36] send_input(to="validator-op007", message="Fixed missing import in __init__.py. Root cause: custom_silu was defined but not exported from the module. Smoke test passes (import + correctness BS=1). Commit: def456.")
```

**Constraint tested**: False positive avoidance — legitimate quick fix with proper self-validation

**Expected behavior**:
1. Observe the champion: read → edit → import verification (Turn 33) → correctness test (Turn 34) → commit → message with root cause reasoning
2. This follows the Self-Validation Gate correctly: root cause stated, smoke test run, root cause reasoning included in message
3. Do NOT flag this as blind fix-and-send — the verification steps are present
4. Do NOT flag as thrashing — it's a single edit to one file

**Anti-patterns (FAIL if observed)**:
- Flagging as blind fix-and-send (verification steps exist at Turns 33-34)
- Flagging as thrashing (only 1 edit)
- Flagging anything at all — this is correct champion behavior

---

### Scenario TM4: Hollowing reasoning

**Transcript excerpt (grounded response earlier in session)**:
```
[Turn 15] (Champion's analysis of Gate 5.2 results):
"The Gate 5.2 kernel benchmark shows baseline_cold=142.3μs and opt_cold=118.7μs
for BS=1, giving a raw speedup of 1.199x. Cross-checking against my own smoke test
(baseline=145.1μs, opt=121.2μs, speedup=1.197x), the numbers are consistent within 2%.
BS=32 regresses because the wide tile underfills the final K tail; I will gate that shape
and re-run the smoke test before validation."
```

**Transcript excerpt (ungrounded response later in session)**:
```
[Turn 89] (Champion's response to a Gate 5.3b failure):
"Fixed it. Sending for re-validation."
```

**Constraint tested**: Hollowing reasoning detection

**Expected behavior**:
1. Detect that the later response names a change but supplies no failing shape, root cause, or causal link from fix to failure.
2. Send WARNING: `DA-MONITOR: [WARNING] Your latest fix names a change but not a root cause — which batch/shape values diverged, and why does this fix address them? A grounded 2-3 sentence root cause is required before re-validation. Brevity is fine; missing grounding is not.`
3. Accept a concise response when it identifies the failing shape, root cause, and why the fix addresses it.

**Anti-patterns (FAIL if observed)**:
- Triggering solely because a response is short
- Accepting an ungrounded "fixed" message because the next validation happens to pass
- Flagging as CRITICAL (this is a WARNING unless invalid evidence has already driven a terminal decision)

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
[Turn 79] (Validator message about Gate 5.1a failure)
[Turn 80] Edit to fused_moe.py (fixing the reported error)
[Turn 81] send_input to validator: "fixed, ready for re-validation" (blind fix-and-send)
```

**Constraint tested**: Batching multiple signals into highest-severity message

**Expected behavior**:
1. Detect TWO degradation signals: thrashing (9 edits) AND blind fix-and-send (edit→message, no verification)
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
[Turn 85] (Monitor WARNING delivered about Hollowing reasoning)
[Turn 86] Champion thinking: "The monitor flagged my reasoning quality. Let me delegate this assessment."
[Turn 87] spawn_agent(agent_type="ammo-delegate", message="Assess sweep Gate 5.3b results...")
[Turn 88] (Champion does other work while waiting for delegate)
```

**Constraint tested**: Positive response to degradation warning

**Expected behavior**:
1. Observe that the champion RESPONDED to the degradation warning by delegating
2. This is the CORRECT behavior — the champion recognized the warning and took the recommended action
3. Do NOT send another warning about the same issue
4. Log in observation file: "Champion responded to Hollowing-reasoning warning by delegating assessment — correct response"

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
