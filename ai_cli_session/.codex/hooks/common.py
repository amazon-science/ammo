# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
from __future__ import annotations
import json,os,re,sqlite3,subprocess,sys
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any
def read_stdin_json():
    try: raw=sys.stdin.read(); return json.loads(raw) if raw.strip() else {}
    except Exception: return {}
def emit(o): print(json.dumps(o,sort_keys=True))
def block_pretool(reason): emit({"decision":"block","reason":reason,"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":reason}})
def block_stop(reason): emit({"decision":"block","reason":reason})
def additional_context(event,text): emit({"hookSpecificOutput":{"hookEventName":event,"additionalContext":text}})
def find_repo_root(cwd=None):
    start=Path(cwd or os.getcwd()).resolve()
    for p in [start,*start.parents]:
        if (p/'.codex'/'hooks').exists(): return p
        nested=p/'ai_cli_session'
        if (nested/'.codex'/'hooks').exists(): return nested
    try:
        out=subprocess.check_output(['git','-C',str(start),'rev-parse','--show-toplevel'],stderr=subprocess.DEVNULL,text=True).strip()
        if out:
            gitroot=Path(out).resolve()
            for rel in (Path('ai_cli_session'),):
                candidate=gitroot/rel
                if (candidate/'.codex'/'hooks').exists(): return candidate
            if (gitroot/'.codex'/'hooks').exists(): return gitroot
    except Exception: pass
    return start
def command_from_payload(payload):
    ti=payload.get('tool_input') or payload.get('toolInput') or {}; return ti.get('command','') if isinstance(ti,dict) else ''
def tool_name_from_payload(payload): return str(payload.get('tool_name') or payload.get('toolName') or '')
def _session_identity_path():
    root=Path(os.environ.get('AMMO_GPU_RES_DIR') or '/tmp/ammo_gpu_res').expanduser()
    return root/'codex_hook_session.json'
def trusted_server_session_identity(payload=None):
    payload=payload if isinstance(payload,dict) else {}
    for value in (os.environ.get('AMMO_SESSION_ID'),):
        if value: return str(value)
    return ''
def trusted_codex_thread_identity(payload=None):
    payload=payload if isinstance(payload,dict) else {}
    for value in (
        os.environ.get('CODEX_SESSION_ID'), payload.get('session_id'), payload.get('sessionId'),
        payload.get('conversation_id'), payload.get('conversationId'),
    ):
        if value: return str(value)
    return ''
def record_trusted_session_identity(payload):
    server_session_id=trusted_server_session_identity(payload)
    codex_thread_id=trusted_codex_thread_identity(payload)
    if not server_session_id or not codex_thread_id: return ''
    path=_session_identity_path(); path.parent.mkdir(parents=True,exist_ok=True)
    existing=load_json(path)
    if isinstance(existing,dict) and (
        existing.get('server_session_id') not in {None,server_session_id}
        or existing.get('codex_thread_id') not in {None,codex_thread_id}
    ):
        raise RuntimeError('Codex hook session identity changed within one server session')
    temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps({
        'server_session_id':server_session_id,'codex_thread_id':codex_thread_id
    },sort_keys=True)+'\n',encoding='utf-8')
    os.replace(temp,path)
    return codex_thread_id
def session_id_from_payload(payload,repo=None):
    trusted=trusted_codex_thread_identity(payload)
    if trusted: return trusted
    for k in ('turn_id','turnId'):
        v=payload.get(k)
        if v: return str(v)
    if repo is not None:
        v=state_session_id(repo)
        if v: return v
    return ''
def prompt_from_payload(payload): return payload.get('prompt') or payload.get('user_prompt') or payload.get('message') or ''
def lifecycle_task_name(payload):
    """Resolve an official subagent lifecycle agent_id through Codex's thread index."""
    agent_id=str(payload.get('agent_id') or payload.get('agentId') or '').strip()
    if not agent_id: return ''
    home=Path(os.environ.get('CODEX_HOME','~/.codex')).expanduser().resolve()
    db=home/'state_5.sqlite'
    if not db.is_file(): return ''
    try:
        connection=sqlite3.connect(f'{db.as_uri()}?mode=ro',uri=True)
        try: row=connection.execute('SELECT agent_path FROM threads WHERE id = ?',(agent_id,)).fetchone()
        finally: connection.close()
    except sqlite3.Error:
        return ''
    if not row or not isinstance(row[0],str) or not row[0]: return ''
    return row[0].rstrip('/').rsplit('/',1)[-1]
def spawn_role_and_name(payload):
    """Infer an AMMO role and stable Codex task name from a spawn/lifecycle payload."""
    ti=payload.get('tool_input') or payload.get('toolInput') or {}
    if not isinstance(ti,dict): ti={}
    role=str(ti.get('agent_type') or ti.get('subagent_type') or ti.get('agentType') or payload.get('agent_type') or payload.get('subagent_type') or '').strip()
    name=str(ti.get('task_name') or ti.get('taskName') or payload.get('task_name') or payload.get('taskName') or ti.get('name') or payload.get('agent_name') or payload.get('agentName') or '').strip()
    event=str(payload.get('hook_event_name') or payload.get('hookEventName') or '')
    if not name and event in {'SubagentStart','SubagentStop'}:
        name=lifecycle_task_name(payload)
    message='\n'.join(str(value) for value in (ti.get('message'),payload.get('message'),payload.get('prompt')) if value)
    agent_path=str(payload.get('agent_path') or payload.get('agentPath') or ti.get('agent_path') or ti.get('agentPath') or '')
    markers={'ammo-implementer':'agents/ammo-implementer.md','ammo-auditor':'agents/ammo-auditor.md','ammo-transcript-monitor':'agents/ammo-transcript-monitor.md'}
    for candidate,marker in markers.items():
        if marker in message or Path(agent_path).name in {f'{candidate}.toml',f'{candidate}.md'}:
            role=candidate; break
    lowered=name.lower().replace('-','_')
    if lowered.startswith(('impl_','implementer_','ammo_implementer_')): role='ammo-implementer'
    elif lowered.startswith(('audit_','auditor_','ammo_auditor_')): role='ammo-auditor'
    elif lowered.startswith(('monitor_','ammo_transcript_monitor_')): role='ammo-transcript-monitor'
    return role,name
def normalize_task_name(value):
    """Return one valid multi_agent_v2 task-path segment."""
    normalized=re.sub(r'[^a-z0-9_]+','_',str(value or '').strip().lower())
    return normalized.strip('_')
def task_names_equivalent(left,right):
    normalized_left=normalize_task_name(left)
    return bool(normalized_left) and normalized_left == normalize_task_name(right)
def expected_monitor_name(implementer_name):
    lowered=normalize_task_name(implementer_name)
    for prefix in ('ammo_implementer_','implementer_','impl_'):
        if lowered.startswith(prefix):
            suffix=lowered[len(prefix):]
            return f'monitor_{suffix}' if suffix else ''
    return f'monitor_{lowered}' if lowered else ''
def op_id_from_spawn_payload(payload,implementer_name='',repo=None):
    ti=payload.get('tool_input') or payload.get('toolInput') or {}
    if not isinstance(ti,dict): ti={}
    texts=[ti.get('message'),payload.get('message'),payload.get('prompt')]
    patterns=(
        r'(?im)^\s*OP_ID\s*:\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s*$',
        r'(?i)\bimplementing optimization\s+([A-Za-z0-9][A-Za-z0-9_.-]*)',
    )
    for text in texts:
        if not isinstance(text,str): continue
        for pattern in patterns:
            match=re.search(pattern,text)
            if match: return match.group(1)
    normalized=normalize_task_name(implementer_name)
    for prefix in ('ammo_implementer_','implementer_','impl_'):
        if normalized.startswith(prefix):
            task_slug=normalized[len(prefix):]
            break
    else:
        task_slug=normalized
    round_match=re.fullmatch(r'r([1-9][0-9]*)_(.+)',task_slug)
    round_hint=int(round_match.group(1)) if round_match else None
    op_slug=round_match.group(2) if round_match else task_slug
    if repo is not None and op_slug:
        artifact=active_artifact_dir(repo,payload)
        state=load_json(artifact/'state.json') if artifact else None
        campaign=state.get('campaign') if isinstance(state,dict) else None
        rounds=campaign.get('rounds') if isinstance(campaign,dict) else None
        candidates=[]
        round_records=rounds if isinstance(rounds,list) else []
        if round_hint is not None:
            round_records=round_records[round_hint-1:round_hint]
        for round_record in reversed(round_records):
            if not isinstance(round_record,dict): continue
            parallel=round_record.get('parallel_tracks')
            tracks=parallel.get('tracks') if isinstance(parallel,dict) else None
            if isinstance(tracks,dict): candidates.extend(str(value) for value in tracks)
            debate=round_record.get('debate')
            selected=debate.get('selected_candidates') if isinstance(debate,dict) else None
            if isinstance(selected,dict): selected=selected.values()
            if isinstance(selected,(list,tuple)):
                for value in selected:
                    if isinstance(value,dict) and value.get('op_id'): candidates.append(str(value['op_id']))
                    elif isinstance(value,str): candidates.append(value)
        matches=[]
        for candidate in candidates:
            if normalize_task_name(candidate) == op_slug and candidate not in matches: matches.append(candidate)
        if len(matches) == 1: return matches[0]
        event=str(payload.get('hook_event_name') or payload.get('hookEventName') or '')
        if event == 'SubagentStart': return ''
    return task_slug
def _monitor_pair_path(repo,payload):
    sid=session_id_from_payload(payload,repo)
    if not sid: return None
    safe=hashlib.sha256(str(sid).encode('utf-8')).hexdigest()[:24]
    root=Path(os.environ.get('AMMO_GPU_RES_DIR') or '/tmp/ammo_gpu_res').expanduser()
    return root/f'codex_monitor_pairs_{safe}.json'
def pending_monitor_pairs(repo,payload):
    path=_monitor_pair_path(repo,payload)
    doc=load_json(path) if path else None
    records=doc.get('pending') if isinstance(doc,dict) else None
    return [record for record in records if isinstance(record,dict)] if isinstance(records,list) else []
def _write_monitor_pairs(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps({'pending':records},sort_keys=True,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)
def record_monitor_pair(repo,payload,implementer_name):
    path=_monitor_pair_path(repo,payload)
    expected=expected_monitor_name(implementer_name)
    if path is None or not expected: return False
    records=pending_monitor_pairs(repo,payload)
    agent_id=str(payload.get('agent_id') or payload.get('agentId') or '')
    existing=next((record for record in records if task_names_equivalent(record.get('implementer_name'),implementer_name)),None)
    candidate_op_id=op_id_from_spawn_payload(payload,implementer_name,repo)
    if existing is None:
        if not candidate_op_id: return False
        records.append({'implementer_name':normalize_task_name(implementer_name),'implementer_agent_id':agent_id,'op_id':candidate_op_id,'expected_monitor_name':expected,'status':'pending'})
    else:
        if agent_id: existing['implementer_agent_id']=agent_id
        if candidate_op_id and candidate_op_id != existing.get('op_id'): existing['op_id']=candidate_op_id
        if not existing.get('op_id'): return False
        existing['expected_monitor_name']=expected
    _write_monitor_pairs(path,records)
    return True
def clear_monitor_pair(repo,payload,monitor_name):
    path=_monitor_pair_path(repo,payload)
    if path is None: return False
    records=pending_monitor_pairs(repo,payload)
    remaining=[record for record in records if not task_names_equivalent(record.get('expected_monitor_name'),monitor_name)]
    if len(remaining)==len(records): return False
    if remaining: _write_monitor_pairs(path,remaining)
    else:
        try: path.unlink()
        except FileNotFoundError: pass
    return True
def start_monitor_pair(repo,payload,monitor_name):
    path=_monitor_pair_path(repo,payload)
    if path is None: return False
    records=pending_monitor_pairs(repo,payload); changed=False
    for record in records:
        if task_names_equivalent(record.get('expected_monitor_name'),monitor_name):
            record['status']='active'
            record['monitor_agent_id']=str(payload.get('agent_id') or payload.get('agentId') or '')
            changed=True
    if changed: _write_monitor_pairs(path,records)
    return changed
def complete_monitor_pair(repo,payload,monitor_name,artifact_dir):
    path=_monitor_pair_path(repo,payload)
    if path is None: return False
    records=pending_monitor_pairs(repo,payload); target=None
    for record in records:
        if task_names_equivalent(record.get('expected_monitor_name'),monitor_name) and record.get('status') == 'active':
            target=record; break
    if target is None: return False
    op_id=str(target.get('op_id') or '').strip()
    if not op_id:
        normalized_monitor=normalize_task_name(monitor_name)
        op_id=normalized_monitor[len('monitor_'):] if normalized_monitor.startswith('monitor_') else ''
    state=load_json(Path(artifact_dir)/'state.json')
    campaign=state.get('campaign') if isinstance(state,dict) else None
    round_id=campaign.get('current_round') if isinstance(campaign,dict) else None
    audit_dir=Path(artifact_dir)/'rounds'/str(round_id)/'tracks'/op_id/'monitor_audits'
    logs=[candidate for candidate in audit_dir.glob('*_observations.md') if candidate.is_file() and candidate.stat().st_size > 0] if audit_dir.is_dir() else []
    queue=Path(artifact_dir)/'monitor_interventions.jsonl'
    final=None
    for entry in read_monitor_jsonl(queue):
        record=entry.get('record')
        if not isinstance(record,dict): continue
        if (record.get('emitter') == 'ammo-transcript-monitor'
                and str(record.get('severity') or '').upper() == 'INFO'
                and str(record.get('target_rollout_id') or '') == str(target.get('implementer_agent_id') or '')
                and 'poll' in str(record.get('summary') or '').lower()):
            final=record
    if not logs or final is None: return False
    chosen=max(logs,key=lambda candidate:candidate.stat().st_mtime_ns)
    target['status']='satisfied'; target['summary_path']=str(chosen); target['summary_sha256']=hashlib.sha256(chosen.read_bytes()).hexdigest(); target['final_summary']=final.get('summary')
    _write_monitor_pairs(path,records)
    return True
def candidate_artifact_dirs(repo):
    dirs=[]; env=os.environ.get('AMMO_ARTIFACT_DIR')
    if env: dirs.append((repo/Path(env)).resolve() if not Path(env).is_absolute() else Path(env).resolve())
    root=repo/'kernel_opt_artifacts'
    if root.exists(): dirs += [p.parent for p in sorted(root.glob('**/state.json'),key=lambda p:p.stat().st_mtime,reverse=True)]
    seen=[]
    for d in dirs:
        if d not in seen: seen.append(d)
    return seen
def active_artifact_dir(repo,payload=None):
    payload=payload if isinstance(payload,dict) else {}
    expected=trusted_server_session_identity(payload)
    candidates=candidate_artifact_dirs(repo)
    if expected:
        for d in candidates:
            state=load_json(d/'state.json')
            if isinstance(state,dict) and str(state.get('session_id') or state.get('sessionId') or '') == expected:
                return d
        return None
    for d in candidates:
        if (d/'state.json').exists(): return d
    return None
def load_json(p):
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return None
def state_session_id(repo):
    ad=active_artifact_dir(repo)
    st=load_json(ad/'state.json') if ad else None
    if isinstance(st,dict):
        for k in ('codex_thread_id','session_id','sessionId'):
            v=st.get(k)
            if v: return str(v)
    return ''
def is_ammo_context(cmd,repo): return ('.codex/skills/ammo' in cmd or 'kernel_opt_artifacts' in cmd or active_artifact_dir(repo) is not None)
# Shell-command classification lives in ONE place: the ammo skill's
# hook_cmd_classify.py, which the shell hooks already call. Import it here so
# the Codex hooks and the shell hooks cannot drift apart, and re-export the
# names this module's other functions (and the sibling guards) call.
#
# The import AND every attribute read below are guarded because EVERY Codex hook
# imports this module. An unguarded ModuleNotFoundError -- or an unguarded
# AttributeError from a classifier that imports but is incomplete, because a
# zero-byte or truncated file still parses -- would kill all five hooks with a
# traceback and an empty stdout, so a missing or partial classifier would
# silently disable the whole enforcement layer -- every PreToolUse deny and the
# Stop gate at once. That is reachable: the Dockerfile ships hooks/ and
# skills/ammo/scripts/ as one `cp -a` tree, and a partial S3 worktree restore on
# cross-pod resume can land one without the other, or truncate one of them. One
# missing attribute degrades all five re-exports together, so the fail direction
# below is the only degraded behavior either way.
#
# Fail direction on failure, matching the shell hooks that guard the same file:
#   is_static_inspection_command -> False   fail-CLOSED, as ammo-pretool-guard.sh
#       does. Nothing earns the inspection fast path, so every caller falls
#       through to its normal guard logic instead of exiting early.
#   split_shell_command -> inline shlex.split. It carries no policy, so the
#       real semantics are reproduced rather than degraded.
#   the private segment helpers -> empty/False, which also degrades
#       command_invokes_gpu_heavy_tool to False. That predicate feeds a hard deny
#       in pre_tool_use_guard.py -- nsys/ncu and the other GPU-heavy tools
#       without a reservation -- so that one guard is fail-OPEN when degraded:
#       its deny becomes an allow. The stderr warning below discloses it.
_CLASSIFY_SCRIPTS_DIR=str(Path(__file__).resolve().parents[1]/'skills'/'ammo'/'scripts')
if _CLASSIFY_SCRIPTS_DIR not in sys.path: sys.path.insert(0,_CLASSIFY_SCRIPTS_DIR)
try:
    import hook_cmd_classify as _cmd_classify
    # Inside the guard on purpose: an incomplete classifier fails here, not at
    # import, and both failures must degrade instead of raising.
    split_shell_command=_cmd_classify.split_shell_command
    _executable_segments=_cmd_classify._executable_segments
    _is_python_executable_name=_cmd_classify._is_python_executable_name
    _is_read_only_python_segment=_cmd_classify._is_read_only_python_segment
    is_static_inspection_command=_cmd_classify.is_static_inspection_command
except Exception as _classify_exc:
    _CLASSIFY_DEGRADED=str(_classify_exc) or _classify_exc.__class__.__name__
    # Loud on stderr, never on stdout: stdout carries the hook verdict document
    # and must stay parseable. A degraded layer must not also be a silent one.
    try:
        sys.stderr.write(
            'AMMO hooks DEGRADED: '+_CLASSIFY_SCRIPTS_DIR+'/hook_cmd_classify.py '
            'unavailable or incomplete ('+_CLASSIFY_DEGRADED+'). Inspection fast '
            'paths are OFF (fail-closed); GPU-heavy tool detection is OFF, so the '
            'unreserved-GPU deny in pre_tool_use_guard.py allows those commands '
            '(fail-open).\n'
        )
    except Exception:
        pass
    import shlex as _shlex
    def split_shell_command(command):
        try: return _shlex.split(command or '')
        except ValueError: return []
    def _executable_segments(parts): return []
    def _is_python_executable_name(name): return False
    def _is_read_only_python_segment(tail): return False
    def is_static_inspection_command(command): return False
else:
    _CLASSIFY_DEGRADED=''
def command_invokes_vllm_bench_latency(command):
    parts=split_shell_command(command)
    names=[Path(part).name for part in parts]
    for idx,name in enumerate(names[:-2]):
        if name == 'vllm' and parts[idx+1:idx+3] == ['bench','latency']:
            return True
        if re.fullmatch(r'python(?:\d+(?:\.\d+)?)?',name) and idx + 4 <= len(parts):
            if parts[idx+1:idx+5] == ['-m','vllm','bench','latency']:
                return True
    return False
def command_invokes_gpu_heavy_tool(command):
    if is_static_inspection_command(command): return False
    parts=split_shell_command(command)
    if not parts: return False
    for name,tail in _executable_segments(parts):
        if name in {'nsys','ncu','torchrun','vllm','run_vllm_bench_latency_sweep.py'}:
            return True
        if name == 'nvidia-smi' and any(token.startswith('--query-compute') for token in tail):
            return True
        if _is_python_executable_name(name) and _is_read_only_python_segment(tail):
            continue
        if re.fullmatch(r'python(?:\d+(?:\.\d+)?)?|pytest',name):
            if any(Path(token).name == 'run_vllm_bench_latency_sweep.py' for token in tail):
                return True
            tail_text=' '.join(tail)
            if re.search(r'(torch|cuda|triton|vllm|benchmark|kernel|gpu)',tail_text,re.IGNORECASE):
                if re.fullmatch(r'-c\s+import\s+(vllm|torch)',tail_text.strip()):
                    return False
                return True
    return False
def monitor_queue_paths(repo,payload=None):
    paths=[]; env=os.environ.get('AMMO_MONITOR_QUEUE')
    if env: paths.append(Path(env).expanduser())
    ad=active_artifact_dir(repo,payload)
    if ad:
        paths.append(ad/'monitor_interventions.jsonl')
        try:
            paths.extend(sorted(ad.glob('**/monitor_interventions.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True))
        except Exception:
            pass
    seen=[]
    for p in paths:
        p=(repo/p).resolve() if not p.is_absolute() else p.resolve()
        if p not in seen: seen.append(p)
    return seen
def read_monitor_jsonl(path):
    entries=[]
    try:
        for line_no,line in enumerate(path.read_text(encoding='utf-8').splitlines(),start=1):
            if not line.strip():
                entries.append({'line_no':line_no,'raw':line,'record':None})
                continue
            try: record=json.loads(line)
            except Exception: record=None
            entries.append({'line_no':line_no,'raw':line,'record':record})
    except FileNotFoundError: pass
    except Exception: pass
    return entries
def record_session_id(record):
    return str(record.get('target_session_id') or record.get('session_id') or record.get('targetSessionId') or '')
def _record_session(record):
    return record_session_id(record)
def _dedupe_nonempty(values):
    seen=[]
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen
def monitor_target_session_ids(repo,payload):
    state_sid=state_session_id(repo)
    if state_sid:
        return _dedupe_nonempty([state_sid])
    return _dedupe_nonempty([session_id_from_payload(payload,repo)])
def stable_monitor_record_id(path,index,record):
    explicit=record.get('record_id') or record.get('id')
    if explicit: return str(explicit)
    body=json.dumps(record,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(f'{path}:{index}:{body}'.encode('utf-8')).hexdigest()[:12]
def _record_open(record):
    status=str(record.get('status') or '').lower()
    if status in {'acknowledged','resolved','dismissed','closed'}: return False
    return not (record.get('acknowledged_at') or record.get('resolved_at'))
def _record_requires_ack(record):
    if record.get('ack_required') is True: return True
    if isinstance(record.get('ack_required'),str): return record.get('ack_required').lower() in {'1','true','yes'}
    return False
def _record_expired(record):
    expires=str(record.get('expires_at') or '').strip()
    if not expires: return False
    try:
        dt=datetime.fromisoformat(expires.replace('Z','+00:00'))
        return dt <= datetime.now(timezone.utc)
    except Exception:
        return False
def pending_monitor_records(repo,payload,severities=None):
    sids=set(monitor_target_session_ids(repo,payload))
    if not sids: return []
    wanted={s.upper() for s in severities} if severities else None
    records=[]
    for path in monitor_queue_paths(repo,payload):
        for entry in read_monitor_jsonl(path):
            rec=entry.get('record')
            if not isinstance(rec,dict): continue
            idx=entry.get('line_no') or 0
            if _record_session(rec) not in sids: continue
            if not _record_open(rec): continue
            if _record_expired(rec): continue
            sev=str(rec.get('severity') or '').upper()
            if wanted and sev not in wanted: continue
            rid=stable_monitor_record_id(path,idx,rec)
            rec=dict(rec); rec['_queue_path']=str(path); rec['_record_id']=rid; records.append(rec)
    return records
def blocking_monitor_records(repo,payload):
    return [r for r in pending_monitor_records(repo,payload,{'CRITICAL','HARD_GATE'}) if _record_requires_ack(r)]
def format_monitor_records(records,limit=3):
    lines=[]
    for rec in records[:limit]:
        sev=str(rec.get('severity') or 'INFO').upper()
        summary=str(rec.get('summary') or '').strip()
        evidence=str(rec.get('evidence') or '').strip()
        action=str(rec.get('recommended_action') or '').strip()
        rid=str(rec.get('_record_id') or rec.get('record_id') or rec.get('id') or '').strip()
        prefix=f"AMMO-MONITOR: [{sev}]"
        if rid: prefix+=f" id={rid}"
        lines.append(f"{prefix} {summary}")
        if evidence: lines.append(f"Evidence: {evidence}")
        if action: lines.append(f"Recommended action: {action}")
    if len(records)>limit: lines.append(f"... {len(records)-limit} more pending monitor item(s).")
    return '\n'.join(lines)
