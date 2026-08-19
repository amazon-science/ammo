#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import glob,os,re,stat
import shlex
import sys
from pathlib import Path
from common import active_artifact_dir,additional_context,block_pretool,blocking_monitor_records,command_from_payload,command_invokes_gpu_heavy_tool,command_invokes_vllm_bench_latency,find_repo_root,format_monitor_records,is_ammo_context,is_static_inspection_command,load_json,pending_monitor_pairs,read_stdin_json,record_session_id,session_id_from_payload,spawn_role_and_name,state_session_id,task_names_equivalent

# Always-on session provisioning guard. Install classification is the shared
# tokenizer in skills/ammo/scripts/hook_cmd_classify.py -- the shell hooks and
# the Codex hooks must deny the same commands, so there is one implementation.
# It parses shell command positions so path-qualified executables, wrappers,
# legal global options, and nested `sh -c` forms are denied without matching
# quoted grep/echo documentation text.
#
# Guarded like the re-export in common.py: this hook runs on EVERY tool call, so
# an unguarded ImportError would replace every PreToolUse verdict with a
# traceback and disable the whole layer, not just this one guard. Fail-OPEN on a
# missing classifier -- the same direction ammo-pip-guard.sh declares for the
# same file -- so the two runtimes still agree when degraded. The pip policy is
# also carried in AGENTS.md prose and the .venv provisioning path.
_CLASSIFY_SCRIPTS_DIR=str(Path(__file__).resolve().parents[1]/'skills'/'ammo'/'scripts')
if _CLASSIFY_SCRIPTS_DIR not in sys.path: sys.path.insert(0,_CLASSIFY_SCRIPTS_DIR)
try:
    from hook_cmd_classify import command_installs as _command_installs
except Exception as _installs_exc:
    try:
        sys.stderr.write(
            'AMMO hooks DEGRADED: install classification unavailable '
            '('+(str(_installs_exc) or _installs_exc.__class__.__name__)+'). '
            'The pip/uv install deny is OFF (fail-open, matching '
            'ammo-pip-guard.sh); every other PreToolUse guard still runs.\n'
        )
    except Exception:
        pass
    def _command_installs(command,depth=0): return False

p=read_stdin_json(); cmd=command_from_payload(p); repo=find_repo_root(p.get('cwd'))
def _guard_package_installs():
    if not cmd or os.environ.get('AMMO_ALLOW_PIP') == '1':
        return
    if _command_installs(cmd):
        block_pretool(
            'BLOCKED: package install/uninstall is forbidden in AMMO sessions. '
            'The session .venv is pre-provisioned; pip/uv pip mutation can invalidate '
            'benchmark results. Source .venv/bin/activate instead. Provisioning may '
            'set AMMO_ALLOW_PIP=1 explicitly.'
        )
        raise SystemExit(0)
def _guard_session_identity_mutation():
    if os.environ.get('AMMO_ALLOW_ANCHOR_MUTATION') == '1': return
    haystack='\n'.join(_payload_texts())
    if 'codex_hook_session.json' in haystack or re.search(
        r'\b(?:rm|unlink|rmdir|shutil\.rmtree)\b[^\n]*(?:AMMO_GPU_RES_DIR|/tmp/ammo_gpu_res)',
        haystack,
    ):
        block_pretool(
            'AMMO trusted-session identity and the GPU reservation dir are owned by the hooks. '
            'Set AMMO_ALLOW_ANCHOR_MUTATION=1 only for validated recovery.'
        )
        raise SystemExit(0)
def _tool_input():
    ti=p.get('tool_input') or p.get('toolInput') or {}
    return ti if isinstance(ti,dict) else {}
def _payload_texts():
    ti=_tool_input()
    texts=[cmd,str(p.get('command') or ''),str(p.get('haystack') or '')]
    for key in ('command','haystack','file_path','filePath','path','content','new_string','newString','old_string','oldString','patch'):
        if key in ti: texts.append(str(ti.get(key) or ''))
    return [t for t in texts if t]
def _vllm_op_default_on(text):
    patterns=[
        r'\bVLLM_OP\d+\b[^\n#]*(?:=\s*(?:True|1\b|["\']1["\'])|:\s*bool\s*=\s*True|,\s*True\b)',
        r'\bVLLM_OP\d+\b[^\n#]*(?:default\s*=\s*(?:True|1|["\']1["\']))',
        r'default\s*=\s*(?:True|1|["\']1["\'])[^\n#]*\bVLLM_OP\d+\b',
        r'os\.environ\.get\(\s*["\']VLLM_OP\d+["\']\s*,\s*["\']1["\']\s*\)',
    ]
    return any(re.search(pattern,text,re.IGNORECASE) for pattern in patterns)
def _env_default_guard_paths():
    ti=_tool_input()
    paths=[]
    for key in ('file_path','filePath','path','target_file','targetFile'):
        value=ti.get(key)
        if isinstance(value,str) and value:
            try: paths.append(Path(value.strip().strip('"\'')).name)
            except Exception: pass
    return paths
def _current_is_child_agent():
    if os.environ.get('CODEX_SUBAGENT') == '1' or os.environ.get('CLAUDE_SUBAGENT') == '1':
        return True
    if p.get('agent_type') or p.get('agentType'):
        return True
    agent_name=str(p.get('agentName') or p.get('agent_name') or '')
    return bool(agent_name and agent_name != 'team-lead')
def _resolve_payload_path(raw):
    try:
        path=Path(str(raw).strip().strip('"\''))
        if not str(path):
            return None
        if not path.is_absolute():
            path=(Path(p.get('cwd') or os.getcwd())/path)
        return path.resolve()
    except Exception:
        return None
def _edit_target_paths():
    ti=_tool_input()
    paths=[]
    for key in ('file_path','filePath','path','target_file','targetFile','notebook_path','notebookPath'):
        value=ti.get(key)
        if isinstance(value,str) and value:
            resolved=_resolve_payload_path(value)
            if resolved is not None:
                paths.append(resolved)
    for text in _payload_texts():
        for match in re.findall(r'^\*\*\* (?:Add|Update|Delete) File:\s+(.+)$',text,flags=re.MULTILINE):
            resolved=_resolve_payload_path(match)
            if resolved is not None:
                paths.append(resolved)
    seen=[]
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen
def _guard_orchestrator_owned_ammo_files():
    if not _current_is_child_agent():
        return
    protected=(
        '/.codex/skills/ammo/references/',
        '/.codex/skills/ammo/orchestration/',
        '/.codex/skills/ammo/agents/',
        '/.codex/agents/',
    )
    for path in _edit_target_paths():
        posix='/' + path.as_posix().lstrip('/')
        if any(token in posix for token in protected):
            block_pretool(
                'Subagents may not edit '+str(path)+'. Files under '
                '.codex/skills/ammo/references/, .codex/skills/ammo/orchestration/, '
                '.codex/skills/ammo/agents/, and .codex/agents/ are orchestrator-owned. '
                'Surface the proposed change to the lead instead.'
            )
            raise SystemExit(0)
def _guard_vllm_op_defaults():
    haystack='\n'.join(_payload_texts())
    if 'envs.py' not in _env_default_guard_paths() or 'VLLM_OP' not in haystack: return
    if _vllm_op_default_on(haystack):
        block_pretool('VLLM_OP feature flags in envs.py must default off. Use default=False/0 and require explicit opt-in before enabling optimized paths.')
        raise SystemExit(0)
def _worktree_root(cwd):
    if not cwd: return None
    try: start=Path(cwd).resolve()
    except Exception: return None
    for path in [start,*start.parents]:
        if path.parent.name == 'worktrees' and path.parent.parent.name == '.codex':
            return path
    return None
def _split_command(command):
    try: return shlex.split(command)
    except ValueError: return []
def _invoked_pythonish_tools(command):
    tools=[]
    for token in _split_command(command):
        if token in {'&&',';','||','|','(',')'} or '=' in token and not token.startswith(('/','.')):
            continue
        name=Path(token).name
        if re.fullmatch(r'python(?:\d+(?:\.\d+)?)?|pytest|pip(?:\d+)?|uv',name):
            tools.append(token)
    return tools
def _uses_worktree_venv(command,worktree):
    activate=str(worktree/'.venv'/'bin'/'activate')
    if re.search(r'(?:^|[;&|]\s*)(?:source|\.)\s+(?:"|\')?'+re.escape(activate)+r'(?:"|\')?',command):
        return True
    if re.search(r'(?:^|[;&|]\s*)(?:source|\.)\s+(?:"|\')?(?:\./)?\.venv/bin/activate(?:"|\')?',command):
        return True
    parts=_split_command(command)
    for idx,token in enumerate(parts[:-1]):
        if token in {'source','.'}:
            try:
                sourced=Path(parts[idx+1]).expanduser()
                if not sourced.is_absolute():
                    sourced=(worktree/sourced)
                if sourced.resolve() == (worktree/'.venv'/'bin'/'activate').resolve():
                    return True
            except Exception:
                pass
    venv_bin=(worktree/'.venv'/'bin').resolve()
    for token in _invoked_pythonish_tools(command):
        try:
            path=Path(token).expanduser()
            if path.is_absolute() and path.resolve().parent == venv_bin:
                return True
        except Exception:
            pass
    return False
def _read_only_pythonish(command):
    parts=_split_command(command)
    if not parts: return False
    meaningful=[x for x in parts if x not in {'env','command'} and not (('=' in x and not x.startswith(('/','.'))))]
    if not meaningful: return False
    joined=' '.join(meaningful)
    first=Path(meaningful[0]).name
    if first.startswith('python'):
        return bool(re.fullmatch(r'python(?:\d+(?:\.\d+)?)?\s+(-V|--version|-VV|--help|-h)',joined))
    if re.fullmatch(r'pip(?:\d+)?',first):
        return bool(re.fullmatch(r'pip(?:\d+)?\s+(--version|-V|list|freeze|show)(?:\s+[\w_.-]+)?',joined))
    if first == 'uv':
        return bool(re.fullmatch(r'uv\s+(--version|-V|help|pip\s+(list|freeze|show)(?:\s+[\w_.-]+)?)',joined))
    if first == 'pytest':
        return bool(re.fullmatch(r'pytest\s+(--version|-h|--help|--collect-only)(?:\s+[\w./:-]+)*',joined))
    return False
def _venv_sentinel():
    root=Path(os.environ.get('AMMO_GPU_RES_DIR') or '/tmp/ammo_gpu_res').expanduser()
    sid=session_id_from_payload(p,repo) or p.get('turn_id') or p.get('turnId') or 'session'
    safe=re.sub(r'[^A-Za-z0-9_.-]+','_',str(sid))[:96] or 'session'
    return root/f'worktree_venv_guard_{safe}.seen'
def _guard_worktree_venv():
    if not cmd: return
    worktree=_worktree_root(p.get('cwd'))
    if not worktree: return
    if not _invoked_pythonish_tools(cmd): return
    if _uses_worktree_venv(cmd,worktree): return
    detail=' Source '+str(worktree/'.venv'/'bin'/'activate')+' or invoke absolute executables under '+str(worktree/'.venv'/'bin')+'.'
    block_pretool('Python, pytest, pip, and uv commands inside .codex/worktrees must use the worktree virtualenv.'+detail)
    raise SystemExit(0)
# vLLM-dependent sweep/profiling scripts that MUST run under a .venv python:
# system python has no vllm, so a bare invocation wastes a sweep slot (or
# worse, picks up a wrong install). Fires on ANY cwd — the worktree guard
# above only covers commands issued from inside .codex/worktrees/.
_SWEEP_SCRIPTS_RE=r'(?:run_vllm_bench_latency_sweep|ncu_sanity_driver)\.py'
def _guard_sweep_script_venv():
    if not cmd: return
    if not re.search(r'python(?:\d+(?:\.\d+)?)?\s+\S*'+_SWEEP_SCRIPTS_RE,cmd): return
    # Allow: an explicit .venv/bin/python prefix (relative or absolute).
    if re.search(r'\.venv/bin/python(?:\d+(?:\.\d+)?)?\s+\S*'+_SWEEP_SCRIPTS_RE,cmd): return
    # Allow: source .venv/bin/activate earlier in the command chain.
    if re.search(r'(?:source|\.)\s+\S*\.venv/bin/activate\s*(?:&&|;)',cmd): return
    block_pretool(
        'vLLM sweep/profiling scripts must run under the session virtualenv — '
        'system python has no vllm. Use .venv/bin/python '
        '.codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py ... or '
        'source .venv/bin/activate first.'
    )
    raise SystemExit(0)
_guard_package_installs()
_guard_session_identity_mutation()
_guard_vllm_op_defaults()
_guard_orchestrator_owned_ammo_files()
_guard_worktree_venv()
_guard_sweep_script_venv()
def _guard_monitor_pairing():
    pending_pairs=[record for record in pending_monitor_pairs(repo,p) if record.get('status') == 'pending']
    if not pending_pairs: return
    role,name=spawn_role_and_name(p)
    expected={str(record.get('expected_monitor_name') or '') for record in pending_pairs}
    if role == 'ammo-transcript-monitor' and any(task_names_equivalent(name,value) for value in expected):
        return
    if cmd and is_static_inspection_command(cmd):
        return
    if cmd and 'resolve_codex_transcript.py' in cmd and not re.search(r'[;&|`$<>]|\n',cmd):
        return
    owed=', '.join(sorted(value for value in expected if value)) or '<matching monitor>'
    block_pretool(
        'AMMO implementer spawn is awaiting its mandatory paired transcript monitor. '
        'Spawn exactly '+owed+' before any further intercepted mutation.'
    )
    raise SystemExit(0)
_guard_monitor_pairing()
def _monitor_ack_shell_safe(command):
    """Allow literal punctuation in quotes, never executable shell syntax."""
    quote=None
    escaped=False
    for char in command:
        if char in '\r\n': return False
        if quote == "'":
            if char == "'": quote=None
            continue
        if escaped:
            if quote is None and char in ';&|<>$`': return False
            escaped=False
            continue
        if char == '\\':
            escaped=True
            continue
        if quote == '"':
            if char == '"': quote=None
            elif char in '$`': return False
            continue
        if char == "'" or char == '"':
            quote=char
        elif char in ';&|<>$`*?[]{}~':
            return False
    return quote is None and not escaped
def _monitor_ack_file_inspection(command):
    """Permit only non-interpreting readers for this security-sensitive script."""
    if not _monitor_ack_shell_safe(command): return False
    try: parts=shlex.split(command)
    except ValueError: return False
    trusted_readers={
        '/bin/cat','/usr/bin/cat','/bin/head','/usr/bin/head','/bin/tail','/usr/bin/tail',
        '/bin/wc','/usr/bin/wc','/bin/stat','/usr/bin/stat','/bin/sha256sum','/usr/bin/sha256sum',
        '/bin/md5sum','/usr/bin/md5sum','/bin/cmp','/usr/bin/cmp',
    }
    if not parts or parts[0] not in trusted_readers: return False
    return any(Path(token).name == 'monitor_queue_ack.py' for token in parts[1:])
def _monitor_ack_script_mentioned(command,depth=0):
    if 'monitor_queue_ack.py' in command: return True
    if depth > 2: return False
    try: parts=shlex.split(command)
    except ValueError: return False
    canonical=Path(__file__).resolve().parents[1]/'skills'/'ammo'/'scripts'/'monitor_queue_ack.py'
    for token in parts:
        if Path(token).name == 'monitor_queue_ack.py': return True
        try:
            candidate=_resolve_monitor_ack_path(token)
            if candidate.exists() and canonical.exists() and os.path.samefile(candidate,canonical): return True
        except (OSError,RuntimeError):
            pass
        if glob.has_magic(token):
            try:
                pattern=token if Path(token).is_absolute() else str(Path(p.get('cwd') or os.getcwd())/token)
                for expanded in glob.glob(pattern):
                    candidate=Path(expanded)
                    if candidate.name == 'monitor_queue_ack.py': return True
                    if candidate.exists() and canonical.exists() and os.path.samefile(candidate,canonical): return True
            except (OSError,RuntimeError):
                pass
        if token != command and any(char.isspace() for char in token) and _monitor_ack_script_mentioned(token,depth+1): return True
        if token != command and ('monitor_queue_' in token or 'ack.py' in token) and _monitor_ack_script_mentioned(token,depth+1): return True
    return False
def _monitor_ack_options(parts,start):
    values={key:[] for key in ('--session-id','--queue','--record-id','--note','--status')}
    all_count=0; idx=start
    while idx < len(parts):
        token=parts[idx]
        if token == '--all':
            all_count+=1; idx+=1; continue
        if not token.startswith('--'): return None
        if '=' in token:
            key,value=token.split('=',1)
        else:
            key=token
            if key not in values or idx+1 >= len(parts): return None
            idx+=1; value=parts[idx]
            if value.startswith('--'): return None
        if key not in values or not value: return None
        values[key].append(value); idx+=1
    if len(values['--session-id']) != 1 or len(values['--queue']) != 1 or len(values['--note']) != 1: return None
    if len(values['--status']) > 1 or (values['--status'] and values['--status'][0] not in {'acknowledged','resolved'}): return None
    if all_count > 1 or bool(all_count) == bool(values['--record-id']): return None
    return values, bool(all_count)
def _resolve_monitor_ack_path(value):
    path=Path(value).expanduser()
    base=Path(p.get('cwd') or os.getcwd())
    return (base/path).resolve() if not path.is_absolute() else path.resolve()
def _trusted_monitor_ack_python(value):
    path=Path(value)
    if not path.is_absolute() or path.parent not in {Path('/bin'),Path('/usr/bin'),Path('/usr/local/bin')}: return False
    if not re.fullmatch(r'python(?:3(?:\.\d+)?)?',path.name): return False
    try:
        link_info=path.lstat()
        target_info=path.resolve(strict=True).stat()
    except (OSError,RuntimeError):
        return False
    if link_info.st_uid != 0 or target_info.st_uid != 0: return False
    if stat.S_IMODE(target_info.st_mode) & 0o022: return False
    return stat.S_ISREG(target_info.st_mode)
def _is_monitor_ack_command(command,blocking_records):
    if not command: return False
    if not _monitor_ack_shell_safe(command): return False
    try: parts=shlex.split(command)
    except ValueError: return False
    if not parts: return False
    exe=parts[0]
    exe_name=Path(exe).name
    if re.fullmatch(r'python(?:\d+(?:\.\d+)?)?',exe_name):
        if not _trusted_monitor_ack_python(exe): return False
        if len(parts) < 2: return False
        script=parts[1]; option_start=2
    else:
        return False
    canonical=Path(__file__).resolve().parents[1]/'skills'/'ammo'/'scripts'/'monitor_queue_ack.py'
    try:
        if _resolve_monitor_ack_path(script) != canonical: return False
    except Exception:
        return False
    parsed=_monitor_ack_options(parts,option_start)
    if parsed is None: return False
    values,ack_all=parsed
    if not blocking_records: return False
    try: queue=_resolve_monitor_ack_path(values['--queue'][0])
    except Exception: return False
    if not Path(values['--queue'][0]).expanduser().is_absolute(): return False
    session=values['--session-id'][0]
    matching=[record for record in blocking_records if record_session_id(record) == session and _resolve_monitor_ack_path(str(record.get('_queue_path') or '')) == queue]
    if not matching: return False
    if ack_all: return True
    pending_ids={str(record.get('_record_id') or '') for record in matching}
    return bool(values['--record-id']) and all(record_id in pending_ids for record_id in values['--record-id'])
def _stage3_reconciliation_gate(record):
    return str(record.get('category') or '').lower() == 'stage3_debate_reconciliation'
def _current_campaign_round():
    ad=active_artifact_dir(repo,p)
    state=load_json(ad/'state.json') if ad else None
    campaign=state.get('campaign') if isinstance(state,dict) else None
    current=campaign.get('current_round') if isinstance(campaign,dict) else None
    return current if isinstance(current,int) and not isinstance(current,bool) and current > 0 else None
def _payload_paths():
    ti=_tool_input()
    paths=[]
    for key in ('file_path','filePath','path','target_file','targetFile'):
        value=ti.get(key)
        if isinstance(value,str) and value:
            resolved=_resolve_payload_path(value)
            if resolved is not None:
                paths.append(resolved)
    for text in _payload_texts():
        for match in re.findall(r'^\*\*\* (?:Add|Update|Delete) File:\s+(.+)$',text,flags=re.MULTILINE):
            resolved=_resolve_payload_path(match)
            if resolved is not None:
                paths.append(resolved)
    seen=[]
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen
def _current_round_proposals_dir():
    ad=active_artifact_dir(repo,p)
    current_round=_current_campaign_round()
    if ad is None or current_round is None:
        return None
    return (ad/'rounds'/str(current_round)/'debate'/'proposals').resolve()
def _path_under(path,base):
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
def _mentions_debate_proposal_artifact():
    base=_current_round_proposals_dir()
    if base is None:
        return False
    return any(_path_under(path,base) for path in _payload_paths())
def _mentions_stage3_transition_or_reconciliation_artifact():
    haystack='\n'.join(_payload_texts())
    ad=active_artifact_dir(repo,p)
    current_round=_current_campaign_round()
    if ad is not None and current_round is not None:
        blocked_paths={
            (ad/'state.json').resolve(),
            (ad/'rounds'/str(current_round)/'debate'/'summary.md').resolve(),
            (ad/'debate'/f'campaign_round_{current_round}'/'summary.md').resolve(),
            (ad/'debate'/f'campaign_round_{current_round}'/'selected_winners.json').resolve(),
        }
        if any(path in blocked_paths for path in _payload_paths()):
            return True
    blocked=[
        r'(?:^|[\\/\s\'"`])state\.json(?:$|[\s\'"`])',
        r'(?:^|[\\/\s\'"`])rounds[\\/]\d+[\\/]debate[\\/]summary\.md(?:$|[\s\'"`])',
        r'(?:^|[\\/\s\'"`])debate[\\/]campaign_round_\d+[\\/]summary\.md(?:$|[\s\'"`])',
        r'(?:^|[\\/\s\'"`])debate[\\/]campaign_round_\d+[\\/]selected_winners\.json(?:$|[\s\'"`])',
        r'current_stage',
        r'selected_winners',
        r'selection_rationale',
        r'rounds_completed',
        r'completed_at',
    ]
    return any(re.search(pattern,haystack) for pattern in blocked)
def _tool_may_continue_under_stage3_reconciliation_gate():
    if is_static_inspection_command(cmd):
        return True
    return _mentions_debate_proposal_artifact() and not _mentions_stage3_transition_or_reconciliation_artifact()
pending=blocking_monitor_records(repo,p)
blocking=[]; monitor_ack_allowed=False
if pending:
    blocking=[record for record in pending if not (_stage3_reconciliation_gate(record) and _tool_may_continue_under_stage3_reconciliation_gate())]
    monitor_ack_allowed=_is_monitor_ack_command(cmd,blocking) if blocking else False
    if blocking and os.environ.get('AMMO_MONITOR_ALLOW_UNACKED')!='1' and not monitor_ack_allowed:
        if cmd and 'monitor_queue_ack.py' not in cmd and is_static_inspection_command(cmd):
            raise SystemExit(0)
        first=str(blocking[0].get('_record_id') or '<record_id>')
        queue=str(blocking[0].get('_queue_path') or '<queue_path>')
        sid=record_session_id(blocking[0]) or session_id_from_payload(p,repo) or '<target_session_id>'
        ack_script=Path(__file__).resolve().parents[1]/'skills'/'ammo'/'scripts'/'monitor_queue_ack.py'
        ack_cmd='/usr/bin/python3 '+shlex.quote(str(ack_script))+' --session-id '+shlex.quote(sid)+' --queue '+shlex.quote(queue)+' --record-id '+shlex.quote(first)+' --note "<what changed or why rebutted>"'
        block_pretool('Pending AMMO monitor intervention requires acknowledgement before the next intercepted tool call.\n'+format_monitor_records(blocking,limit=2)+'\nTo acknowledge after reading, run: '+ack_cmd); raise SystemExit(0)
if monitor_ack_allowed: raise SystemExit(0)
if cmd and _monitor_ack_script_mentioned(cmd):
    if not pending and _monitor_ack_file_inspection(cmd): raise SystemExit(0)
    block_pretool('AMMO monitor acknowledgement denied: no matching current blocking record for the exact session, absolute queue, and record binding.'); raise SystemExit(0)
if not cmd: raise SystemExit(0)
if not is_ammo_context(cmd,repo): raise SystemExit(0)
if is_static_inspection_command(cmd): raise SystemExit(0)
if '.claude/skills/ammo' in cmd or '.claude/worktrees' in cmd: block_pretool('Use .codex/skills/ammo and .codex/worktrees, not .claude paths.'); raise SystemExit(0)
warnings=[]
if command_invokes_vllm_bench_latency(cmd) and 'run_vllm_bench_latency_sweep.py' not in cmd: warnings.append('AMMO reminder: raw vllm bench latency detected. Use run_vllm_bench_latency_sweep.py for official AMMO evidence.')
if any(re.search(x,cmd) for x in [r'TORCH_COMPILE_DISABLE\s*=\s*1',r'VLLM_TORCH_COMPILE_LEVEL\s*=\s*[01]\b',r'--enforce-eager\b',r'--disable-cuda-graph\b']) and os.environ.get('AMMO_ALLOW_NONPARITY')!='1' and 'AMMO_ALLOW_NONPARITY=1' not in cmd: warnings.append('AMMO reminder: command appears to disable production parity. Official AMMO evidence should use CUDA graphs/torch.compile parity.')
def _profiles_official_sweep(command):
    if 'run_vllm_bench_latency_sweep.py' not in command:
        return False
    if not any(flag in command for flag in ('--nsys-profile', '--torch-profile')):
        return False
    slot_match=re.search(r'--slot(?:=|\s+)(["\']?)([^\s"\']+)\1',command)
    if slot_match:
        slot=slot_match.group(2)
        return (
            slot in {'baseline','integration','golden_capture'}
            or slot.startswith(('opt/','opt_correctness/'))
        )
    # Preserve the pre-v2 spelling for older campaign commands.
    return bool(re.search(r'--mode(?:=|\s+)(opt|optimized|compare|integration)(?:\s|$)',command))
if _profiles_official_sweep(cmd): block_pretool('Profiled sweep output is not admissible official E2E timing or correctness evidence. Use profiling, opt_profiling/{op_id}, or post_ship_profiling slots.'); raise SystemExit(0)
def _unreserved_gpu_sentinel():
    sid=state_session_id(repo) or os.environ.get('AMMO_SESSION_ID') or session_id_from_payload(p,repo)
    if not sid: return None
    safe=re.sub(r'[^A-Za-z0-9_.-]+','_',str(sid))[:96] or 'session'
    root=Path(os.environ.get('AMMO_GPU_RES_DIR') or '/tmp/ammo_gpu_res').expanduser()
    return root/f'codex_unreserved_gpu_guard_{safe}.seen'
def _has_explicit_no_gpu(command):
    return bool(re.search(r'CUDA_VISIBLE_DEVICES\s*=\s*(?:""|\'\')',command))
if command_invokes_gpu_heavy_tool(cmd) and 'gpu_reservation.py' not in cmd and not _has_explicit_no_gpu(cmd) and os.environ.get('AMMO_ALLOW_UNRESERVED_GPU')!='1' and 'AMMO_ALLOW_UNRESERVED_GPU=1' not in cmd:
    sentinel=_unreserved_gpu_sentinel()
    if sentinel is None:
        raise SystemExit(0)
    first_notice=not sentinel.exists()
    try:
        sentinel.parent.mkdir(parents=True,exist_ok=True)
        sentinel.touch(exist_ok=True)
    except Exception:
        pass
    if first_notice:
        reason='GPU-heavy AMMO command detected without reservation. Use gpu_reservation.py reserve, CUDA_VISIBLE_DEVICES="" for explicit no-GPU commands, or AMMO_ALLOW_UNRESERVED_GPU=1 with justification. This warning fires only once per session.'
        if warnings:
            reason += '\n\n' + '\n'.join(warnings)
        block_pretool(reason)
        raise SystemExit(0)
if warnings:
    additional_context('PreToolUse','\n'.join(warnings))
