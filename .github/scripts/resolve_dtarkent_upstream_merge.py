from __future__ import annotations
import subprocess
from pathlib import Path

ROOT = Path.cwd()

def read(p): return (ROOT/p).read_text(encoding="utf-8")
def write(p,s): (ROOT/p).write_text(s,encoding="utf-8",newline="\n")

def resolve_blocks(path, chooser):
    s=read(path); out=[]; pos=0; i=0
    while True:
        a=s.find("<<<<<<< HEAD", pos)
        if a<0:
            out.append(s[pos:]); break
        out.append(s[pos:a])
        a_nl=s.find("\n",a)+1
        mid=s.find("\n=======\n",a_nl)
        end=s.find("\n>>>>>>> upstream/main",mid)
        if mid<0 or end<0: raise RuntimeError(f"Malformed conflict in {path}")
        ours=s[a_nl:mid]
        theirs=s[mid+9:end]
        out.append(chooser(i,ours,theirs))
        i+=1
        end_nl=s.find("\n",end+1)
        pos=len(s) if end_nl<0 else end_nl+1
    write(path,"".join(out))
    print(path, "resolved", i)

def bg(i, ours, theirs):
    assert i==0
    return '''                from agent.background_review_breaker import install_review_breaker

                _breaker = install_review_breaker(
                    review_agent,
                    threshold=max(2, _resolve_review_max_iterations(task_cfg) // 2),
                )
            except Exception:
                logger.debug(
                    "review refusal breaker install failed (continuing without)",
                    exc_info=True,
                )

            try:
                request_admitted = (
                    review_run is None or review_run.begin_request(review_agent)'''

def gateway(i, ours, theirs):
    assert i==0
    old="def validate_media_delivery_path(path: str) -> Optional[str]:"
    new='def validate_media_delivery_path(path: str, session_key: str = "") -> Optional[str]:'
    if old not in ours: raise RuntimeError("gateway signature not found in ours")
    return ours.replace(old,new,1)

def both_ours_theirs(i,ours,theirs):
    return ours.rstrip()+"\n\n"+theirs.lstrip()

def both_theirs_ours(i,ours,theirs):
    return theirs.rstrip()+"\n\n"+ours.lstrip()

resolve_blocks(Path("agent/background_review.py"), bg)
p=Path("agent/background_review.py")
s=read(p)
old='''                    _review_history = (
                        _digest_history(messages_snapshot) if _routed
                        else messages_snapshot
                    )'''
new='''                    _review_history = _review_history_for_fork(
                        messages_snapshot,
                        routed=_routed,
                        parent_cache_effective=_parent_cache_effective(agent),
                    )'''
if old not in s:
    raise RuntimeError("upstream review history block not found")
write(p,s.replace(old,new,1))

resolve_blocks(Path("agent/codex_responses_adapter.py"), lambda i,o,t:t)
resolve_blocks(Path("agent/context_compressor.py"), lambda i,o,t:t)
resolve_blocks(Path("agent/conversation_compression.py"), lambda i,o,t:o)
resolve_blocks(Path("gateway/platforms/base.py"), gateway)
resolve_blocks(Path("tests/agent/test_codex_responses_adapter.py"), lambda i,o,t:t)
resolve_blocks(Path("tests/agent/test_compaction_anti_thrash.py"), lambda i,o,t:t)
resolve_blocks(Path("tests/agent/test_would_grow_refusal_runway.py"), lambda i,o,t:t)
resolve_blocks(Path("tests/cron/test_cron_script.py"), both_ours_theirs)
resolve_blocks(Path("tests/hermes_cli/test_dashboard_unified_launch.py"), both_theirs_ours)

r=subprocess.run(["git","diff","--name-only","--diff-filter=U"],capture_output=True,text=True,check=True)
for name in r.stdout.splitlines():
    txt=read(Path(name))
    if "<<<<<<<" in txt or "\n=======\n" in txt or "\n>>>>>>>" in txt:
        raise RuntimeError(f"Conflict markers remain in {name}")

subprocess.run(["git","diff","--check"],check=True)
print("resolver complete")
