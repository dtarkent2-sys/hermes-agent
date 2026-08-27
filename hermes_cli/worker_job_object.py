"""Windows Job Object containment for dispatcher-spawned kanban workers.

**The incident this fixes (2026-08-27, engineer-lane crash loop):**

A kanban worker run crashed while running ``pytest tests/hermes_cli/``. On
Windows, ``_default_spawn`` launches workers with ``start_new_session=True``,
which is *silently ignored* — there is no POSIX process-group equivalent, so
when the worker process died uncleanly (memory exhaustion → WER
RADAR_PRE_LEAK_64 → unclean exit), its children were **orphaned and kept
running**. The orphaned pytest ballooned 15.9GB → 20.3GB, and the resulting
memory pressure killed *live* workers on unrelated tasks, producing the
"pid not alive" crash loop (runs 496/499/511/512 and prior incidents).

**The fix — kernel-enforced tree teardown:**

Every dispatcher-spawned kanban worker creates a Win32 Job Object with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` and assigns *itself* to it at
startup. The job handle stays open in the worker process; every descendant
the worker ever spawns (pytest, node, git, delegation children, …) is
automatically a job member. When the worker process terminates — by ANY
mechanism: clean exit, os._exit, OOM kill, WER, taskkill, access violation —
its handle to the job closes, and the **kernel immediately terminates every
process in the job**. There is no dispatcher detection race and no window in
which an orphan can accumulate memory.

POSIX needs none of this: workers spawn with ``start_new_session=True`` and
the kernel reaps the whole tree via process-group semantics; this module is
a no-op there.

**Why self-assignment is safe on modern Windows:**

A process can only ever join ONE job. Two cases:

1. The spawner (gateway/desktop) already put us in a job → the call fails
   with ERROR_ACCESS_DENIED and we degrade gracefully (see below).
2. We are jobless (the common dispatcher case — the gateway's own children
   break away via ``CREATE_BREAKAWAY_FROM_JOB``) → assignment succeeds.

Either way the failure mode is the pre-existing behavior (children may
orphan if the worker dies) — never a new one.

**Design constraints (do not loosen without re-reading the incident):**

- ``HERMES_KANBAN_TASK`` gates activation: ONLY dispatcher-spawned workers
  self-job. An interactive ``hermes`` CLI session that spawns a pytest must
  NOT have its whole terminal session torn down by an unrelated child exit.
- Best-effort by contract: any Win32 error → log at debug, return False.
  Job Object setup must never break worker startup — the dispatcher's claim
  TTL and crash detection remain the backstop for the no-job case.
- The handle is intentionally leaked (never closed explicitly): keeping it
  open is the mechanism. Closing it would kill the worker's own tree.

Requires no third-party dependencies: ctypes against kernel32 only.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# Set once the worker's job is armed. Diagnostic, not user config.
job_object_armed = False

# Win32 constants (job object docs, winnt.h / winbase.h)
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x00001000
_JOB_NAME_KANBAN_WORKER = "HermesKanbanWorker"

# Diagnostics from the last arm attempt ("ok", reason strings).
_last_arm_result: str = "not attempted"


def arm_kanban_worker_job_object() -> bool:
    """Create + enter a kill-on-close Job Object for this kanban worker.

    Called from the CLI entry path when ``HERMES_KANBAN_TASK`` is set.
    Idempotent: a second call is a no-op (a process can only be in one
    job, and we keep the module-level handle open for the process
    lifetime).

    Returns True when the worker is now contained in a kill-on-close job,
    False otherwise (non-Windows, not a worker, or best-effort failure —
    all of which leave the pre-existing orphan behavior unchanged).
    """
    global job_object_armed, _last_arm_result

    if job_object_armed:
        _last_arm_result = "ok"
        return True
    if not _IS_WINDOWS:
        _last_arm_result = "non-windows"
        return False
    if not (os.environ.get("HERMES_KANBAN_TASK") or "").strip():
        # Not a dispatcher-spawned worker: containment would be wrong for
        # interactive sessions (see module docstring).
        _last_arm_result = "not-a-worker"
        return False

    import ctypes
    from ctypes import wintypes

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(wintypes.ULONG)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9  # JOBOBJECTINFOCLASS enum value

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    CreateJobObjectW = kernel32.CreateJobObjectW
    CreateJobObjectW.restype = wintypes.HANDLE
    CreateJobObjectW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]

    SetInformationJobObject = kernel32.SetInformationJobObject
    SetInformationJobObject.restype = wintypes.BOOL
    SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]

    AssignProcessToJobObject = kernel32.AssignProcessToJobObject
    AssignProcessToJobObject.restype = wintypes.BOOL
    AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    GetCurrentProcess = kernel32.GetCurrentProcess
    GetCurrentProcess.restype = wintypes.HANDLE
    GetCurrentProcess.argtypes = []

    # A named job would let two simultaneous workers collide on one shared
    # object (the name is a global namespace handle) — and error 998
    # (ERROR_NOACCESS) shows this kernel rejects the named create via this
    # signature anyway. An ANONYMOUS job is correct for a per-worker
    # containment: it is referenced only by the handle we keep open, dies
    # with the worker process, and can never be shared by accident.
    job = CreateJobObjectW(None, None)
    if not job:
        _last_arm_result = f"create-failed winerror={ctypes.get_last_error()}"
        logger.debug("kanban worker job object: %s", _last_arm_result)
        return False

    # KILL_ON_JOB_CLOSE is the whole point: when the worker process dies,
    # its (only) handle to the job closes and the kernel terminates every
    # remaining member — the worker's whole descendant tree — instantly.
    # BREAKAWAY_OK lets a child that must outlive the worker (none are
    # known today; future-proofing) still escape via the documented
    # CreateProcess breakaway flags rather than being silently trapped.
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | 0x00000800
    )
    ok = SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        _last_arm_result = f"setinfo-failed winerror={ctypes.get_last_error()}"
        logger.debug("kanban worker job object: %s", _last_arm_result)
        ctypes.windll.kernel32.CloseHandle(job)
        return False

    # Self-assignment: this process joins its own job. Every child it
    # spawns from here on is a member. On success the worker must NEVER
    # close this handle — the open handle IS the kill switch.
    if not AssignProcessToJobObject(job, GetCurrentProcess()):
        _last_arm_result = f"assign-failed winerror={ctypes.get_last_error()}"
        logger.debug("kanban worker job object: %s", _last_arm_result)
        ctypes.windll.kernel32.CloseHandle(job)
        return False

    job_object_armed = True
    _last_arm_result = "ok"
    logger.debug(
        "kanban worker job object armed (kill-on-close, tree teardown on death)"
    )
    return True


def job_object_status() -> str:
    """Human-readable diagnostic of the last arm attempt (for `hermes doctor`)."""
    return _last_arm_result