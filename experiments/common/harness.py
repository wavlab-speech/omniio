"""Reproducibility record written next to every result: code version, library
versions, node, and whether the node was shared."""
import json
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Dict


def _git_commit(path: str) -> str:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _version(mod: str) -> str:
    try:
        return __import__(mod).__version__
    except Exception:
        return "n/a"


def node_shared() -> Any:
    """True when other users' processes run on this node (Slurm shared node)."""
    try:
        out = subprocess.run(["ps", "-eo", "user=", "--no-headers"], capture_output=True, text=True, timeout=10).stdout.split()
        me = os.environ.get("USER", "")
        others = {u for u in out if u not in (me, "root") and not u.startswith(("message", "chrony", "rpc", "postfix", "telegraf", "munge", "dbus", "polkit", "systemd", "nscd", "ntp", "avahi", "sshd", "nobody", "daemon", "_", "statd", "lp", "mail"))}
        return sorted(others) or False
    except Exception:
        return None


def provenance(repo: str = None) -> Dict[str, Any]:
    repo = repo or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "hostname": socket.gethostname(), "arch": platform.machine(),
            "python": sys.version.split()[0], "pyarrow": _version("pyarrow"), "av": _version("av"), "torch": _version("torch"),
            "numpy": _version("numpy"), "soundfile": _version("soundfile"), "git_commit": _git_commit(repo),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "node_shared_with": node_shared(),
            "omniio_window_repo": os.environ.get("OMNIIO_WINDOW_REPO")}


def write_result(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(payload)
    payload["provenance"] = provenance()
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
