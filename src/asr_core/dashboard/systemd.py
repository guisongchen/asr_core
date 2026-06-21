import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServiceState:
    name: str
    active: bool
    status: str
    uptime: Optional[str]


class SystemdManager:
    """Thin wrapper around `systemctl --user` for service management."""

    SERVICES = [
        "asr-core",
        "asr-core-dashboard",
        "voice-to-text",
        "lp998-listener",
    ]

    def __init__(self, user: bool = True):
        self.user = user
        self._base = ["systemctl", "--user"] if user else ["systemctl"]

    def _run(self, *args, capture: bool = True, check: bool = False):
        cmd = [*self._base, *args]
        try:
            return subprocess.run(
                cmd, capture_output=capture, text=True, check=check, timeout=10
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="timeout")
        except FileNotFoundError:
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="systemctl not found")

    def status(self, name: str) -> ServiceState:
        result = self._run("is-active", name)
        active = result.returncode == 0 and result.stdout.strip() == "active"

        result = self._run("show", name, "--property=ActiveState,SubState,UptimeMillis")
        props = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                props[key] = value

        uptime_ms = props.get("UptimeMillis", "")
        uptime = self._format_uptime(uptime_ms) if uptime_ms else None

        return ServiceState(
            name=name, active=active, status=props.get("SubState", "unknown"), uptime=uptime
        )

    def all_statuses(self) -> list[ServiceState]:
        return [self.status(name) for name in self.SERVICES]

    def start(self, name: str) -> str:
        if name not in self.SERVICES:
            raise ValueError(f"Unknown service: {name}")
        result = self._run("start", name, check=False)
        return result.stderr.strip() or result.stdout.strip() or "ok"

    def stop(self, name: str) -> str:
        if name not in self.SERVICES:
            raise ValueError(f"Unknown service: {name}")
        result = self._run("stop", name, check=False)
        return result.stderr.strip() or result.stdout.strip() or "ok"

    def restart(self, name: str) -> str:
        if name not in self.SERVICES:
            raise ValueError(f"Unknown service: {name}")
        result = self._run("restart", name, check=False)
        return result.stderr.strip() or result.stdout.strip() or "ok"

    def logs(self, name: str, lines: int = 50) -> str:
        if name not in self.SERVICES:
            raise ValueError(f"Unknown service: {name}")
        journalctl = shutil.which("journalctl")
        if not journalctl:
            return "journalctl not available"
        base = ["journalctl", "--user"] if self.user else ["journalctl"]
        result = subprocess.run(
            [*base, "-u", name, "-n", str(lines), "--no-pager"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout or result.stderr

    @staticmethod
    def _format_uptime(millis: str) -> Optional[str]:
        try:
            ms = int(millis)
        except ValueError:
            return None
        seconds = ms // 1000
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h {minutes % 60}m"
        days = hours // 24
        return f"{days}d {hours % 24}h"
