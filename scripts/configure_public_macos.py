#!/usr/bin/env python3
"""Install the LegalChatModel origin and Cloudflare Tunnel as user services.

The Cloudflare tunnel token is requested without echo and stored outside the
repository with mode 0600.  No model weights, user records, or secrets are
copied to GitHub or Cloudflare by this installer.
"""

from __future__ import annotations

import argparse
import getpass
import os
import plistlib
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path.home() / "Library" / "Application Support" / "LegalAI-public"
LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
ORIGIN_LABEL = "ai.legalchatmodel.origin"
TUNNEL_LABEL = "ai.legalchatmodel.tunnel"


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _validate_team_domain(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path:
        raise argparse.ArgumentTypeError(
            "team domain must be an HTTPS origin, for example "
            "https://your-team.cloudflareaccess.com"
        )
    return value


def _validate_aud(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 512 or any(ch.isspace() for ch in value):
        raise argparse.ArgumentTypeError("AUD must be a non-empty tag without spaces")
    return value


def _find_cloudflared(data_dir: Path, override: str | None) -> Path:
    candidates = [
        Path(override).expanduser() if override else None,
        Path(shutil.which("cloudflared")) if shutil.which("cloudflared") else None,
        data_dir / "bin" / "cloudflared",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise SystemExit(
        "cloudflared was not found. Install it first or pass --cloudflared /path/to/cloudflared."
    )


def _origin_port_is_busy() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8765), timeout=0.25):
            return True
    except OSError:
        return False


def _agent_is_loaded(label: str) -> bool:
    domain = f"gui/{os.getuid()}"
    return _launchctl("print", f"{domain}/{label}", check=False).returncode == 0


def _read_tunnel_token(token_file: str | None) -> str:
    if token_file:
        token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
    elif not sys.stdin.isatty():
        token = sys.stdin.read().strip()
    else:
        token = getpass.getpass("Paste the Cloudflare tunnel token (input hidden): ").strip()
    if len(token) < 50 or any(ch.isspace() for ch in token):
        raise SystemExit("The tunnel token is empty or malformed; no service was installed.")
    return token


def _plist(label: str, arguments: list[str], log_dir: Path) -> dict[str, object]:
    return {
        "Label": label,
        "ProgramArguments": arguments,
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "ProcessType": "Background",
        "StandardOutPath": str(log_dir / f"{label}.stdout.log"),
        "StandardErrorPath": str(log_dir / f"{label}.stderr.log"),
        "EnvironmentVariables": {
            "HOME": str(Path.home()),
            "LEGAL_PUBLIC_DATA_DIR": str(log_dir.parent),
            "PYTHONUNBUFFERED": "1",
        },
    }


def _launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _replace_agent(label: str, plist_path: Path) -> None:
    domain = f"gui/{os.getuid()}"
    _launchctl("bootout", f"{domain}/{label}", check=False)
    result = _launchctl("bootstrap", domain, str(plist_path), check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise SystemExit(f"Could not start {label}: {detail}")
    _launchctl("enable", f"{domain}/{label}", check=False)


def configure(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.expanduser().resolve()
    log_dir = data_dir / "logs"
    token_path = data_dir / "secrets" / "tunnel-token"
    env_path = data_dir / "public.env"
    cloudflared = _find_cloudflared(data_dir, args.cloudflared)
    if not args.no_start and _origin_port_is_busy() and not _agent_is_loaded(ORIGIN_LABEL):
        raise SystemExit(
            "127.0.0.1:8765 is already used by another local service. Stop the "
            "personal/local LegalChatModel launcher before installing public mode; "
            "otherwise the tunnel could reach the wrong data and authentication mode."
        )
    token = _read_tunnel_token(args.token_file)

    for folder in (data_dir, log_dir, data_dir / "feedback", data_dir / "uploads"):
        folder.mkdir(parents=True, exist_ok=True)
        os.chmod(folder, 0o700)

    _atomic_write(token_path, (token + "\n").encode(), 0o600)
    del token

    env_lines = [
        f"export CF_ACCESS_TEAM_DOMAIN={shlex.quote(args.team_domain)}",
        f"export CF_ACCESS_AUD={shlex.quote(args.aud)}",
        f"export LEGAL_PUBLIC_DATA_DIR={shlex.quote(str(data_dir))}",
        "",
    ]
    _atomic_write(env_path, "\n".join(env_lines).encode(), 0o600)

    origin_plist = LAUNCH_AGENTS / f"{ORIGIN_LABEL}.plist"
    tunnel_plist = LAUNCH_AGENTS / f"{TUNNEL_LABEL}.plist"
    LAUNCH_AGENTS.mkdir(parents=True, exist_ok=True)

    origin = _plist(
        ORIGIN_LABEL,
        ["/bin/zsh", str(ROOT / "scripts" / "public_chat_ui.sh")],
        log_dir,
    )
    tunnel = _plist(
        TUNNEL_LABEL,
        [
            str(cloudflared),
            "tunnel",
            "--no-autoupdate",
            "run",
            "--token-file",
            str(token_path),
        ],
        log_dir,
    )
    _atomic_write(origin_plist, plistlib.dumps(origin), 0o600)
    _atomic_write(tunnel_plist, plistlib.dumps(tunnel), 0o600)

    if not args.no_start:
        _replace_agent(ORIGIN_LABEL, origin_plist)
        _replace_agent(TUNNEL_LABEL, tunnel_plist)

    print("Public service configuration installed.")
    print(f"Data: {data_dir}")
    print(f"Logs: {log_dir}")
    print("Tunnel token: stored locally with mode 0600 (not printed).")
    if args.no_start:
        print("Services were not started (--no-start).")
    else:
        print("Origin and tunnel launch agents are running and will restart at login.")


def status(args: argparse.Namespace) -> None:
    domain = f"gui/{os.getuid()}"
    for label in (ORIGIN_LABEL, TUNNEL_LABEL):
        result = _launchctl("print", f"{domain}/{label}", check=False)
        state = "installed/running" if result.returncode == 0 else "not loaded"
        print(f"{label}: {state}")
    data_dir = args.data_dir.expanduser().resolve()
    print(f"Origin log: {data_dir / 'logs' / f'{ORIGIN_LABEL}.stderr.log'}")
    print(f"Tunnel log: {data_dir / 'logs' / f'{TUNNEL_LABEL}.stderr.log'}")


def uninstall(args: argparse.Namespace) -> None:
    domain = f"gui/{os.getuid()}"
    for label in (ORIGIN_LABEL, TUNNEL_LABEL):
        _launchctl("bootout", f"{domain}/{label}", check=False)
        (LAUNCH_AGENTS / f"{label}.plist").unlink(missing_ok=True)
    if args.delete_data:
        raise SystemExit(
            "Service files were removed. Public user data was NOT deleted; "
            "delete it only through the app's account/retention controls or a reviewed migration."
        )
    print("Public services removed. Public SQL, uploads, feedback, and secrets were retained.")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("LEGAL_PUBLIC_DATA_DIR", DEFAULT_DATA_DIR)),
    )
    sub = ap.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="write and optionally start both launch agents")
    install.add_argument("--team-domain", required=True, type=_validate_team_domain)
    install.add_argument("--aud", required=True, type=_validate_aud)
    install.add_argument("--cloudflared")
    install.add_argument("--token-file", help="read the token from this file; otherwise prompt/STDIN")
    install.add_argument("--no-start", action="store_true")
    install.set_defaults(func=configure)

    show = sub.add_parser("status", help="show whether the two launch agents are loaded")
    show.set_defaults(func=status)

    remove = sub.add_parser("uninstall", help="stop services but retain all public records")
    remove.add_argument("--delete-data", action="store_true", help=argparse.SUPPRESS)
    remove.set_defaults(func=uninstall)
    return ap


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
