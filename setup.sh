#!/usr/bin/env bash
# kb-agent bootstrap — self-locating, runtime-agnostic (OpenClaw / Hermes / standalone).
# Creates a venv inside the skill directory and installs the package editable.
# Usage: bash <skill_dir>/setup.sh   (or just ./setup.sh)
set -euo pipefail

# Prevent host environment from contaminating the venv. Hermes / OpenClaw /
# shell profiles often set PYTHONPATH (or PIP_TARGET/PREFIX) to point at their
# own site-packages — if it leaks through, pip installs deps into the wrong
# interpreter's dir and the CLI breaks with native-extension import errors.
unset PYTHONPATH 2>/dev/null || true
unset PIP_TARGET 2>/dev/null || true
unset PIP_PREFIX 2>/dev/null || true

# Resolve the skill directory from this script's own location — works no matter
# where the skill is installed (~/.hermes/skills/, ~/.openclaw/workspace-coding/skills/, ~/my-skills/, etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 1. Pick a Python interpreter (>=3.10) ----------------------------------
# Respect an explicit override first, then probe common binaries by version.
pick_python() {
    if [ -n "${KB_AGENT_PYTHON:-}" ]; then
        echo "$KB_AGENT_PYTHON"
        return
    fi
    for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 \
           && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            echo "$candidate"
            return
        fi
    done
    echo ""
}

PYTHON="$(pick_python)"
if [ -z "$PYTHON" ]; then
    echo "ERROR: no Python >=3.10 found. Install one or set KB_AGENT_PYTHON=/path/to/python3." >&2
    exit 1
fi
echo "Using Python: $("$PYTHON" --version 2>&1) ($(command -v "$PYTHON"))"

# ---- 2. Create venv if missing ----------------------------------------------
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "Creating virtual environment ..."
    "$PYTHON" -m venv "$SCRIPT_DIR/.venv"
fi

# ---- 3. Bootstrap pip if missing (macOS system Python often lacks it) -------
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    echo "pip missing — bootstrapping via ensurepip ..."
    "$VENV_PY" -m ensurepip --upgrade 2>/dev/null \
        || curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_PY"
fi

# ---- 4. Install (absolute path — never rely on PATH/activate) ---------------
# Prefer a fast PyPI mirror (CN direct), fall back to a local proxy, then
# plain direct. Mirrors are far faster and more reliable on throttled/GFW
# networks than a proxy tunnel. Respect an existing HTTP_PROXY/HTTPS_PROXY.
_pip_flags=""

# Tier 1: PyPI mirrors (fastest in CN) — override with KB_AGENT_PIP_INDEX
if [ -n "${KB_AGENT_PIP_INDEX:-}" ]; then
    _pip_flags="-i $KB_AGENT_PIP_INDEX"
    echo "Using PyPI index: $KB_AGENT_PIP_INDEX"
elif [ -z "${HTTP_PROXY:-}" ] && [ -z "${HTTPS_PROXY:-}" ]; then
    for _m in "https://pypi.tuna.tsinghua.edu.cn/simple" "https://mirrors.aliyun.com/pypi/simple"; do
        if curl -sI --max-time 3 "$_m/pip/" >/dev/null 2>&1; then
            _pip_flags="-i $_m"
            echo "Using PyPI mirror: $_m"
            break
        fi
    done
fi

# Tier 2: mirror unreachable → probe a local proxy (Clash/V2Ray common ports)
if [ -z "$_pip_flags" ] && [ -z "${HTTP_PROXY:-}" ] && [ -z "${HTTPS_PROXY:-}" ]; then
    for _p in 7890 7897 1087 8080; do
        if curl --proxy "http://127.0.0.1:$_p" -sI --max-time 3 https://pypi.org >/dev/null 2>&1; then
            _pip_flags="--proxy http://127.0.0.1:$_p"
            echo "Using proxy: http://127.0.0.1:$_p"
            break
        fi
    done
fi

# Tier 3: neither → direct (flags empty)

"$VENV_PY" -m pip install --upgrade pip $_pip_flags
"$VENV_PY" -m pip install -e "$SCRIPT_DIR" $_pip_flags
# PDF support (pymupdf) — separate install is faster than re-installing the
# whole editable package with the [pdf] extra.
"$VENV_PY" -m pip install $_pip_flags pymupdf || echo "  (pymupdf install skipped — PDF support optional)"

echo
echo "✓ kb-agent ready."
echo "  venv:     $SCRIPT_DIR/.venv"
echo "  activate: source $SCRIPT_DIR/.venv/bin/activate"
echo "  CLI:      $VENV_PY -m kb_agent.tools.cli <cmd> <args>"