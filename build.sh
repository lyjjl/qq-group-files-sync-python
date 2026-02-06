#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Only Linux is supported by this build script." >&2
  exit 1
fi

ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in
  x86_64) ARCH="amd64" ;;
  aarch64|arm64) ARCH="arm64" ;;
  *) echo "Unsupported architecture: $ARCH_RAW" >&2; exit 1 ;;
esac

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY_REQ_MINOR="${PY_REQ_MINOR:-3.13}" # default to 3.13 (Nuitka 用不了 3.14)
CLEAN="${CLEAN:-0}"
NUITKA_MODE="${NUITKA_MODE:-onefile}"
ONEFILE_NO_COMPRESSION="${ONEFILE_NO_COMPRESSION:-0}"
RELEASE=0

print_help() {
  cat <<'USAGE'
Usage: ./build.sh [--release]

  --release  Enable maximum optimization + compression.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      RELEASE=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_help >&2
      exit 1
      ;;
  esac
done

TOOLS_DIR="${TOOLS_DIR:-$ROOT_DIR/.tools}"
UV_HOME="${UV_HOME:-$TOOLS_DIR/uv}"
UV_BIN="$UV_HOME/uv"
UV_CACHE_DIR="${UV_CACHE_DIR:-$TOOLS_DIR/uv-cache}"
UV_PY_INSTALL_DIR="${UV_PY_INSTALL_DIR:-$TOOLS_DIR/uv-python}"
UV_PY_BIN_DIR="${UV_PY_BIN_DIR:-$TOOLS_DIR/uv-python-bin}"

if [[ "$CLEAN" == "1" ]]; then
  rm -rf .venv dist "$TOOLS_DIR"
fi

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_uv() {
  if [[ -x "$UV_BIN" ]]; then
    return 0
  fi
  mkdir -p "$TOOLS_DIR"

  if need_cmd curl; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_HOME" sh
  elif need_cmd wget; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_HOME" sh
  else
    echo "Need curl or wget to bootstrap uv." >&2
    exit 1
  fi

  if [[ ! -x "$UV_BIN" ]]; then
    echo "uv bootstrap failed: $UV_BIN not found/executable" >&2
    exit 1
  fi
}

pick_python_313() {
  if need_cmd "python${PY_REQ_MINOR}"; then
    echo "python${PY_REQ_MINOR}"
    return 0
  fi

  ensure_uv
  mkdir -p "$UV_CACHE_DIR" "$UV_PY_INSTALL_DIR" "$UV_PY_BIN_DIR"

  env \
    UV_CACHE_DIR="$UV_CACHE_DIR" \
    UV_PYTHON_INSTALL_DIR="$UV_PY_INSTALL_DIR" \
    UV_PYTHON_BIN_DIR="$UV_PY_BIN_DIR" \
    "$UV_BIN" python install "$PY_REQ_MINOR"

  local py_path
  py_path="$(env \
    UV_CACHE_DIR="$UV_CACHE_DIR" \
    UV_PYTHON_INSTALL_DIR="$UV_PY_INSTALL_DIR" \
    UV_PYTHON_BIN_DIR="$UV_PY_BIN_DIR" \
    "$UV_BIN" python find "$PY_REQ_MINOR")"

  if [[ -z "$py_path" || ! -x "$py_path" ]]; then
    echo "uv-managed Python not found after install." >&2
    exit 1
  fi
  echo "$py_path"
}

PYTHON_BIN="${PYTHON_BIN:-$(pick_python_313)}"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate || true
  if python - <<'PY' >/dev/null 2>&1
import os, sys
ver = os.environ.get("PY_REQ_MINOR", "3.13")
maj, minor = ver.split(".")
ok = (sys.version_info[0] == int(maj) and sys.version_info[1] == int(minor))
raise SystemExit(0 if ok else 1)
PY
  then
    :
  else
    deactivate >/dev/null 2>&1 || true
    rm -rf .venv
  fi
  deactivate >/dev/null 2>&1 || true
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -m pip --version >/dev/null 2>&1; then
  if python -m ensurepip --help >/dev/null 2>&1; then
    python -m ensurepip --upgrade
  else
    echo "pip missing inside venv; install distro python-venv/python-pip equivalents." >&2
    exit 1
  fi
fi

python -m pip install -U pip setuptools wheel

# deps
python -m pip install -U nuitka \
  "typer>=0.12" \
  "rich>=13.7" \
  "pydantic>=2.6" \
  "websockets>=12.0" \
  "httpx>=0.27.0" \
  "aiofiles>=23.2" \
  "jinja2>=3.1"

if [[ "$RELEASE" == "1" ]]; then
  python -m pip install -U zstandard
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_NAME="qq-sync-linux-${ARCH}"

NUITKA_ARGS=(
  --follow-imports
  --output-dir=dist
  --output-filename="$OUTPUT_NAME"
  --include-data-dir=src/templates=templates
  --include-package-data=rich
  --include-package=rich._unicode_data
  --include-module=rich._unicode_data.unicode17-0-0
  --include-package=websockets
  --include-module=websockets.asyncio
  --include-module=websockets.exceptions
)

if [[ "$NUITKA_MODE" == "onefile" ]]; then
  NUITKA_ARGS+=(--onefile)
  if [[ "$RELEASE" == "1" ]]; then
    NUITKA_ARGS+=(--lto=yes)
  elif [[ "$ONEFILE_NO_COMPRESSION" == "1" ]]; then
    NUITKA_ARGS+=(--onefile-no-compression)
  fi
elif [[ "$NUITKA_MODE" == "standalone" ]]; then
  NUITKA_ARGS+=(--standalone)
else
  echo "Unknown NUITKA_MODE: $NUITKA_MODE (use onefile/standalone)" >&2
  exit 1
fi

python -m nuitka "${NUITKA_ARGS[@]}" main.py

echo "Build complete: dist/$OUTPUT_NAME"
