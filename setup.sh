#!/usr/bin/env bash
# Setup script for QUIT Agent
# Creates a single .venv at the repo root, installs the agent and web UI.
#
# Usage:
#   bash setup.sh              # CPU-only (no local LLM inference)
#   bash setup.sh --with-local # also installs torch + transformers for local models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
AGENT_DIR="$SCRIPT_DIR/Quit_v0_3"
WEB_DIR="$SCRIPT_DIR/Quit_v0_3_web"

WITH_LOCAL=0
for arg in "$@"; do
    [[ "$arg" == "--with-local" ]] && WITH_LOCAL=1
done

echo "==> Creating virtual environment at $VENV_DIR"
# Prefer Anaconda Python to avoid missing C-extension issues on minimal systems
PYTHON_BIN="python3"
for candidate in /opt/anaconda/bin/python3.12 /opt/anaconda/bin/python3 /usr/local/bin/python3.12; do
    if "$candidate" -c "import socket" &>/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
echo "    Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
"$PYTHON_BIN" -m venv "$VENV_DIR"

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "==> Upgrading pip"
"$PIP" install --upgrade pip

# Remove any stale editable installs that also expose a quit_agent package,
# which would shadow the v0_3 source and cause hard-to-diagnose import errors.
echo "==> Removing stale quit_agent installs (if any)"
"$PIP" uninstall quit-agent hermes-research-os-v2 hermes_research_os_v2 -y 2>/dev/null || true

echo "==> Installing QUIT Agent (editable) with web UI dependencies"
"$PIP" install -e "$AGENT_DIR[web]"

# ── System tool checks ────────────────────────────────────────────────────────
echo "==> Checking system tools"

_check() {
    local tool=$1 label=$2 required=$3 hint_deb=$4 hint_mac=$5 hint_rpm=$6
    if command -v "$tool" &>/dev/null; then
        echo "    [OK] $label: $("$tool" --version 2>&1 | head -1)"
    elif [[ $required == "required" ]]; then
        echo ""
        echo "    [MISSING] $label — required, PDF generation will fail without it."
        echo "      Debian/Ubuntu : $hint_deb"
        echo "      macOS         : $hint_mac"
        echo "      RHEL/CentOS   : $hint_rpm"
        echo ""
    else
        echo "    [OPTIONAL] $label not found (only needed for: $required)."
        echo "      Debian/Ubuntu : $hint_deb"
    fi
}

# LaTeX: prefer latexmk; fall back to pdflatex/xelatex/lualatex
if command -v latexmk &>/dev/null; then
    echo "    [OK] latexmk: $(latexmk --version 2>&1 | head -1)"
elif command -v pdflatex &>/dev/null || command -v xelatex &>/dev/null || command -v lualatex &>/dev/null; then
    echo "    [OK] pdflatex/xelatex/lualatex found (latexmk missing — install for best results)"
    echo "      Debian/Ubuntu : sudo apt-get install -y latexmk"
else
    echo ""
    echo "    [MISSING] LaTeX compiler — required, PDF generation will fail without it."
    echo "      Debian/Ubuntu : sudo apt-get install -y texlive-full latexmk"
    echo "      macOS         : brew install --cask mactex"
    echo "      RHEL/CentOS   : sudo yum install -y texlive texlive-latexmk"
    echo ""
fi

# bibtex ships with texlive; check separately in case of partial installs
_check bibtex "bibtex" \
    "required" \
    "sudo apt-get install -y texlive-bibtex-extra" \
    "included in MacTeX" \
    "sudo yum install -y texlive-bibtex"

# git: needed when the agent clones reference repos
_check git "git" \
    "cloning reference repos (BUILD_SPEC / CODE stage)" \
    "sudo apt-get install -y git" \
    "brew install git" \
    "sudo yum install -y git"

if [[ $WITH_LOCAL -eq 1 ]]; then
    echo "==> Installing local LLM dependencies (torch + transformers)"
    "$PIP" install -e "$AGENT_DIR[local]"
fi

echo ""
echo "Setup complete."
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To start the web UI:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $WEB_DIR && python server.py"
echo ""
echo "To run the agent directly:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $AGENT_DIR && quit-agent --config config.json"
echo ""
if [[ $WITH_LOCAL -eq 0 ]]; then
    echo "Note: local LLM inference (torch/transformers) not installed."
    echo "      Re-run with --with-local to add it."
fi
