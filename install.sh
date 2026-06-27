set -e
ORIGINAL_DIR=$(pwd)

cleanup() {
    deactivate || true
    cd "$ORIGINAL_DIR" || true
}
trap cleanup EXIT

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment for stt-tts..."
    python3.12 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo Installation for stt-tts complete


if [ ! -d ".venv_tools" ]; then
    echo "Creating virtual environment for stt-tts preparation tools..."
    python3.12 -m venv .venv_tools
fi

source .venv_tools/bin/activate
pip install --upgrade pip
pip install -r requirements-tools.txt

python install.py

echo Installation for stt-tts preparation tools complete