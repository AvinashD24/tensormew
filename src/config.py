import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROM_PATH = ROOT / "rom" / "pokemon_crystal.gbc"
SYM_PATH = ROOT / "rom" / "pokemon_crystal.sym"
RAM_PATH = Path(str(ROM_PATH) + ".ram")
ROM_SHA1 = "f2f52230b536214ef7c9924f483392993e226cfb"
STATES_DIR = ROOT / "states"
MANIFEST_PATH = STATES_DIR / "manifest.json"
RUNS_DIR = ROOT / "runs"
CONFIG_DIR = ROOT / "config"


def sha1_file(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()
