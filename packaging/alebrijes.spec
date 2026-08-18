# PyInstaller spec for the packaged desktop app (Phase 4).
#
# --onedir, not --onefile: at ~550-700MB, --onefile would re-extract the
# whole payload into a temp dir on *every* launch (a multi-second delay
# each time); --onedir unpacks once at install time. See PHASES.md Phase 4.
#
# Run from the repo root:
#   pyinstaller packaging/alebrijes.spec
#
# Expects, before running:
# - CPU-only torch installed (see README) -- this spec doesn't control
#   which torch build gets bundled, only what's already importable in the
#   build venv.
# - The HF weights cache populated at packaging/build/hf_cache/ (see
#   PHASES.md's build-pipeline step 1 -- CI authenticates with a
#   project-owned token and downloads kyutai/pocket-tts there before this
#   spec runs). Bundled at runtime path hf_cache/, matching what
#   alebrijes/hf_bundle.py looks for.
# - spacy's en_core_web_sm model installed (`python -m spacy download
#   en_core_web_sm`) so its package data can be collected.
#
# Known-incomplete: this is a first pass, not yet verified on real macOS/
# Windows hardware (see PHASES.md). torch/spacy in particular are known to
# need hidden-import/collect-submodules iteration once actually run on the
# target platforms -- don't assume this "just works" without a real test.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

REPO_ROOT = Path.cwd()
HF_CACHE_DIR = REPO_ROOT / "packaging" / "build" / "hf_cache"

# collect_data_files, not collect_submodules: submodules PyInstaller finds by
# statically tracing actual imports (what Analysis() below does) already
# cover normal code paths. Blanket collect_submodules on a package this size
# pulls in things like spacy's own internal test suite -- confirmed bloating
# a local trial build to multiple GB for zero runtime benefit. Only reach for
# collect_submodules/hiddenimports on a *specific* module PyInstaller
# actually failed to find at runtime, not preemptively.
datas = [
    (str(REPO_ROOT / "src" / "alebrijes" / "frontend" / "index.html"), "alebrijes/frontend"),
]
datas += collect_data_files("imageio_ffmpeg")
datas += collect_data_files("en_core_web_sm")

if HF_CACHE_DIR.exists():
    datas.append((str(HF_CACHE_DIR), "hf_cache"))
else:
    print(
        f"WARNING: {HF_CACHE_DIR} not found -- building without bundled pocket-tts "
        "weights. The packaged app will need network + an HF login at runtime, "
        "defeating the point. See PHASES.md Phase 4 build-pipeline step 1."
    )

hiddenimports: list[str] = []

a = Analysis(
    [str(REPO_ROOT / "src" / "alebrijes" / "gui.py")],
    pathex=[str(REPO_ROOT / "src")],
    datas=datas,
    hiddenimports=hiddenimports,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Alebrijes",
    console=False,  # windowed app -- no terminal window
    icon=None,  # TODO: add a real app icon before shipping
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="Alebrijes",
)
