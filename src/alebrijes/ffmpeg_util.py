"""Locates ffmpeg, bundled via the `imageio-ffmpeg` pip package.

Ships a real static ffmpeg binary (with AAC encoding, confirmed) as
installable package data — wheels exist for Windows, macOS (Intel + Apple
Silicon), and Linux — instead of requiring the end user to install ffmpeg
themselves, or requiring a separate Phase 4 CI step to fetch, checksum, and
`--add-binary` a static build by hand. `render.py`'s pre-flight check and
`synth/parallel.py`'s `build_m4b` both use this, so they can never disagree
about whether ffmpeg is available.
"""

from __future__ import annotations


def find_ffmpeg() -> str:
    """Returns the ffmpeg binary's path.

    Raises RuntimeError with a clear, actionable message if it can't be
    found — should only happen with a broken/incomplete install, since
    imageio-ffmpeg ships the binary as package data for every platform this
    project supports.
    """
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is not installed (it provides ffmpeg). Reinstall with: pip install -e \".[dev]\""
        ) from exc

    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as exc:
        raise RuntimeError(
            f"ffmpeg could not be located via imageio-ffmpeg: {exc} "
            "Try reinstalling it: pip install --force-reinstall imageio-ffmpeg"
        ) from exc
