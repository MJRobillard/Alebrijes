"""Assembly stage: merge per-span/per-chapter audio into final output.

The actual merge/mux logic lives in `alebrijes.synth.parallel` (it's produced
as a side effect of `run_parallel_synthesis`); this module re-exports it under
the pipeline-stage name so callers reason about the pipeline in terms of its
six stages rather than needing to know synth and assemble currently share
one implementation file.
"""

from alebrijes.synth.parallel import build_m4b, merge_wav_files

__all__ = ["build_m4b", "merge_wav_files"]
