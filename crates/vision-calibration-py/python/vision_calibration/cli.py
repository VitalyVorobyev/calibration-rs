"""The ``vcal`` command line: the same program as the Rust ``vcal`` binary.

``pip install vision-calibration`` installs it as a console script::

    vcal init <folder>              draft <folder>/dataset.toml from the file names
    vcal check <manifest>           check a manifest and its images, without detecting
    vcal run <manifest> [-o <dir>]  detect, calibrate, write calibration.json and quality.json
    vcal sample list                list the sample datasets
    vcal sample fetch <id>          download a sample dataset; prints its manifest's path

``vcal <command> --help`` describes each command.
"""

from __future__ import annotations

import signal
import sys
from collections.abc import Sequence

from . import _vision_calibration as _native

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``vcal`` with ``argv`` (default: the process arguments) and exit
    with its status: 0 done, 1 an error, 2 a wrong command line, 3 a ``run``
    whose quality verdict is fail."""
    args = sys.argv[1:] if argv is None else list(argv)
    # The calibration runs in Rust without returning to Python, so Python's
    # Ctrl-C handler would only fire once it is done: let the signal end the
    # process instead.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    raise SystemExit(_native.vcal_main(["vcal", *args]))


if __name__ == "__main__":
    main()
