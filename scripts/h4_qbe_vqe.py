#!/usr/bin/env python
"""Run linear H4 at 1.0 Angstrom spacing with the generic QBE-VQE workflow."""

from __future__ import annotations

from pathlib import Path

if __package__:
    from .run_qbe_vqe import main as run_qbe_vqe
else:
    from run_qbe_vqe import main as run_qbe_vqe


XYZ = """4
Linear H4 at 1.0 Angstrom spacing
H 0.000000 0.000000 0.000000
H 1.000000 0.000000 0.000000
H 2.000000 0.000000 0.000000
H 3.000000 0.000000 0.000000
"""


def main() -> int:
    """Run the linear H4 BE2 calculation through the shared QBE-VQE runner."""

    output_dir = Path("results/h4_qbe_vqe")
    geometry_path = output_dir / "h4.xyz"
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text(XYZ, encoding="utf-8")
    return run_qbe_vqe(
        [
            str(geometry_path),
            "--basis",
            "sto-3g",
            "--be-order",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
