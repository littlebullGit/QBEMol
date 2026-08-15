#!/usr/bin/env python
"""Run F2 at 1.42 Angstrom with the generic QBE-VQE workflow."""

from __future__ import annotations

from pathlib import Path

if __package__:
    from .run_qbe_vqe import main as run_qbe_vqe
else:
    from run_qbe_vqe import main as run_qbe_vqe


XYZ = """2
F2 at 1.42 Angstrom
F 0.000000 0.000000 0.000000
F 0.000000 0.000000 1.420000
"""


def main() -> int:
    """Run the F2 BE1 calculation through the shared QBE-VQE runner."""

    output_dir = Path("results/f2_qbe_vqe")
    geometry_path = output_dir / "f2.xyz"
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text(XYZ, encoding="utf-8")
    return run_qbe_vqe(
        [
            str(geometry_path),
            "--basis",
            "sto-3g",
            "--be-order",
            "1",
            "--selector",
            "lookahead",
            "--output-dir",
            str(output_dir),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
