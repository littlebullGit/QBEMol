# Release Manifest

| Path | Purpose |
|---|---|
| `README.md` | Installation and execution instructions. |
| `data/results/` | Machine-readable H4, F2, H6, and n-butane results with provenance. |
| `scripts/run_qbe_vqe.py` | Molecule-independent BE-VQE runner. |
| `scripts/lookahead_utils.py` | Shared lookahead implementation. |
| `scripts/validate_adapt_implementations.py` | FastAdapt/MatrixFree paper validation. |
| `scripts/h4_qbe_vqe.py` | H4 production control. |
| `scripts/f2_qbe_vqe.py` | F2 production run. |
| `scripts/h6_qbe_vqe.py` | H6 CEO-manifest production scan. |
| `scripts/n-butane_qbe_vqe.py` | n-butane W12 production scan. |
| `data/h6_manifests/` | H6 CEO manifest registry and operator pools. |
| `data/n_butane_cid7843.sdf` | Pinned n-butane source conformer. |
| `quemb/src/quemb/` | Required BE and VQE source. |
| `quemb/external/eigen/` | Required C++ build dependency. |
| `quemb/tests/` | Focused regression tests. |
| `pyproject.toml`, `uv.lock`, `requirements-paper.txt` | Environment metadata. |
| `LICENSE`, `NOTICE`, `CITATION.cff` | License, attribution, and citation. |

Generated logs, new run outputs, Hamiltonians, caches, and virtual environments
are excluded. Normalized publication results under `data/results/` are included.
