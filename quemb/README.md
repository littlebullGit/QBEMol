# Bundled QuEmb Subpackage

This directory contains the QuEmb-derived code required by QBEMol's molecular
VQE-QBE paper workflows. It is not a full upstream QuEmb distribution. The
retained public scope is molecular Bootstrap Embedding through `quemb.molbe`,
plus the QBEMol VQE fragment-solver additions.

## QBEMol Additions

- `quemb.molbe.vqe_solver`: VQE, ADAPT-VQE, FastAdaptVQE, and
  MatrixFreeAdaptVQE integration with molecular BE.
- `quemb.molbe.fast_adapt_vqe`: direct statevector gradient evaluation for
  ADAPT screening.
- `quemb.molbe.matrix_free_adapt_vqe`: matrix-free Pauli application for ADAPT
  screening.

## Retained Compatibility Code

A small `quemb.kbe` compatibility subset remains because some molecular modules
import `quemb.kbe.pfrag.Frags` for shared optimizer and fragment helper logic.
Periodic BE entry points are not part of this release.

## Upstream Attribution

The bundled code derives from QuEmb
(https://github.com/troyvvgroup/quemb), licensed under Apache-2.0. See
`../NOTICE`, `NOTICE`, and `LICENSE` for attribution and license details.
