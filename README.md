# QBEMol

QBEMol couples molecular bootstrap embedding (BE) to VQE fragment solvers. Run
all commands below from the repository root.

## Install

Python 3.10-3.12 and [uv](https://docs.astral.sh/uv/) are required. 

Linux:

```bash
uv sync --locked
```

macOS:

```bash
brew install libomp
LIBOMP="$(brew --prefix libomp)"
export CMAKE_ARGS="-DOpenMP_CXX_FLAGS=-Xpreprocessor;-fopenmp;-I${LIBOMP}/include \
  -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=${LIBOMP}/lib/libomp.dylib"
uv sync --locked
```

If Apple Clang fails to build QuEmb, use Homebrew GCC once:

```bash
brew install gcc
./uv_run_with_gcc.sh python -c "import quemb"
```

## Run an XYZ Molecule

Save this example as `h4.xyz`:

```text
4
linear H4
H 0.000000 0.000000 0.000000
H 1.000000 0.000000 0.000000
H 2.000000 0.000000 0.000000
H 3.000000 0.000000 0.000000
```

Greedy BE2-FastAdaptVQE:

```bash
uv run --no-sync python scripts/run_qbe_vqe.py h4.xyz \
  --basis sto-3g --be-order 2 --ansatz adapt_fast \
  --output-dir results/h4
```

Lookahead:

```bash
uv run --no-sync python scripts/run_qbe_vqe.py h4.xyz \
  --basis sto-3g --be-order 2 --ansatz adapt_fast \
  --selector lookahead --lookahead-top-k 5 \
  --output-dir results/h4-lookahead
```

Both commands write `result.json` in the output directory. Run
`uv run --no-sync python scripts/run_qbe_vqe.py --help` for all options.

## Run the Paper Calculations

Install the paper package versions first:

```bash
uv pip install -r requirements-paper.txt
mkdir -p logs
```

Run long commands one at a time. Each command writes a log and its results under
`results/`.

H4/F2 implementation validation:

```bash
uv run --no-sync python -u scripts/validate_adapt_implementations.py \
  2>&1 | tee logs/adapt_validation.log
```

H4 QBE-VQE:

```bash
uv run --no-sync python -u scripts/h4_qbe_vqe.py \
  2>&1 | tee logs/h4_qbe_vqe.log
```

F2 QBE-VQE:

```bash
uv run --no-sync python -u scripts/f2_qbe_vqe.py \
  2>&1 | tee logs/f2_qbe_vqe.log
```

H6 QBE-VQE:

```bash
uv run --no-sync python -u scripts/h6_qbe_vqe.py \
  2>&1 | tee logs/h6_qbe_vqe.log
```

n-butane QBE-VQE:

```bash
uv run --no-sync python -u scripts/n-butane_qbe_vqe.py \
  2>&1 | tee logs/n_butane_qbe_vqe.log
```

## Data

H4, F2, H6, and n-butane results are in `data/results/`. H6 and n-butane were run on Brown University's Oscar cluster; H4 and F2 were run on a local workstation.

## Citation

Use `CITATION.cff`. QBEMol is licensed under Apache-2.0, with QuEmb attribution
in `quemb/NOTICE`.
