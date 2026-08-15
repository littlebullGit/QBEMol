# macOS Setup Guide (GCC Toolchain)

## Prerequisites

- **Homebrew**: Ensure Homebrew is installed from https://brew.sh.
- **Python**: Versions `3.10` – `<3.13` supported.
- **Conda (optional but recommended)**: Allows environment isolation.

## Install System Packages

1. **Install GCC with OpenMP support**

```bash
brew update
brew install gcc
```

2. **Optional utilities**

```bash
brew install cmake ninja hdf5
```

## Configure Compiler Environment

Add the Homebrew GCC binaries to your shell environment. Update the version suffix (`-15`, etc.) to match the installed GCC release.

```bash
export HOMEBREW_GCC_PREFIX="$(brew --prefix gcc)"
export CC="$HOMEBREW_GCC_PREFIX/bin/gcc-15"
export CXX="$HOMEBREW_GCC_PREFIX/bin/g++-15"
export FC="$HOMEBREW_GCC_PREFIX/bin/gfortran-15"
export PATH="$HOMEBREW_GCC_PREFIX/bin:$PATH"
```

If you need the runtime libraries discovered at link/runtime:

```bash
export LDFLAGS="-L$HOMEBREW_GCC_PREFIX/lib"
export CPPFLAGS="-I$HOMEBREW_GCC_PREFIX/include"
```

## Python Environment

### Using Conda base environment

```bash
conda activate base
conda install python=3.11
pip install --upgrade pip
```

### Install project dependencies

```bash
pip install -r requirements.txt  # if present
pip install -r tests/test_requirements.txt
pip install .
```

For VQE features install the Qiskit stack:

```bash
pip install qiskit qiskit-nature qiskit-algorithms
```

## Build & Install

1. Clone the repository:

```bash
git clone --recurse-submodules git@github.com:littlebullGit/quemb.git
cd quemb
```

2. Ensure compiler environment variables (`CC`, `CXX`, `FC`) are set in the current shell.

3. Install the package locally:

```bash
pip install .
```

## Testing

Run targeted tests (after installing `pytest`):

```bash
python -m pytest tests/molbe_vqe_solver_test.py
```

Or run the full suite:

```bash
python -m pytest
```

If `pytest` is unavailable, install it with `pip install pytest` inside the active environment.

## Documentation (Optional)

```bash
cd docs
pip install -r requirements.txt
make html
```

## Notes

- The Homebrew GCC toolchain provides OpenMP support required by some dependencies.
- Keep `PATH` and compiler variables in your shell profile (`~/.zshrc`, `~/.bashrc`) to persist settings.
- Adjust version numbers if Homebrew upgrades GCC.
- For optional backends (e.g., ORCA), follow additional instructions in `docs/source/install.rst`.
