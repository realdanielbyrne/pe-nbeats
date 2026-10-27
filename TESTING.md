# TESTING.md

Testing conventions and procedures for this repository, intended for both human developers and AI coding agents working on `pe_nbeats`.

## Overview

This project uses **pytest** for testing. Tests live in the `tests/` directory.

The fastest feedback loop is usually:

1. run the smallest relevant test target first
2. fix the issue
3. re-run the same target
4. expand to a broader test scope only after the targeted check passes

## Prerequisites

### Python version

- **Python** >= 3.12, < 3.15

Use a supported interpreter before running the test suite.

### Activate your virtual environment

Activate the project virtual environment before running tests so `pytest`, `torch`, `lightning`, and the editable package are resolved from the correct interpreter.

```bash
source .venv/bin/activate
```

If you use a different environment manager, activate that environment instead.

Recommended setup for a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

## Running Tests

Common pytest commands:

```bash
pytest tests/                        # run all tests
pytest tests/ -v                     # run all tests with verbose output
pytest tests/test_blocks.py          # run a specific test file
pytest tests/test_blocks.py -k "TestGenericArchitecture"  # run a specific test class
pytest tests/test_blocks.py -k "test_block_output_shape"  # run a specific test method or substring match
pytest tests/ -x                     # stop on first failure
```

```bash
pytest tests/test_models.py -k "trend_thetas_dim" -x -v
```

Notes:

- `-k` performs substring matching, so it can target a class name, method name, or keyword fragment.
- Prefer the smallest relevant scope first: single method/class pattern -> single file -> full suite.

## Test File Organization

- `tests/test_blocks.py` — block shapes, attributes, registries, and block-specific regression coverage
- `tests/test_loaders.py` — `DataModule` setup, slicing, and train/validation/test split behavior
- `tests/test_models.py` — width selection, optimizer dispatch, forward pass behavior, and `sum_losses` semantics

There are also a few more specialized regression or study-oriented tests in `tests/`. When a change touches experiment scripts, convergence logic, or newer study scaffolding, review those files as well.

## Writing New Tests

### Naming conventions

- Test files must follow the `test_*.py` pattern so pytest discovers them automatically.
- Test classes should typically be named `Test...`.
- Test functions and methods should be named `test_...`.

### Add tests with feature changes

When you implement a new feature or fix a bug:

1. update an existing test if the behavior belongs to an already-covered area
2. add a new test when the behavior is new, previously untested, or regression-prone
3. run the most relevant existing test file before expanding to a broader suite

### Adding tests for new block types

If you add a new block type:

1. add the block class in `src/pe_nbeats/blocks/blocks.py`
2. add the block name to `src/pe_nbeats/constants.py`
3. add any needed routing logic in `src/pe_nbeats/models.py`
4. add or update tests in `tests/test_blocks.py`

`tests/test_blocks.py` contains the parametrized `TestAllBlocksOutputShapes` suite. If the new block is registered correctly and fits the existing constructor conventions, that parametrized test is often the first line of coverage for output-shape validation.

You should still add focused tests if the new block has unique behavior, parameters, basis logic, stochastic behavior, or special routing requirements.

### When to reuse existing tests vs. create new ones

Reuse existing tests when:
- you are changing a code path already covered by an existing test module
- the expected behavior is the same and only the implementation changed

Create a new test when:
- you are adding a new public parameter or mode
- you fixed a bug that could regress later
- the behavior is specific enough that extending an unrelated test would make the suite harder to understand

## CI/CD Note

Tests do **not** run automatically in CI before PyPI publishing.

The GitHub Actions workflow in `.github/workflows/python-publish.yml` publishes to PyPI on GitHub release, but it does not execute the test suite first.

That means local test execution is the primary safety check before publishing or opening a pull request.

## Common Issues

### Wrong Python interpreter

If imports fail unexpectedly, confirm the active interpreter is Python 3.12+ and that the virtual environment is activated.

### Package not installed in the environment

If `import pe_nbeats` fails, install the project into the active environment:

```bash
pip install -e .
```

### `-k` matches more than expected

`pytest -k` uses substring matching. If your filter is too broad, it may run multiple tests. Use a more specific class or method name when narrowing scope.

### Running too much, too early

When debugging, avoid starting with the full suite unless the change is broad. Run the smallest relevant target first for faster feedback.

### No CI safety net

Because tests are not enforced in CI prior to publish, always run the relevant local tests for your change. For code changes, targeted tests should be considered mandatory.

