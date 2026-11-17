# SensitivityMap

SensitivityMap is a small Python library for inspecting NumPy weight matrices. It computes per-output-row perturbation scores and basic singular-value statistics for matrices supplied by the caller. It is a prototype utility, not a model-training framework or a command-line application.

## Implemented functionality

- `row_sensitivity(weights, samples=64, eps=1e-2, seed=0)` generates standard-normal inputs, compares the matrix output with outputs after scaling each weight row by `1 + eps`, and returns one score per row. The score is the mean absolute output change divided by the maximum absolute value in the unperturbed output batch.
- `sensitivity_map(layers, samples=64, eps=1e-2, seed=0)` applies that calculation to a non-empty dictionary of layer names and 2-D weight matrices.
- `layer_spectrum(weights)` returns singular values in descending order. `effective_rank(weights, energy=0.99)` returns the smallest count whose cumulative singular-value sum reaches the requested fraction.
- `layer_report(weights)` reports matrix shape, Frobenius norm, spectral norm, condition number, and effective rank. `analyze_model(layers)` produces reports for a non-empty mapping, and `find_bottlenecks(reports, top_k=1)` ranks report entries by ascending effective rank.

Inputs are plain two-dimensional NumPy-compatible arrays; sensitivity matrices use shape `(output_features, input_features)`. The module does not load model checkpoints or datasets, and the sensitivity scores are not gradients, dataset-specific attributions, or measures of predictive accuracy.

## Requirements and installation

The package metadata in `pyproject.toml` declares Python 3.9 or newer and NumPy 1.24 or newer. To install from a local clone:

```bash
python -m pip install .
```

For development and tests, install the listed requirements as well:

```bash
python -m pip install -r requirements.txt
```

## Example

```python
import numpy as np
from sensitivitymap import analyze_model, sensitivity_map

layers = {
    "projection": np.array([[1.0, 0.0], [0.0, 0.5]]),
}

scores = sensitivity_map(layers, samples=64, eps=1e-2, seed=0)
print(scores["projection"])  # one score for each output row

reports = analyze_model(layers)
print(reports["projection"]["effective_rank"])
```

The library exposes functions; it does not provide a CLI. Matrix values and layer dictionaries must be prepared by the calling code.

## Tests and CI

The repository contains pytest coverage for sensitivity-score behavior, input validation, singular values, effective rank, reports, and bottleneck ordering. On the checked-out commit, the suite was run locally with Python 3.12.3 and pytest 9.2.0: **16 tests passed**. The GitHub Actions workflow defines test jobs for Python 3.10, 3.11, and 3.12, but those hosted workflow results are not reported here.

Run the tests locally with:

```bash
python -m pytest -q
```

## Scope and limitations

- Sensitivity scores depend on a seeded batch of random Gaussian inputs and on `samples` and `eps`; they do not measure importance for a particular dataset or task.
- Each output row is evaluated by recomputing the matrix output after perturbation. This is a straightforward analysis routine, not an optimized large-model or batched-layer implementation; no benchmark results are provided.
- Spectral analysis operates on supplied 2-D matrices only. There is no framework adapter, checkpoint/model loader, visualization, training integration, or command-line interface in the repository.
- `effective_rank` uses the cumulative sum of singular values (default threshold 0.99), not the sum of squared singular values.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for its terms.
