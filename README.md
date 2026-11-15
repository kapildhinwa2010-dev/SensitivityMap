# SensitivityMap

[![CI](https://github.com/kapildhinwa2010-dev/SensitivityMap/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SensitivityMap/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/SensitivityMap/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SensitivityMap/actions/workflows/codeql.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Map per-row output sensitivity of weight matrices via perturbation analysis.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

row_sensitivity() perturbs each row of a weight matrix by a small
relative epsilon and measures the mean relative change of the output
over random inputs - a finite-difference map of which output rows
actually matter. sensitivity_map() runs this for every layer of a
{name: matrix} dict. The shared spectral base (effective rank,
condition number, bottleneck ranking) is included for context.

## Install

    pip install -r requirements.txt

## Quick start

    from sensitivitymap import sensitivity_map

    scores = sensitivity_map(layers, samples=64, eps=1e-2)
    print(scores["fc1"][:5])  # first five rows of fc1

## API

- row_sensitivity(weights, samples=64, eps=1e-2, seed=0) - (out,) array
- sensitivity_map(layers, samples, eps, seed) - {name: array}
- layer_spectrum, effective_rank, layer_report, analyze_model,
  find_bottlenecks (shared spectral base)

## Honest results and limitations

- This is a finite-difference estimate on random Gaussian inputs, not a
  gradient or an attribution method: scores depend on the sampled
  inputs and say nothing about a specific dataset.
- Cost is O(out_features) forward passes per layer - fine for toy
  scales, slow for large layers.
- A high-sensitivity row is not automatically "important" for accuracy;
  it just moves the output most under this perturbation.

## License

MIT - see [LICENSE](LICENSE).
