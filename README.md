# SensitivityMap

Layer-wise spectral analysis for deep networks.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

![CI](https://github.com/kapildhinwa2010-dev/SensitivityMap/actions/workflows/ci.yml/badge.svg)
![CodeQL](https://github.com/kapildhinwa2010-dev/SensitivityMap/actions/workflows/codeql.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- Singular-value spectra and effective rank per layer
- Condition numbers and bottleneck ranking
- `analyze_model()` reports every layer at once

## 📦 Install

```bash
pip install -r requirements.txt
```

## 🚀 Quick start

```python
from sensitivitymap import analyze_model, find_bottlenecks

layers = {'a': np.eye(16), 'b': low_rank_matrix}
reports = analyze_model(layers)
print(find_bottlenecks(reports, top_k=1))
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 📄 License

MIT — see [LICENSE](LICENSE).
