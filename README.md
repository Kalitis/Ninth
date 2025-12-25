# Ninth: The Native Language of Neuro-Symbolic AI

![Version](https://img.shields.io/badge/version-v0.6.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)

**Ninth** is a minimal, Turing-complete, differentiable stack-based programming language designed for **Integrated Function Calling** within LLMs.

Unlike traditional tool use (JSON/Python), Ninth allows Large Language Models to "think" in code. It combines the simplicity of Forth with the power of PyTorch autograd.

> **"The model doesn't see the code execution, it only sees the result. It's like a Ghost in the Machine."**

## 📁 Project Structure

```
ninth/
├── src/                 # Source code
│   ├── core/            # Core VM implementation
│   ├── operations/      # Stack and math operations
│   ├── memory/          # Memory management
│   ├── control/         # Control flow operations
│   ├── generators/      # Tensor generation operations
│   └── autograd/        # Autograd-related operations
├── tests/              # Test suite
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── e2e/            # End-to-end tests
├── docs/               # Documentation
│   ├── api/            # API reference and language spec
│   ├── examples/       # Example programs
│   └── tutorials/      # Tutorial guides
├── examples/           # Example programs
├── config/             # Configuration files
└── requirements.txt    # Python dependencies
```

## 🚀 Quick Start

### Installation

Ninth is extremely lightweight. You only need `torch`.

```bash
pip install torch numpy matplotlib
```

### Running

```bash
cd src/core
python vm.py
```

## 📖 Documentation

- [Main Documentation](docs/README.md) - Overview and getting started
- [Language Specification](docs/api/language_spec.md) - Technical details
- [Examples](docs/examples/) - Example programs and use cases
- [Tutorials](docs/tutorials/) - Step-by-step guides

## ⚡ Quick Example

```forth
[PROGRAM_START]
3 4 [ADD]     // Stack: 7
5 [MUL]       // Stack: 35
[PRINT]
[PROGRAM_END]
```

## 📄 License

MIT License. Free to use for research and revolution.
