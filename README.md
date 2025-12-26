# Ninth: The Native Language of Neuro-Symbolic AI

![Version](https://img.shields.io/badge/version-v0.6.2-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)

**Ninth** is a minimal, Turing-complete, differentiable stack-based programming language. It is designed to be the "Machine Code" of AI — a language where the program itself is a tensor, and the interpreter is a differentiable computational graph.

Unlike traditional languages where AI calls external tools (Python/JSON), Ninth allows Large Language Models to **think** in executable, differentiable code.

> **"The code is not just instructions; it is topology. The variables are not just values; they are weights."**

## 🌟 Key Features (v0.6.2 "Chimera")

* **Everything is a Tensor:** From scalars to strings, every element on the stack is a PyTorch tensor.
* **Stateful Modules:** A class-based system (`[MODULE]`, `[INIT]`, `[FORWARD]`) similar to `torch.nn.Module`, but fully inspectable and modifiable at runtime.
* **Self-Hosted Autograd:** Optimization algorithms (like SGD, Adam) are written in Ninth itself, not hardcoded in C++.
* **Dual Memory Model:**
    * **Stack:** For transient data flow.
    * **Scope:** Local variables (`->x`) and Persistent State (`@W`).

## 🚀 Quick Start

### Installation
Ninth requires only Python and PyTorch.

```bash
pip install torch

```

### The "DeepNet" Example

This program defines a neural network and an optimizer completely within the language:

```forth
// 1. Define a Linear Layer
"Linear" [MODULE]
    [INIT]
        -> out_dim -> in_dim
        [in_dim out_dim] -> @W  // Persistent Weight
        [1 out_dim]      -> @b  // Persistent Bias
    [RET]
    [FORWARD]
        -> x
        x @W [MATMUL] @b [ADD]
    [RET]
[END_MODULE]

// 2. Instantiate Network & Data
1 5 10 {Linear} -> @net         // Create layer: 10->5->1
(1 1 1 1 1 1 1 1 1 1) -> input  // Input Tensor

// 3. Forward Pass
input @net [CALL] ->> prediction
"Result:" [PRINT] prediction [PEEK]

```

## 📁 Project Structure

```
ninth/
├── src/
│   ├── core/           # VM and Tokenizer
│   ├── modules/        # Standard Library (SGD, MSE, etc.)
│   └── ops/            # Tensor Operations (Math, Shape)
├── examples/
│   ├── 01_basics.nth
│   ├── 02_linear_regression.nth
│   └── 03_deep_net_optimizer.nth
├── docs/
│   └── SPECIFICATION.md
└── main.py             # Entry point

```

## 📄 License

MIT License.
