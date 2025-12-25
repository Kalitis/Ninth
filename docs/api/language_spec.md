### 2. Файл `DOCUMENTATION.md`
Техническая спецификация для тех, кто хочет писать код.

# Ninth Language Specification (v0.5.1)

## 1. Core Concept
Ninth operates on a global **Stack**. All operations consume arguments from the top of the stack and push results back.
The fundamental data type is **torch.Tensor**. Even scalars (like `1.0`) are 0-d tensors.

## 2. Instruction Set

### Stack Manipulation
| Opcode | Description | Example |
| :--- | :--- | :--- |
| `[DUP]` | Duplicate top item | `A -> A A` |
| `[DROP]` | Remove top item | `A -> ` |
| `[SWAP]` | Swap top two items | `A B -> B A` |
| `[PEEK]` | Print top item (debug) | `A -> A` |
| `[PRINT]` | Print and pop item | `A -> ` |

### Math & Tensors
| Opcode | Description | Example |
| :--- | :--- | :--- |
| `[ADD]`, `[SUB]` | Addition / Subtraction | `3 2 [ADD] -> 5` |
| `[MUL]`, `[DIV]` | Multiplication / Division | `4 2 [DIV] -> 2` |
| `[MATMUL]` | Matrix Multiplication | `A(2x4) B(4x1) [MATMUL] -> C(2x1)` |
| `[RELU]`, `[SIGMOID]` | Activation functions | `-1 [RELU] -> 0` |
| `[SUM]` | Sum all elements | `[1 2] [SUM] -> 3` |
| `[ROUND]` | Round to nearest integer | `0.6 [ROUND] -> 1.0` |

### Smart Generators (v0.5.1)
| Opcode | Description | Example |
| :--- | :--- | :--- |
| `[RANDN]` | Random Normal Tensor | `[2 4] [RANDN]` creates 2x4 matrix |
| `[ZEROS]` | Zero Tensor | `[10] [ZEROS]` creates vector of 10 zeros |

### Memory & Variables
| Opcode | Description | Example |
| :--- | :--- | :--- |
| `[STORE]` | Save value to memory | `10 "a" [STORE]` |
| `[LOAD]` | Load value from memory | `"a" [LOAD] -> 10` |

### Autograd (The Magic)
| Opcode | Description |
| :--- | :--- |
| `[VAR]` | Mark tensor as trainable (`requires_grad=True`). Used for weights. |
| `[BACKWARD]` | Compute gradients from the scalar at the top of the stack (Loss). |
| `[GRAD]` | Retrieve calculated gradient for a variable. |
| `[ZERO_GRAD]`| Manually zero out gradients (optional, usually handled by update logic). |

### Control Flow
| Opcode | Description |
| :--- | :--- |
| `[IF] ... [ELSE] ... [ENDIF]` | Conditional execution. Checks if top > 0. |
| `[REPEAT] ... [END]` | Loop N times. `10 [REPEAT] ... [END]` |
| `[DEF] "name" ... [RET]` | Define a function/macro. |
| `[CALL] "name"` | Call a defined function. |

## 3. Syntax Rules
1. **Tokens**: Tokens are separated by spaces.
2. **Vectors**: Vectors are written as `[N M]`. Internally converted to shape tensors.
3. **Strings**: `"name"` implies a string literal (used for variable names).
4. **Programs**: Must start with `[PROGRAM_START]` and end with `[PROGRAM_END]` for block execution.

## 4. Example: Custom Function
```forth
[DEF] "squared_diff"
    [SUB] [DUP] [MUL]
[RET]

10 5 [CALL] "squared_diff" 

// Result: 25
