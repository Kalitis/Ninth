```markdown
# Ninth Language Specification
**Version:** v0.6.0 (Codename: Chimera)
**Date:** 2023-12-25

## 1. Introduction
Ninth is a stack-based, concatenative language optimized for defining and training neural networks. It combines the data flow control of Forth with the object-oriented state management of modern DL frameworks.

## 2. Syntax & Literals

### 2.1 Data Tensors `( ... )`
Creates a data tensor on the stack.
* `(1 2 3)` -> Vector `[3]`
* `((1 0) (0 1))` -> Matrix `[2, 2]`
* Scalars `5`, `0.01` are automatically cast to 0-D tensors.

### 2.2 Shape Literals `[ ... ]`
Defines dimensions for tensor creation. Used with generating opcodes or initialization.
* `[256 512]` -> Pushes a shape definition to the stack (or creates a tensor depending on context).
* Can accept variables: `[in_dim out_dim]`.

### 2.3 Semantic Strings `" ... "`
* Currently acts as string literals for debug/printing.
* *Future:* Will resolve to embedding vectors.

### 2.4 Configuration Objects `{ ... }`
Used for creating instances of Modules with named parameters (kwargs).
* `{Linear [std->0.02]}`
* Syntax: `{ModuleName [key->value]}`

## 3. Memory Model

Ninth v0.6.0 employs a **Dual-Scope Memory Model**:

### 3.1 The Data Stack
* **LIFO (Last In, First Out).**
* Stores transient data (activations, shapes, temporary calculation results).
* Cleared upon scope exit unless returned.

### 3.2 Variable Scopes
Variables are accessed via tokens.

| Syntax | Type | Scope | Lifetime |
| :--- | :--- | :--- | :--- |
| `-> name` | **Local** | Current Function/Block | Dies after `[RET]` |
| `name` | **Access** | Local (Search 1st) -> State (Search 2nd) | N/A |
| `-> @name` | **State** | Current Module Instance | **Persistent** (lives as long as the object) |
| `@name` | **Access** | Current Module Instance | N/A |

### 3.3 Assignment Operators
* `-> x`: **Pop & Bind**. Takes value off stack, saves to `x`.
* `->> x`: **Peek & Bind**. Copies value from stack top, saves to `x`. (Stack remains unchanged).

## 4. Module System (The Chimera Arch)

Modules are the building blocks of Ninth. They encapsulate State (`@W`) and Logic (`[FORWARD]`).

### 4.1 Definition
```forth
"Name" [MODULE]
    [INIT] ... [RET]    // Constructor
    [FORWARD] ... [RET] // Execution logic
[END_MODULE]

```

### 4.2 Lifecycle

1. **Instantiation:** `args {Name} -> @obj`.
* Executes `[INIT]` block.
* Allocates `@` variables into `instance.state`.


2. **Execution:** `input @obj [CALL]`.
* Executes `[FORWARD]` block.
* Context switches: `@vars` refer to `@obj`'s state.



## 5. Opcodes & Primitives

### 5.1 Stack Manipulation

* `[DUP]`: Duplicate top item.
* `[DROP]`: Discard top item.
* `[SWAP]`: Swap top two items.
* `[OVER]`: Copy second item to top.
* `[PEEK]`: Print top item without popping.
* `[PRINT]`: Pop and print.

### 5.2 Math & Tensor Ops

* `[ADD]`, `[SUB]`, `[MUL]`, `[DIV]`: Element-wise operations (Broadcastable).
* `[MATMUL]`: Matrix multiplication.
* `[MSE_LOSS]`: Mean Squared Error.
* `[SQUEEZE]`, `[UNSQUEEZE]`, `[FLATTEN]`, `[RESHAPE]`: Shape manipulation.

### 5.3 Autograd & Optimization (Low-Level)

* `[BACKWARD]`: Computes gradients for the tensor on stack.
* `[NO_GRAD] ... [END_NO_GRAD]`: Block where operations are excluded from the computational graph.
* `[GRAD]`: Pushes the gradient of a weight tensor to the stack.
* `[SUB_ASSIGN_GRAD]`: `(weight lr grad -- )`. Performs `weight -= lr * grad` in-place.
* `[ZERO_GRAD]`: Recursively clears gradients for a module instance.

### 5.4 Introspection (Meta-Programming)

* `[PARAMS]`: Takes an Instance, returns a List of all trainable parameters (recursively).
* `[FOREACH]`: Iterates over a list.
* Syntax: `list { -> item ... } [FOREACH]`



## 6. Control Flow

* `[RET]`: Return from current block.
* `[IF] ... [ELSE] ... [ENDIF]` (*Reserved/Experimental*).

## 7. Safety Protocols

1. **LIFO Construction:** When initializing networks, arguments must be pushed in reverse order of the layer's expected arguments (e.g., `10 5 1` for `1->5->10`).
2. **State Isolation:** Local variables cannot be accessed outside their defining block. `@` variables are only accessible via their instance or within the instance's methods.

```