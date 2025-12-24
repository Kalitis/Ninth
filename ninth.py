import torch
import re
import sys

# Ninth VM v0.5.1 - Stable ND Edition
class NinthVM:
    def __init__(self):
        self.stack = []
        self.memory = {}
        self.functions = {}
        self.trace = False

        self.ops = {
            # Stack
            "[DUP]": self._op_dup, "[DROP]": self._op_drop, "[SWAP]": self._op_swap,
            "[PEEK]": self._op_peek, "[PRINT]": self._op_print,

            # Math
            "[ADD]": lambda: self._binary_op(lambda a, b: a + b),
            "[SUB]": lambda: self._binary_op(lambda a, b: a - b),
            "[MUL]": lambda: self._binary_op(lambda a, b: a * b),
            "[DIV]": lambda: self._binary_op(lambda a, b: a / b),
            "[MATMUL]": lambda: self._binary_op(torch.matmul),
            "[POW]": lambda: self._binary_op(torch.pow),
            "[SQRT]": lambda: self._unary_op(torch.sqrt),
            "[SUM]": lambda: self._unary_op(torch.sum),
            "[RELU]": lambda: self._unary_op(torch.relu),
            "[SIGMOID]": lambda: self._unary_op(torch.sigmoid),
            "[ROUND]": lambda: self._unary_op(torch.round),

            # Logic
            "[EQ]": lambda: self._binary_op(lambda a, b: (a == b).float()),
            "[GT]": lambda: self._binary_op(lambda a, b: (a > b).float()),
            "[LT]": lambda: self._binary_op(lambda a, b: (a < b).float()),

            # Memory
            "[STORE]": self._op_store, "[LOAD]": self._op_load,
            
            # Generators (Smart)
            "[RANDN]": self._op_randn_smart, 
            "[ZEROS]": self._op_zeros_smart,

            # Autograd
            "[VAR]": self._op_var,
            "[BACKWARD]": self._op_backward,
            "[GRAD]": self._op_grad,
            "[ZERO_GRAD]": self._op_zero_grad,
        }

    def _tokenize(self, text):
        text = re.sub(r"//.*", "", text) # Remove comments
        text = text.replace("[PROGRAM_START]", "").replace("[PROGRAM_END]", "")
        # Handle [2 4] vector syntax
        def repl(match): return match.group(0).replace(" ", ",")
        text = re.sub(r"\[[\d\s]+\]", repl, text)
        return [t for t in text.split() if t.strip()]

    def execute(self, code_input):
        tokens = self._tokenize(code_input) if isinstance(code_input, str) else code_input
        if not tokens: return None
        pc = 0
        while pc < len(tokens):
            token = tokens[pc]
            
            # Control Flow
            if token == "[DEF]":
                pc += 1; name = tokens[pc]; body = []; pc += 1; nesting = 0
                while pc < len(tokens):
                    if tokens[pc] == "[DEF]": nesting += 1
                    if tokens[pc] == "[RET]":
                        if nesting == 0: break
                        nesting -= 1
                    body.append(tokens[pc]); pc += 1
                self.functions[name] = body
            elif token == "[CALL]":
                pc += 1; name = tokens[pc]
                if name in self.functions: self.execute(self.functions[name])
                else: print(f"Error: Function '{name}' not defined")
            elif token == "[IF]":
                cond = self.stack.pop()
                if not (cond.item() > 0 if isinstance(cond, torch.Tensor) else cond):
                    pc = self._skip_block(tokens, pc, target=["[ELSE]", "[ENDIF]"])
            elif token == "[ELSE]": pc = self._skip_block(tokens, pc, target=["[ENDIF]"])
            elif token in ["[ENDIF]", "[RET]", "[PROGRAM_START]", "[PROGRAM_END]"]: pass
            elif token == "[REPEAT]":
                count = int(self.stack.pop().item()); start = pc + 1
                end = self._skip_block(tokens, pc, target=["[END]"])
                body = tokens[start:end]
                for _ in range(count): self.execute(body)
                pc = end
            
            # Vector Literals
            elif re.match(r"\[\d+(,\d+)*\]", token):
                nums = [int(n) for n in token.strip("[]").split(",")]
                self.stack.append(torch.tensor(nums))

            # Ops & Literals
            elif token in self.ops: self.ops[token]()
            elif self._is_number(token): self.stack.append(torch.tensor(float(token.replace(',', '.'))))
            elif token.startswith('"'): self.stack.append(token.strip('"'))
            pc += 1
        return self.stack[-1] if self.stack else None

    # --- Smart Generators ---
    def _op_randn_smart(self):
        shape_info = self.stack.pop()
        if shape_info.dim() > 0: self.stack.append(torch.randn(*shape_info.long().tolist()))
        else:
            dim2 = int(shape_info.item()); dim1 = int(self.stack.pop().item())
            self.stack.append(torch.randn(dim1, dim2))

    def _op_zeros_smart(self):
        shape_info = self.stack.pop()
        if shape_info.dim() > 0: self.stack.append(torch.zeros(*shape_info.long().tolist()))
        else:
            dim2 = int(shape_info.item()); dim1 = int(self.stack.pop().item())
            self.stack.append(torch.zeros(dim1, dim2))

    # --- Helpers ---
    def _skip_block(self, tokens, current_pc, target):
        nesting, pc = 0, current_pc + 1
        while pc < len(tokens):
            t = tokens[pc]
            if t in ["[IF]", "[REPEAT]"]: nesting += 1
            if t in ["[ENDIF]", "[END]"]:
                if nesting == 0 and t in target: return pc
                nesting -= 1
            if t == "[ELSE]" and nesting == 0 and "[ELSE]" in target: return pc
            pc += 1
        return pc

    def _is_number(self, s):
        try: float(s.replace(',', '.')); return True
        except: return False

    def _binary_op(self, func):
        if len(self.stack) >= 2: b, a = self.stack.pop(), self.stack.pop(); self.stack.append(func(a, b))
    def _unary_op(self, func):
        if len(self.stack) >= 1: self.stack.append(func(self.stack.pop()))
    def _op_dup(self): self.stack.append(self.stack[-1])
    def _op_drop(self): self.stack.pop()
    def _op_swap(self): a, b = self.stack.pop(), self.stack.pop(); self.stack.extend([a, b])
    def _op_peek(self): print(f"PEEK: {self.stack[-1]}")
    def _op_print(self): print(f"OUT: {self.stack.pop()}")
    def _op_store(self): name, val = self.stack.pop(), self.stack.pop(); self.memory[name] = val
    def _op_load(self): self.stack.append(self.memory[self.stack.pop()])
    def _op_var(self): 
        v = self.stack.pop().detach().clone().requires_grad_(True); v.retain_grad(); self.stack.append(v)
    def _op_backward(self):
        loss = self.stack.pop()
        if loss.numel() > 1: loss = loss.sum()
        loss.backward(retain_graph=True)
    def _op_grad(self):
        n = self.stack.pop(); g = self.memory[n].grad
        self.stack.append(g if g is not None else torch.tensor(0.0))
    def _op_zero_grad(self):
        n = self.stack.pop(); 
        if self.memory[n].grad is not None: self.memory[n].grad.zero_()

if __name__ == "__main__":
    vm = NinthVM()
    print("Ninth v0.5.1 Initialized.")
    # Example test
    vm.execute('[2 4] [RANDN] [PEEK]')