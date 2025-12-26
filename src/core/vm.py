import torch
import re
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, Regex

# === Ninth VM v0.6.1: The Chimera (HF Tokenizers Edition) ===

class ModuleDef:
    def __init__(self, name):
        self.name = name
        self.init_code = []
        self.forward_code = []

class ModuleInstance:
    def __init__(self, definition):
        self.definition = definition
        self.state = {} # Persistent memory (@vars)
    
    def __repr__(self):
        return f"<Instance: {self.definition.name}>"

class NinthVM:
    def __init__(self):
        self.stack = []
        self.scopes = [{}]      # Local scopes (Data Plane)
        self.modules = {}       # Class definitions
        self.context = None     # Current 'self' (Instance) for @vars
        
        # --- Tokenizer Setup ---
        self.tokenizer = self._build_tokenizer()

        # System Ops
        self.ops = {
            # Stack & Flow
            "[DUP]": self._op_dup, "[DROP]": self._op_drop, "[SWAP]": self._op_swap,
            "[PEEK]": self._op_peek, "[PRINT]": self._op_print,
            "[OVER]": self._op_over,
            
            # Math
            "[ADD]": lambda: self._binary_op(torch.add),
            "[SUB]": lambda: self._binary_op(torch.sub),
            "[MUL]": lambda: self._binary_op(torch.mul),
            "[DIV]": lambda: self._binary_op(torch.div),
            "[MATMUL]": lambda: self._binary_op(torch.matmul),
            "[MSE_LOSS]": self._op_mse_loss,
            
            # Autograd Primitives (Level 1)
            "[BACKWARD]": self._op_backward,
            "[ZERO_GRAD]": self._op_zero_grad,
            "[GRAD]": self._op_grad,         # Get gradient tensor
            "[SUB_ASSIGN_GRAD]": self._op_sub_assign_grad, # In-place update: W -= lr * grad
            
            # Introspection (Level 2)
            "[PARAMS]": self._op_params,     # Get list of trainable params from instance
            "[FOREACH]": self._op_foreach,   # Iterate list
            
            # Objects
            "[CALL]": self._op_call,         # Run [FORWARD]

            # --- Shape Manipulation ---
            "[SQUEEZE]": self._op_squeeze,   # [1, 10, 1] -> [10]
            "[UNSQUEEZE]": self._op_unsqueeze, # [10] -> [1, 10] (нужен индекс)
            "[FLATTEN]": self._op_flatten,   # [2, 2] -> [4]
            "[RESHAPE]": self._op_reshape,   # Изменение формы по списку
        }

    # --- Tokenizer Construction ---
    def _build_tokenizer(self):
        # Мы используем WordLevel модель, так как VM оперирует цельными токенами.
        # Для VM важно не разбивать [MATMUL] на ["[", "MAT", "MUL", "]"].
        tokenizer = Tokenizer(models.WordLevel(vocab={}))
        
        # Regex паттерн, полностью повторяющий логику вашего re.findall.
        # Обратите внимание: Rust Regex (используемый в tokenizers) не поддерживает некоторые фичи Python re,
        # но данный паттерн стандартен.
        # 1. ->> и ->
        # 2. @переменные
        # 3. "строки"
        # 4. [блоки] (шейпы, опкоды)
        # 5. {блоки} (лямбды, инстанцирование)
        # 6. (блоки) (данные)
        # 7. Любые другие непробельные последовательности (числа, имена переменных)
        pattern = r'\->>|\->|@[a-zA-Z0-9_]+|"[^"]*"|\[[^\]]*\]|\{[^\}]*\}|\([^\)]*\)|[^\s\[\]\{\}\(\)]+'
        
        # Используем Split с behavior='isolated'. 
        # Это заставляет токенайзер выделять совпадения в отдельные токены, отделяя их от всего остального.
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(pattern), behavior='isolated'),
        ])
        
        return tokenizer

    def _tokenize(self, text):
        # 1. Удаляем комментарии (старый добрый Python re здесь быстрее и надежнее)
        text = re.sub(r"//.*", "", text)
        
        # 2. Используем пре-токенизацию HF Tokenizers.
        # Метод pre_tokenize_str возвращает список кортежей (токен, (offset_start, offset_end)).
        # Нам нужны только сами строки.
        splits = self.tokenizer.pre_tokenizer.pre_tokenize_str(text)
        
        # 3. Фильтрация.
        # Tokenizer 'isolated' split может оставить пробелы между токенами как отдельные токены.
        # Нам нужно выбросить пустые строки или строки состоящие только из пробелов.
        tokens = [s for s, _ in splits if s.strip()]
        
        return tokens

    # --- Execution Core ---
    def execute(self, code, local_scope=None):
        # Если код - строка, токенизируем через HF Tokenizers
        tokens = self._tokenize(code) if isinstance(code, str) else code
        if local_scope is not None: self.scopes.append(local_scope)

        pc = 0
        while pc < len(tokens):
            token = tokens[pc]
            
            # 1. Module Definition Phase
            if token == "[MODULE]":
                pc = self._parse_module(tokens, pc)
                continue

            # 2. Control Flow Contexts
            if token == "[NO_GRAD]":
                end_idx = self._find_block_end(tokens, pc, "[NO_GRAD]", "[END_NO_GRAD]")
                block = tokens[pc+1:end_idx]
                with torch.no_grad():
                    self.execute(block)
                pc = end_idx + 1
                continue

            # 3. Rebinding (->) and Peek-Bind (->>)
            if token == "->" or token == "->>":
                is_peek = (token == "->>")
                pc += 1 
                if pc >= len(tokens): raise Exception(f"Syntax Error: {token} expected variable name")
                var_name = tokens[pc]
                
                val = self.stack[-1] if is_peek else self.stack.pop()
                
                if var_name.startswith("@"):
                    if self.context:
                        self.context.state[var_name] = val
                    else:
                        self.scopes[0][var_name] = val
                else:
                    self.scopes[-1][var_name] = val
                
                pc += 1
                continue

            # 4. Variable Access (Push to Stack)
            elif token.startswith("@"):
                if self.context and token in self.context.state:
                    self.stack.append(self.context.state[token])
                elif token in self.scopes[0]:
                    self.stack.append(self.scopes[0][token])
                else:
                    raise Exception(f"Undefined State Variable: {token}")
                
            # 5. Instantiation: {ClassName [k->v]}
            elif token.startswith("{"):
                content = token.strip("{}").strip()
                parts = content.split() # Внутри блока используем обычный split, так как структура простая
                
                if parts and parts[0] in self.modules:
                    self._instantiate(token)
                else:
                    self.stack.append(token)

            # 6. Opcodes & Literals
            elif token in self.ops: 
                self.ops[token]()
            
            elif token.startswith('['):
                inner = token.strip("[]")
                # Простая эвристика: если внутри только капс и нет цифр - это неизвестный опкод
                if inner.isupper() and not any(c.isdigit() for c in inner):
                     print(f"Warning: Unknown Opcode encountered: {token}")
                else:
                    self._op_shape_literal(token)
                    
            elif token.startswith('('): self._op_data_literal(token)
            elif token.startswith('"'): self.stack.append(token.strip('"'))
            elif self._is_number(token): self.stack.append(torch.tensor(float(token)))
            elif self._resolve_local(token) is not None:
                self.stack.append(self._resolve_local(token))
            else:
                pass 

            pc += 1

        if local_scope is not None: self.scopes.pop()

    # --- Module System ---
    def _parse_module(self, tokens, start_pc):
        name = self.stack.pop()
        mod_def = ModuleDef(name)
        
        curr = start_pc + 1
        mode = None 
        
        while curr < len(tokens):
            t = tokens[curr]
            if t == "[END_MODULE]":
                curr += 1 
                break
            elif t == "[INIT]": mode = 'init'
            elif t == "[FORWARD]": mode = 'forward'
            elif t == "[RET]": mode = None
            else:
                if mode == 'init': mod_def.init_code.append(t)
                elif mode == 'forward': mod_def.forward_code.append(t)
            curr += 1
            
        self.modules[name] = mod_def
        print(f"System: Module '{name}' compiled (Tokenizer: HF).")
        return curr 

    def _instantiate(self, token):
        content = token.strip("{}")
        parts = content.split()
        class_name = parts[0]
        
        kwargs = {}
        for p in parts[1:]:
            if "->" in p:
                k, v = p.split("->")
                if self._is_number(v): v = torch.tensor(float(v))
                kwargs[k] = v
        
        if class_name not in self.modules: raise Exception(f"Unknown Module: {class_name}")
        instance = ModuleInstance(self.modules[class_name])
        
        old_ctx = self.context
        self.context = instance
        self.execute(instance.definition.init_code, local_scope=kwargs)
        self.context = old_ctx
        self.stack.append(instance)

    def _op_call(self):
        instance = self.stack.pop()
        if not isinstance(instance, ModuleInstance): raise Exception("CALL expects Instance")
        
        old_ctx = self.context
        self.context = instance
        self.execute(instance.definition.forward_code, local_scope={})
        self.context = old_ctx

    # --- Autograd & Optimizer Primitives ---
    def _collect_params(self, instance):
        params = []
        for k, v in instance.state.items():
            if isinstance(v, torch.Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, ModuleInstance):
                params.extend(self._collect_params(v))
        return params

    def _op_params(self):
        instance = self.stack.pop()
        params = self._collect_params(instance)
        self.stack.append(params)
        
    def _op_foreach(self):
        code_block = self.stack.pop() 
        target_list = self.stack.pop()
        clean_code = code_block.strip("{}").strip()
        for item in target_list:
            self.stack.append(item)
            self.execute(clean_code)

    def _op_grad(self):
        t = self.stack.pop()
        self.stack.append(t.grad if t.grad is not None else torch.zeros_like(t))

    def _op_sub_assign_grad(self):
        grad = self.stack.pop()
        lr = self.stack.pop()
        weight = self.stack.pop()
        with torch.no_grad():
            weight.sub_(grad * lr)

    def _op_backward(self): self.stack.pop().backward()

    def _op_zero_grad(self):
        target = self.stack.pop()
        if isinstance(target, ModuleInstance):
            for key, val in target.state.items():
                if isinstance(val, torch.Tensor) and val.grad is not None:
                    val.grad.zero_()
        elif isinstance(target, list):
            for val in target:
                if isinstance(val, torch.Tensor) and val.grad is not None:
                    val.grad.zero_()
        elif isinstance(target, torch.Tensor) and target.grad is not None:
            target.grad.zero_()

    # --- Standard Ops ---
    def _op_mse_loss(self):
        target = self.stack.pop()
        pred = self.stack.pop()
        loss = torch.nn.functional.mse_loss(pred, target)
        self.stack.append(loss)
        
    def _op_shape_literal(self, token):
        content = token.strip("[]")
        parts = content.split()
        dims = []

        for p in parts:
            if p.isdigit():
                dims.append(int(p))
            else:
                val = self._resolve_local(p)
                if val is None and self.context and p in self.context.state:
                    val = self.context.state[p]
                
                if val is None:
                    raise ValueError(f"Shape Literal Error: Dimension variable '{p}' is undefined.")

                if isinstance(val, torch.Tensor):
                    dims.append(int(val.item()))
                else:
                    dims.append(int(val))
        self.stack.append(torch.randn(*dims, requires_grad=True))
    
    def _op_data_literal(self, token):
        s = token.replace("(", " ( ").replace(")", " ) ")
        parts = s.split()
        stack = [[]]
        for p in parts:
            if p == "(":
                new_list = []
                stack.append(new_list)
            elif p == ")":
                if len(stack) > 1:
                    completed = stack.pop()
                    stack[-1].append(completed)
            else:
                try:
                    val = float(p)
                    stack[-1].append(val)
                except ValueError:
                    pass
        if len(stack[0]) > 0:
            data = stack[0][0]
            self.stack.append(torch.tensor(data, dtype=torch.float32))
        else:
             self.stack.append(torch.tensor([], dtype=torch.float32))

    def _resolve_local(self, name):
        if name in self.scopes[-1]: return self.scopes[-1][name]
        return None
    
    def _find_block_end(self, tokens, start, start_tok, end_tok):
        nest = 1
        i = start + 1
        while i < len(tokens) and nest > 0:
            if tokens[i] == start_tok: nest += 1
            elif tokens[i] == end_tok: nest -= 1
            i += 1
        return i - 1

    # Helpers
    def _is_number(self, s):
        try: float(s); return True
        except: return False
    def _binary_op(self, func):
        b, a = self.stack.pop(), self.stack.pop()
        self.stack.append(func(a, b))
    def _op_dup(self): self.stack.append(self.stack[-1])
    def _op_drop(self): self.stack.pop()
    def _op_swap(self): a, b = self.stack.pop(), self.stack.pop(); self.stack.extend([b, a])
    def _op_peek(self): print(f"PEEK ({type(self.stack[-1])}):\n{self.stack[-1]}")
    def _op_over(self): self.stack.append(self.stack[-2])
    def _op_print(self): print(f"OUT: {self.stack.pop()}")
    def _op_squeeze(self): self.stack.append(self.stack.pop().squeeze())
    def _op_unsqueeze(self):
        dim = int(self.stack.pop().item())
        self.stack.append(self.stack.pop().unsqueeze(dim))
    def _op_flatten(self): self.stack.append(self.stack.pop().flatten())
    def _op_reshape(self):
        shape_tensor = self.stack.pop()
        data_tensor = self.stack.pop()
        shape = shape_tensor.long().tolist() if shape_tensor.dim() > 0 else [int(shape_tensor.item())]
        self.stack.append(data_tensor.reshape(*shape))

# === RUNTIME DEMO ===
if __name__ == "__main__":
    vm = NinthVM()
    print("=== Ninth v0.7.0 Chimera: HF Tokenizers Edition ===\n")

    main_script = """
    // === 1. Базовый Линейный Слой ===
    "Linear" [MODULE]
        [INIT]
            -> out_dim -> in_dim
            [in_dim out_dim] -> @W
            [1 out_dim] -> @b 
        [RET]

        [FORWARD]
            -> x
            x @W [MATMUL] @b [ADD]
        [RET]
    [END_MODULE]

    // === 2. Композитный Модуль (DeepNet) ===
    "DeepNet" [MODULE]
        [INIT]
            -> out -> hidden -> in
            in hidden {Linear} -> @layer1
            hidden out {Linear} -> @layer2
        [RET]

        [FORWARD]
            -> x
            x @layer1 [CALL] ->> h_raw
            h_raw @layer2 [CALL]
        [RET]
    [END_MODULE]

    // === 3. Оптимизатор SGD ===
    "SGD" [MODULE]
        [INIT]
            -> lr -> target_model
            target_model -> @model
            lr -> @lr
        [RET]

        [FORWARD]
            [NO_GRAD]
                @model [PARAMS] 
                { 
                    -> p
                    p [GRAD] -> g
                    p @lr g [SUB_ASSIGN_GRAD]
                } [FOREACH]
            [END_NO_GRAD]
        [RET]
    [END_MODULE]

    // === 4. Скрипт Обучения ===
    "--- Start Training ---" [PRINT]

    // Создаем "Глубокую" сеть: 10 входов -> 5 скрытых -> 1 выход
    10 5 1 {DeepNet} -> @net

    // Создаем оптимизатор, скорость 0.05
    @net 0.005 {SGD} -> @opt

    // Данные (Batch size 1, Features 10)
    (1.0 0.5 0.5 1.0 0.0 0.0 1.0 0.5 0.5 1.0) -> input
    (0.0) -> target

    // --- ШАГ 1 ---
    "Step 1:" [PRINT]
    input @net [CALL] ->> pred1
    pred1 [PEEK]

    pred1[SQUEEZE] target[SQUEEZE] [MSE_LOSS] ->> loss1
    "Loss 1:" [PRINT] loss1 [PEEK]

    loss1 [BACKWARD]
    @opt [CALL]
    @net [ZERO_GRAD]

    // --- ШАГ 2 ---
    "Step 2:" [PRINT]
    input @net [CALL] ->> pred2
    
    pred2[SQUEEZE] target[SQUEEZE] [MSE_LOSS] ->> loss2
    "Loss 2:" [PRINT] loss2 [PEEK]

    loss2 [BACKWARD]
    @opt [CALL]
    @net [ZERO_GRAD]
    
    "--- Done ---" [PRINT]
    """
    
    vm.execute(main_script)