import torch
import re
import sys

# === Ninth VM v3.0: The Chimera ===
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

    # --- Tokenizer ---
    def _tokenize(self, text):
        text = re.sub(r"//.*", "", text)
        # Regex catches: ->>, ->, @vars, strings, brackets
        pattern = r'\->>|\->|@[a-zA-Z0-9_]+|"[^"]*"|\[[^\]]*\]|\{[^\}]*\}|\([^\)]*\)|[^\s\[\]\{\}\(\)]+'
        return [t for t in re.findall(pattern, text) if t.strip()]

    # --- Execution Core ---
    def execute(self, code, local_scope=None):
        tokens = self._tokenize(code) if isinstance(code, str) else code
        if local_scope is not None: self.scopes.append(local_scope)

        pc = 0
        while pc < len(tokens):
            token = tokens[pc]
            
            # 1. Module Definition Phase
            if token == "[MODULE]":
                # Special parser for module structure
                pc = self._parse_module(tokens, pc)
                continue

            # 2. Control Flow Contexts
            if token == "[NO_GRAD]":
                # Extract block until [END_NO_GRAD]
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
                        # Мы внутри инстанса (например, в INIT или FORWARD)
                        self.context.state[var_name] = val
                    else:
                        # Мы в Глобальном скоупе. 
                        # Сохраняем @var в самый первый скоуп (Global State)
                        self.scopes[0][var_name] = val
                else:
                    # Обычная локальная переменная
                    self.scopes[-1][var_name] = val
                
                pc += 1
                continue

            # 4. Variable Access (Push to Stack)
            elif token.startswith("@"):
                # 1. Сначала ищем в текущем контексте (если мы внутри модуля)
                if self.context and token in self.context.state:
                    self.stack.append(self.context.state[token])
                # 2. Если не нашли или мы в глобале, ищем в глобальном реестре
                elif token in self.scopes[0]:
                    self.stack.append(self.scopes[0][token])
                else:
                    raise Exception(f"Undefined State Variable: {token}")
                
            # 5. Instantiation: {ClassName [k->v]}
            elif token.startswith("{"):
                # Убираем скобки и лишние пробелы
                content = token.strip("{}").strip()
                parts = content.split()
                
                if parts and parts[0] in self.modules:
                    # Если первое слово - имя модуля (например {Linear})
                    self._instantiate(token)
                else:
                    # В противном случае это блок кода (лямбда)
                    # Просто кладем его на стек как строку для [FOREACH] или других нужд
                    self.stack.append(token)

            # 6. Opcodes & Literals
            elif token in self.ops: 
                self.ops[token]()
            
            elif token.startswith('['):
                # Проверка: Шейп ли это?
                # Шейпы обычно содержат цифры или lower_case переменные.
                # Если это похоже на ЗАБЫТЫЙ ОПКОД (все буквы заглавные), кидаем ошибку понятную.
                inner = token.strip("[]")
                if inner.isupper() and not any(c.isdigit() for c in inner):
                     # Это похоже на [TYPO] или [UNKNOWN_TAG], а не на [256 10]
                     print(f"Warning/Error: Unknown Opcode or Tag encountered: {token}")
                     # Можно сделать raise Exception, если хотите строгий режим
                else:
                    self._op_shape_literal(token)
            elif token.startswith('['): self._op_shape_literal(token) # [2 2]
            elif token.startswith('('): self._op_data_literal(token)  # (1 2)
            elif token.startswith('"'): self.stack.append(token.strip('"'))
            elif self._is_number(token): self.stack.append(torch.tensor(float(token)))
            elif self._resolve_local(token) is not None:
                self.stack.append(self._resolve_local(token))
            else:
                pass # print(f"Unknown: {token}")

            pc += 1

        if local_scope is not None: self.scopes.pop()

    # --- Module System ---
    def _parse_module(self, tokens, start_pc):
        # Format: "Name" [MODULE] ... [END_MODULE]
        name = self.stack.pop()
        mod_def = ModuleDef(name)
        
        curr = start_pc + 1
        mode = None # 'INIT' or 'FORWARD'
        
        while curr < len(tokens):
            t = tokens[curr]
            if t == "[END_MODULE]":
                curr += 1 # <--- ВАЖНО: Пропускаем сам токен [END_MODULE]
                break
            elif t == "[INIT]": mode = 'init'
            elif t == "[FORWARD]": mode = 'forward'
            elif t == "[RET]": mode = None
            else:
                if mode == 'init': mod_def.init_code.append(t)
                elif mode == 'forward': mod_def.forward_code.append(t)
            curr += 1
            
        self.modules[name] = mod_def
        print(f"System: Module '{name}' compiled.")
        return curr # Теперь возвращает индекс ПОСЛЕ модуля

    def _instantiate(self, token):
        # {Linear [out->10]}
        content = token.strip("{}")
        parts = content.split()
        class_name = parts[0]
        
        # Parse kwargs
        kwargs = {}
        for p in parts[1:]:
            if "->" in p:
                k, v = p.split("->")
                # Simple number parsing for config
                if self._is_number(v): v = torch.tensor(float(v))
                kwargs[k] = v
        
        if class_name not in self.modules: raise Exception(f"Unknown Module: {class_name}")
        
        instance = ModuleInstance(self.modules[class_name])
        
        # Run INIT phase
        old_ctx = self.context
        self.context = instance
        # Inject kwargs into local scope for INIT
        self.execute(instance.definition.init_code, local_scope=kwargs)
        self.context = old_ctx
        
        self.stack.append(instance)

    def _op_call(self):
        # stack: instance [CALL] -> result
        instance = self.stack.pop()
        if not isinstance(instance, ModuleInstance): raise Exception("CALL expects Instance")
        
        # Run FORWARD phase
        old_ctx = self.context
        self.context = instance
        # Arguments must be on stack already
        self.execute(instance.definition.forward_code, local_scope={})
        self.context = old_ctx

    # --- Autograd & Optimizer Primitives ---
    def _collect_params(self, instance):
        params = []
    # 1. Перебираем всё, что лежит в state текущего модуля
        for k, v in instance.state.items():
        # Если это тензор с градиентом — берем
            if isinstance(v, torch.Tensor) and v.requires_grad:
                params.append(v)
        # Если это ВЛОЖЕННЫЙ МОДУЛЬ — рекурсивно ныряем в него
            elif isinstance(v, ModuleInstance):
                params.extend(self._collect_params(v))
        return params

    def _op_params(self):
        instance = self.stack.pop()
        # Используем рекурсивный сборщик
        params = self._collect_params(instance)
        self.stack.append(params)

        
    def _op_foreach(self):
        # Стек: [list] [block]
        code_block = self.stack.pop() # "{ -> p ... }"
        target_list = self.stack.pop()
        
        # Убираем фигурные скобки, оставляя чистый код
        clean_code = code_block.strip("{}").strip()
        
        for item in target_list:
            self.stack.append(item) # Кладем элемент на стек
            self.execute(clean_code) # Выполняем блок

  

    def _op_grad(self):
        t = self.stack.pop()
        self.stack.append(t.grad if t.grad is not None else torch.zeros_like(t))

    def _op_sub_assign_grad(self):
        # stack: weight lr grad -> void (In-place update: w -= lr * grad)
        grad = self.stack.pop()
        lr = self.stack.pop()
        weight = self.stack.pop()
        
        # In-place operation to keep the reference inside the Instance valid
        # This is the "System" level optimization
        with torch.no_grad():
            weight.sub_(grad * lr)

    def _op_backward(self): self.stack.pop().backward()

    def _op_zero_grad(self):
        target = self.stack.pop()
        
        # Вариант 1: Очистка инстанса (Модуля)
        if isinstance(target, ModuleInstance):
            # Проходим по всем переменным состояния
            for key, val in target.state.items():
                if isinstance(val, torch.Tensor) and val.grad is not None:
                    val.grad.zero_()
                    
        # Вариант 2: Очистка списка тензоров (например, результат [PARAMS])
        elif isinstance(target, list):
            for val in target:
                if isinstance(val, torch.Tensor) and val.grad is not None:
                    val.grad.zero_()
                    
        # Вариант 3: Одиночный тензор
        elif isinstance(target, torch.Tensor) and target.grad is not None:
            target.grad.zero_()

    # --- Standard Ops ---
    def _op_mse_loss(self):
        target = self.stack.pop()
        pred = self.stack.pop()
        loss = torch.nn.functional.mse_loss(pred, target)
        self.stack.append(loss)
        
    def _op_shape_literal(self, token):
        # Токен вида "[in_dim out_dim]" или "[256 10]"
        content = token.strip("[]")
        parts = content.split()
        dims = []

        for p in parts:
            # 1. Если это число-литерал (например "256")
            if p.isdigit():
                dims.append(int(p))
            
            # 2. Если это переменная (например "in_dim")
            else:
                # Ищем в локальном скоупе (куда попали аргументы -> out_dim -> in_dim)
                val = self._resolve_local(p)
                
                # Если не нашли в локальном, ищем в атрибутах инстанса (@dim)
                if val is None and self.context and p in self.context.state:
                    val = self.context.state[p]
                
                if val is None:
                    raise ValueError(f"Shape Literal Error: Dimension variable '{p}' is undefined.")

                # Если это тензор (а в Ninth всё тензор), извлекаем int
                if isinstance(val, torch.Tensor):
                    dims.append(int(val.item()))
                else:
                    dims.append(int(val))

        # 3. Создаем тензор
        # По умолчанию считаем, что [Shape] используется для весов, поэтому randn + requires_grad
        # Для нулей используйте {ZEROS}
        self.stack.append(torch.randn(*dims, requires_grad=True))
    
    def _op_data_literal(self, token):
        # Безопасный парсинг S-expression без eval()
        # 1. Нормализуем пробелы вокруг скобок для сплита
        s = token.replace("(", " ( ").replace(")", " ) ")
        # 2. Разбиваем на токены
        parts = s.split()
        
        stack = [[]] # Корневой список
        
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
                    # Пытаемся распарсить число
                    val = float(p)
                    stack[-1].append(val)
                except ValueError:
                    pass # Игнорируем мусор, если попался
        
        # Результат лежит внутри корневого списка: [[1, 2]] -> [1, 2]
        if len(stack[0]) > 0:
            data = stack[0][0]
            # Создаем тензор
            self.stack.append(torch.tensor(data, dtype=torch.float32))
        else:
            # Пустой тензор
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
    def _op_squeeze(self):
        # Убирает все размерности равные 1
        self.stack.append(self.stack.pop().squeeze())

    def _op_unsqueeze(self):
        # Добавляет 1 в указанный индекс измерения
        dim = int(self.stack.pop().item())
        self.stack.append(self.stack.pop().unsqueeze(dim))

    def _op_flatten(self):
        # Схлопывает тензор в одномерный вектор
        self.stack.append(self.stack.pop().flatten())

    def _op_reshape(self):
        # Ожидает на стеке: Тензор, затем [Shape]
        # В Ninth v3.0 мы можем передать форму как список (результат парсинга [])
        # Но так как [2 2] сейчас создает тензор, нам нужен способ передать именно размеры.
        # Для простоты сделаем так: берем форму из тензора на вершине.
        shape_tensor = self.stack.pop()
        data_tensor = self.stack.pop()
        shape = shape_tensor.long().tolist() if shape_tensor.dim() > 0 else [int(shape_tensor.item())]
        self.stack.append(data_tensor.reshape(*shape))

# === RUNTIME DEMO ===
if __name__ == "__main__":
    vm = NinthVM()
    print("=== Ninth v3.0 Chimera: Self-Hosted Optimizer ===\n")

    # 1. Определяем слой Linear
    linear_code = """
    "Linear" [MODULE]
        [INIT]
            -> out_dim -> in_dim
            // std приходит из kwargs, если нет - 0.02 (упрощено)
            
            // Создаем веса с автоградом (стандартный [SHAPE] теперь делает requires_grad=True)
            [in_dim out_dim] -> @W
            [1 out_dim] -> @b 
        [RET]

        [FORWARD]
            -> x
            x @W [MATMUL] @b [ADD]
        [RET]
    [END_MODULE]
    """
    vm.execute(linear_code)

    # 2. Пишем SGD на самом Ninth!
    # Он принимает модель и lr, и имеет метод [FORWARD] (он же шаг)
    sgd_code = """
    "SGD" [MODULE]
        [INIT]
            -> lr -> target_model
            target_model -> @model
            lr -> @lr
        [RET]

        [FORWARD]
            [NO_GRAD]
                @model [PARAMS] // Получаем список тензоров весов
                
                // Итерируем
                { 
                    -> p
                    p [GRAD] -> g
                    // Вызываем low-level примитив обновления: p -= lr * g
                    p @lr g [SUB_ASSIGN_GRAD]
                } [FOREACH]
                
            [END_NO_GRAD]
        [RET]
    [END_MODULE]
    """
    vm.execute(sgd_code)

    # 3. Пользовательский скрипт
    main_script = """
    "--- Training Loop Start ---" [PRINT]

    // Создаем сеть (Вход 5, Выход 1)
    5 1 {Linear} -> @net

    // Создаем оптимизатор, передаем ему сеть
    @net 0.1 {SGD} -> @opt

    // Данные (Dummy Data)
    (1 1 1 1 1) -> input    // 1x5
    (0.5)       -> target   // Цель

    // Шаг обучения 1
    input @net [CALL]  ->> pred
    "Prediction 1:" [PRINT] pred [PEEK]

    pred[SQUEEZE] target[SQUEEZE]  [MSE_LOSS] ->> loss
    "Loss 1:" [PRINT] loss [PEEK]
    
    loss [BACKWARD] // Считаем градиенты
    
    @opt [CALL]     // Запускаем наш SGD на Ninth!
    
    // Шаг обучения 2 (проверяем, обучилось ли)
    @net [ZERO_GRAD] // (Условно, или внутри оптимизатора)
    
    input @net [CALL] ->> pred2
    "Prediction 2 (After Step):" [PRINT] pred2 [PEEK]
    
    pred2[SQUEEZE] target[SQUEEZE] [MSE_LOSS] ->> loss2
    "Loss 2:" [PRINT] loss2 [PEEK]
    """
    
    #vm.execute(main_script)
    main_script = """
    // === 1. Базовый Линейный Слой (как раньше) ===
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


// === 2. Композитный Модуль (Сеть из двух слоев) ===
// Это тест на вложенность инстансов!
"DeepNet" [MODULE]
    [INIT]
        // Аргументы: input_size, hidden_size, output_size
        -> out -> hidden -> in
        
        // Создаем первый слой (Input -> Hidden)
        in hidden {Linear} -> @layer1
        
        // Создаем второй слой (Hidden -> Output)
        hidden out {Linear} -> @layer2
    [RET]

    [FORWARD]
        -> x
        // Прогоняем через первый слой
        x @layer1 [CALL] ->> h_raw
        
        // Тут была бы активация, например Relu. 
        // Пока пропустим, будет Deep Linear Network.
        
        // Прогоняем через второй слой
        h_raw @layer2 [CALL]
    [RET]
[END_MODULE]


// === 3. Оптимизатор SGD (Тот самый, самописный) ===
"SGD" [MODULE]
    [INIT]
        -> lr -> target_model
        target_model -> @model
        lr -> @lr
    [RET]

    [FORWARD]
        [NO_GRAD]
            // Магия: [PARAMS] должен рекурсивно достать веса 
            // из @layer1 и @layer2 внутри DeepNet
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

// Создаем оптимизатор для этой сети
@net 0.05 {SGD} -> @opt

// Данные (Batch size 1, Features 10)
(1.0 0.5 0.5 1.0 0.0 0.0 1.0 0.5 0.5 1.0) -> input
(1.0) -> target // Мы хотим, чтобы сеть выдала 1.0

// --- ШАГ 1 ---
"Step 1:" [PRINT]
input @net [CALL] ->> pred1
pred1 [PEEK]

// Считаем ошибку
pred1[SQUEEZE] target[SQUEEZE] [MSE_LOSS] ->> loss1
"Loss 1:" [PRINT] loss1 [PEEK]

// Учимся
loss1 [BACKWARD]
@opt [CALL]      // Обновляем веса
@net [ZERO_GRAD] // Обнуляем градиенты (нужна реализация в VM!)

// --- ШАГ 2 ---
"Step 2:" [PRINT]
input @net [CALL] ->> pred2
pred2 [PEEK]

pred2[SQUEEZE] target[SQUEEZE] [MSE_LOSS] ->> loss2
"Loss 2:" [PRINT] loss2 [PEEK]

loss2 [BACKWARD]
@opt [CALL]
@net [ZERO_GRAD]

// --- ШАГ 3 ---
"Step 3:" [PRINT]
input @net [CALL] ->> pred3
pred3 [PEEK]

pred3[SQUEEZE] target[SQUEEZE] [MSE_LOSS] ->> loss3
"Loss 3:" [PRINT] loss3 [PEEK]
    """
    vm.execute(main_script)