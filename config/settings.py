# Configuration for Ninth VM
VM_CONFIG = {
    'version': '0.5.1',
    'name': 'Ninth VM',
    'stack_size_limit': 1000,
    'memory_size_limit': 1000,
    'enable_tracing': False,
    'default_tensor_dtype': 'float32',
}

# Operation categories
OP_CATEGORIES = {
    'stack': ['DUP', 'DROP', 'SWAP', 'PEEK', 'PRINT'],
    'math': ['ADD', 'SUB', 'MUL', 'DIV', 'MATMUL', 'POW', 'SQRT', 'SUM', 'RELU', 'SIGMOID', 'ROUND'],
    'logic': ['EQ', 'GT', 'LT'],
    'memory': ['STORE', 'LOAD'],
    'generators': ['RANDN', 'ZEROS'],
    'autograd': ['VAR', 'BACKWARD', 'GRAD', 'ZERO_GRAD'],
    'control': ['DEF', 'CALL', 'IF', 'ELSE', 'ENDIF', 'REPEAT', 'END', 'RET'],
}