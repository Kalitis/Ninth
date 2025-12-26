import torch
import torch.nn as nn
import numpy as np

class FoNE(nn.Module):
    def __init__(self, dim, scales=32, min_freq=1.0, max_freq=1000.0):
        super().__init__()
        self.dim = dim
        self.scales = scales
        
        # --- Log-Linear Frequency Setup ---
        # Создаем частоты, распределенные по логарифмической шкале.
        # Это позволяет покрыть широкий динамический диапазон.
        # freq: от 2^0 до 2^k
        
        # start = log2(min_freq), stop = log2(max_freq)
        start = np.log2(min_freq)
        stop = np.log2(max_freq)
        
        # Генерируем показатели степени
        exponents = torch.linspace(start, stop, scales)
        
        # Сами частоты: 2^exponent * PI
        # PI нужен, чтобы периодичность совпадала с естественными циклами,
        # но для FoNE главное — разнообразие частот.
        freqs = (2 ** exponents) * np.pi
        
        # Регистрируем как буфер (не обучаемый параметр, но часть стейта)
        # Shape: [1, scales] для удобного бродкастинга
        self.register_buffer("B", freqs.unsqueeze(0))

        # --- Projection MLP ---
        # Превращает [sin, cos] признаки в вектор размерности модели (embedding_dim).
        # Даже случайная инициализация здесь работает как LSH (Locality Sensitive Hashing).
        self.projection = nn.Sequential(
            nn.Linear(scales * 2, dim),
            nn.GELU(), # GELU лучше ReLU для гладких числовых представлений
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        # x: скаляр или тензор формы [batch, 1]
        # Если пришел просто float или 0-d tensor, делаем unsqueeze
        if isinstance(x, float) or isinstance(x, int):
            x = torch.tensor([[float(x)]])
        elif isinstance(x, torch.Tensor) and x.dim() == 0:
            x = x.view(1, 1)
        elif isinstance(x, torch.Tensor) and x.dim() == 1:
            x = x.unsqueeze(1) # [batch] -> [batch, 1]
            
        # 1. Fourier Features
        # x: [B, 1], self.B: [1, scales] -> x_proj: [B, scales]
        x_proj = x @ self.B 
        
        # 2. Concatenate Sin & Cos
        # result: [B, scales * 2]
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        # 3. Project to Embedding Space
        return self.projection(features)