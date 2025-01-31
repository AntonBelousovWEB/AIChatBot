import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создаем тензоры на нужном устройстве
a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

# Засекаем время на CPU
start = time.time()
c_cpu = a.cpu() @ b.cpu()
print("CPU time:", time.time() - start)

# Засекаем время на GPU
start = time.time()
c_gpu = a @ b  # GPU версия
torch.cuda.synchronize()  # Синхронизация
print("GPU time:", time.time() - start)