import torch

# 创建一个包含1到5的张量
tensor = torch.ones(1, 6)

a = tensor.shape
b = tensor.size()

print(a[0], a[1])