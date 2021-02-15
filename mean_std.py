import torch


tensor = torch.Tensor(1000000)
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)

print(tensor)
tensor = torch.nn.functional.relu(tensor)
print(tensor)
print(tensor.mean().item())
print(tensor.std().item())
