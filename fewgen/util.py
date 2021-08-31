import torch

def all_to(inputs, device):
  if torch.is_tensor(inputs):
    return inputs.to(device)
  if isinstance(inputs, tuple):
    return tuple([all_to(i, device) for i in inputs])
  if isinstance(inputs, list):
    return [all_to(i, device) for i in inputs]
  if isinstance(inputs, dict):
    return {k:all_to(v, device) for k, v in inputs.items()}
