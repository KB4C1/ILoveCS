import torch
import gc

print(torch.cuda.memory_allocated())  # скільки зайнято
print(torch.cuda.memory_reserved())  