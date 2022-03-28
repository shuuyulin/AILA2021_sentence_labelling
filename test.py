import torch

# a = torch.Tensor([[1,0,3], [4,5,0]])
# print(a[:, a.nonzero()])

# M = torch.randn(16,4)
# idx_mask = torch.Tensor([[1,1,0,0]]*16).long()
# print(M)
# print((idx_mask != 0).sum())
# print(M[idx_mask != 0])
# print([M.select(i) for i in idx_mask.nonzero()])
import numpy as np
a = np.array([1,2,3,4,5,6,7,8,9])
print(a[[1,2,3]])