import torch
a = torch.tensor([[3]]).type(torch.FloatTensor)
b = torch.tensor([[2]]).type(torch.FloatTensor)
c = torch.tensor([[2]]).type(torch.FloatTensor)
print(a.sub_(b))
