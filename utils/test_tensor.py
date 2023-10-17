import torch

# 1. Create a tensor t with shape (3, 275, 7)
t = torch.randn(3, 275, 7)

# 2. Create a tensor j with shape (3, 275) with random True or False values
j = torch.rand(3, 275) > 0.5  # This creates a tensor with values between 0 and 1, and then checks if they're greater than 0.5, which gives a roughly 50% chance for True or False.

print(t.shape)  # Should print torch.Size([3, 275, 7])
print(j.shape)  # Should print torch.Size([3, 275])
print(t[j].shape)        # Will print the tensor with random True/False values
