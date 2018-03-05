import torch
from torch import autograd
from torch import nn

# Define the architecture.
x0 = autograd.Variable(torch.Tensor([1, 1]))
# x1 = Variable(torch.Tensor([1, 0]))
# x2 = Variable(torch.Tensor([0, 1]))
# x3 = Variable(torch.Tensor([0, 0]))
w0 = autograd.Variable(torch.randn(2, 3), requires_grad=True)
w1 = autograd.Variable(torch.randn(3, 1), requires_grad=True)
y0 = autograd.Variable(torch.Tensor([10]))
# y1 = Variable(torch.Tensor([1]))
# y2 = Variable(torch.Tensor([1]))
# y3 = Variable(torch.Tensor([0]))

# Compute.
while True:
    # Propagation.
    a1 = nn.ReLU()(x0 @ w0)
    a2 = a1 @ w1
    loss = nn.L1Loss()(a2, y0)
    print(a2[0])
    input('enter to continue')
    # Backpropagation.
    loss.backward()
    w0.data -= 0.001 * w0.grad.data
    w1.data -= 0.001 * w1.grad.data
