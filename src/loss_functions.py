from torch import sum, where, zeros
import torch

def sigmoid_cross_entropy_with_logits(input, target):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = input.clamp(min=0)
    # loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    loss = max_val - input * target + (1 + (-abs(input)).exp()).log()

    # loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduce=False, size_average=False)
    # # input = torch.autograd.Variable(torch.randn(3, 4))
    # # target = torch.autograd.Variable(torch.randn(3, 4))
    # loss = loss_fn(input, target)

    # print(input)
    # print(target)
    # print(loss)
    # print(input.size(), target.size(), loss.size())
    # print(loss.mean().size())
    # print(loss.sum())

    return loss

def binary_cross_entropy_with_logits(input, target):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    # max_val = input.clamp(min=0)
    # # loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    # loss = max_val - input * target + (1 + (-abs(input)).exp()).log()

    loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduce=False, size_average=False)
    # input = torch.autograd.Variable(torch.randn(3, 4))
    # target = torch.autograd.Variable(torch.randn(3, 4))
    loss = loss_fn(input, target)
    print(input)
    print(target)
    print(loss)
    print(input.size(), target.size(), loss.size())
    print(loss.mean().size())
    print(loss.mean())

    # return loss.mean()
    return loss

def f1_loss(input, target, epsilon=1E-8):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    tp = sum(input * target, dim=0) # size = [1, ncol]
    tn = sum((1 - target) * (1 - input), dim=0) # size = [1, ncol]
    fp = sum((1 - target) * input, dim=0) # size = [1, ncol]
    fn = sum(target * (1 - input), dim=0) # size = [1, ncol]
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    f1 = 2 * p * r / (p + r + epsilon)
    # f1 = where(f1 != f1, zeros(f1.size()), f1)
    return 1 - f1.mean()
