import torch

def getWeights(dis, ab_params):
    """
    """
    dis = dis * ab_params[0]
    dis = dis + ab_params[1]
    w = torch.sigmoid(dis).sum(-1)
    w = w ** ab_params[2]
    return w