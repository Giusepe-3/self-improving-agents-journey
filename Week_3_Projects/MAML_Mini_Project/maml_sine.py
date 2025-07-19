import math, torch, torch.nn as nn, torch.optim as optim, higher
from tqdm import trange
device = "cuda" if torch.cuda.is_available() else "cpu"

# Meta-data generator
def task_batch(batch_size=25, k_support=5, k_query=5):
    amps = torch.rand(batch_size, 1) * 4 + 0.1
    phases = torch.rand(batch_size, 1) * math.pi
    xs = torch.rand(batch_size, k_support+k_query, 1)*10 - 5
    ys = amps.unsqueeze(2) * torch.sin(xs + phases.unsqueeze(2))
    xs, ys = xs.to(device), ys.to(device)
    return xs[:, :k_support], ys[:, :k_support], xs[:, k_support:], ys[:, k_support:]

# Model and Optimizer
def Net(nn.Module):
    def __init__(self):
        sd