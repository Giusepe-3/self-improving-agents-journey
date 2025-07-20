import torch, torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST 
from torchvision.transforms import ToTensor
from dataclasses import dataclass
import copy
import numpy as np, random
import matplotlib.pyplot as plt

RNG_Seed = 42
torch.manual_seed(RNG_Seed)
np.random.seed(RNG_Seed)
random.seed(RNG_Seed)

LR_MIN, LR_MAX = 1e-5, 1e-1 # Lr space
MOM_MIN, MOM_MAX = 0.0, 0.90 # Mom space

MU = 20 # Parents keep each generation
LAMBDA = 40 # Children produced per generation
GENERATIONS = 30 # Total passes
PATIENCE = 5 # Gens without progress


# Start Dataset
train_full = MNIST(root=".", download=True, transform=ToTensor())
sub_idx = torch.arange(0, 6_000)
train_small = torch.utils.data.Subset(train_full, sub_idx)

# simple 90 / 10 split so fitness() can "see" a held-out set
val_size = int(0.1 * len(train_small))
train_size = len(train_small) - val_size
train_split, val_split = torch.utils.data.random_split(
    train_small, [train_size, val_size], generator=torch.Generator().manual_seed(RNG_Seed)
)

train_loader = DataLoader(train_split, batch_size=128, shuffle=True)
val_loader = DataLoader(val_split, batch_size=128, shuffle=False)
# End Dataset

# Start Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)
# End Model

# Start Fitness
def fitness(lr: float, mom: float)-> float:
    """
    One epoch of SGD on train_split, return validation accuracy ∈ [0,1].
    Re-seed to keep the evaluation deterministic.
    """
    torch.manual_seed(RNG_Seed)
    model = Net()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
    loss_fn = nn.CrossEntropyLoss()

    # training loop (one epoch)
    model.train()
    for xb, yb in train_loader:
        # Forward pass
        pred = model(xb)
        # Loss
        loss = loss_fn(pred, yb)
        # zero_grad -> backward -> step
        opt.zero_grad()
        loss.backward()
        opt.step() 
        pass
    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total
# End Model

# Start Chromosome
@dataclass
class Chromo:
    lr: float # Learning-rate gene
    mom: float # Momentum gene
    score: float = None # Gets filled after evaluation

def random_chromo() -> Chromo:
    """
    Unifrom sample for mom and lr.
    """
    lr = 10 ** np.random.uniform(np.log10(LR_MIN), np.log10(LR_MAX))
    mom = np.random.uniform(MOM_MIN, MOM_MAX)

    return Chromo(lr=lr, mom=mom) 

def init_population(size: int) -> list[Chromo]:
    pop = [random_chromo() for _ in range(size)]
    for c in pop:
        c.score = fitness(c.lr, c.mom)
    return pop 
# End Chromosome

# Start Mutate

def mutate(parent: Chromo, sigma_lr_log: float=0.3, sigma_mom: float=0.1) -> Chromo:
    """
    Return a *fresh* child Chromo lightly perturbed from parent.
    Not evaluation fitness inside. 
    """
    child = copy.deepcopy(parent)

    log_lr = np.log10(child.lr)
    log_lr += np.random.normal(0.0, sigma_lr_log)
    child.lr = np.clip(10 ** log_lr, LR_MIN, LR_MAX)

    child.mom += np.random.normal(0.0, sigma_mom)
    child.mom = np.clip(child.mom, MOM_MIN, MOM_MAX)

    child.score = None
    return child 

# End Mutate

cache = {}
# Start Eval

def evaluate(chromo: Chromo) -> float:
    key = (round(chromo.lr, 8), round(chromo.mom, 8))
    if key not in cache:
        cache[key] = fitness(chromo.lr, chromo.mom)
    chromo.score = cache[key]
    return chromo.score

# End Eval

# Init population
population = init_population(MU)
best_hist = [max(c.score for c in population)]


if __name__ == "__main__":
    # Main evolution loop
    no_improve = 0
    for gen in range(GENERATIONS):
        # Spawn children
        children: list[Chromo] = [
            mutate(np.random.choice(population)) for _ in range(LAMBDA)
        ]

        # Evaluate new guys
        for ch in children:
            evaluate(ch)
        
        # Survivor selection
        population = sorted(
            population + children,
            key=lambda c: c.score,
            reverse=True
        )[:MU]

        # Bookkeeping
        best = population[0].score
        best_hist.append(best)
        print(f"Gen {gen:02d} best={best:.10f} lr={population[0].lr:.2E} mom={population[0].mom:.2f}")

        # Early stopping
        if best > max(best_hist[:-1]):
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print("Plateau detected - stopping.")
            break

best_overall = population[0]
print("\n=== Finished ===")
print(best_overall)

plt.plot(best_hist)
plt.xlabel("Generation")
plt.ylabel("Best val acc")
plt.show()