import math, torch, torch.nn as nn, torch.optim as optim, higher
from tqdm import trange

# ─────────────────── device & AMP ───────────────────
device   = "cuda" if torch.cuda.is_available() else "cpu"
use_amp  = (device == "cuda")
scaler   = torch.amp.GradScaler('cuda', enabled=use_amp)   # new API (no warning)

# ─────────────────── task sampler ───────────────────
def task_batch(batch_size=25, k_support=5, k_query=5):
    amps   = torch.rand(batch_size, 1) * 4 + 0.1
    phases = torch.rand(batch_size, 1) * math.pi
    xs     = torch.rand(batch_size, k_support + k_query, 1) * 10 - 5
    ys     = amps.unsqueeze(2) * torch.sin(xs + phases.unsqueeze(2))
    xs, ys = xs.to(device), ys.to(device)
    return xs[:, :k_support], ys[:, :k_support], xs[:, k_support:], ys[:, k_support:]

# ─────────────────── model ───────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.ReLU(),
            nn.Linear(40, 40), nn.ReLU(),
            nn.Linear(40, 1)
        )
    def forward(self, x): return self.net(x)

# ─────────────────── constants exposed to plot script ───────────────────
INNER_LR        = 1e-2          # needed by plot_adaptation.py

# ─────────────────── training loop (runs only if called directly) ───────────────────
if __name__ == "__main__":
    meta_model = Net().to(device)
    meta_optim = optim.Adam(meta_model.parameters(), lr=1e-3)
    inner_opt  = optim.SGD(meta_model.parameters(), lr=INNER_LR)

    INNER_STEPS    = 5
    TASKS_PER_META = 25
    META_STEPS     = 60_000

    for step in trange(META_STEPS, desc="Meta‑training"):
        meta_optim.zero_grad()
        loss_q = 0.

        # ----- iterate over tasks in the meta‑batch -----
        for _ in range(TASKS_PER_META):
            # fresh copy of the model for *this* task
            with higher.innerloop_ctx(
                    meta_model,
                    inner_opt,                        # can reuse the same optimiser obj
                    copy_initial_weights=False,
                    track_higher_grads=False) as (fnet, diffopt):

                xs_s, ys_s, xs_q, ys_q = task_batch()

                # inner loop: adapt to support set
                for _ in range(INNER_STEPS):
                    support_loss = nn.functional.mse_loss(fnet(xs_s), ys_s)
                    diffopt.step(support_loss)

                # evaluate adapted weights on query set
                loss_q += nn.functional.mse_loss(fnet(xs_q), ys_q)

        meta_loss = loss_q / TASKS_PER_META
        meta_loss.backward()
        meta_optim.step()

    torch.save(meta_model.state_dict(), "maml_sine.pth")
    print("Done – weights saved to maml_sine.pth")
