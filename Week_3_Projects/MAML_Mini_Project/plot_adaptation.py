import math, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from maml_sine import Net, task_batch, INNER_LR, device   # imports WITHOUT retraining

def adapt_and_plot(steps: int, style: str, label: str) -> None:
    model = Net().to(device)
    model.load_state_dict(torch.load("maml_sine.pth", map_location=device))

    opt = optim.SGD(model.parameters(), lr=INNER_LR)

    # one fresh sine‑wave task
    xs_s, ys_s, _, _ = task_batch(batch_size=1)
    xs_full = torch.linspace(-5, 5, 200).view(-1, 1).to(device)

    for _ in range(steps):
        loss = nn.functional.mse_loss(model(xs_s.squeeze(0)), ys_s.squeeze(0))
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        plt.plot(xs_full.cpu(), model(xs_full).cpu(), style, label=label, linewidth=2)

adapt_and_plot(0, "k--", "0 steps")
adapt_and_plot(1, "r-",  "1 step")
adapt_and_plot(5, "g-",  "5 steps")

# ground‑truth sine
amp, phase = torch.rand(1).item()*4+0.1, torch.rand(1).item()*math.pi
xs = torch.linspace(-5, 5, 200)
plt.plot(xs, amp*torch.sin(xs+phase), "b:", label="target")

plt.title("MAML fast adaptation")
plt.legend()
plt.savefig("adaptation.png")
plt.show()
print("Figure saved to adaptation.png")
