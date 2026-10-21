import sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir, out_dir = sys.argv[1], sys.argv[2]
f = glob.glob(f"{run_dir}/events.out.tfevents*")[0]
ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()

def series(tag):
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])

def smooth(y, w=0.9):
    out, last = [], y[0]
    for v in y:
        last = w * last + (1 - w) * v
        out.append(last)
    return np.array(out)

panels = [
    ("eval/return", "Eval return", None),
    ("eval/ee_tracking_error", "Eval EE tracking error (m)", "log"),
    ("train/mean_value_loss", "Value loss", "log"),
    ("train/mean_surrogate_loss", "Surrogate loss", None),
    ("train/mean_entropy", "Policy entropy", None),
    ("train/global_action_std", "Action std (mean)", None),
    ("train/mean_kl", "Mean KL", None),
    ("train/learning_rate", "Learning rate", "log"),
]
env_steps = dict(zip(*series("train/step")))

with PdfPages(f"{out_dir}/ppo_training_curves.pdf") as pdf:
    fig, axes = plt.subplots(4, 2, figsize=(11, 13))
    for ax, (tag, title, scale) in zip(axes.flat, panels):
        x, y = series(tag)
        ax.plot(x, y, color="#9bb7d4", lw=0.8, label="raw")
        ax.plot(x, smooth(y), color="#1f4e79", lw=1.8, label="smoothed (EMA 0.9)")
        ax.set_title(title); ax.set_xlabel("PPO iteration (x 2048 env steps)")
        if scale: ax.set_yscale(scale)
        ax.grid(alpha=0.3)
        fig_single, axs = plt.subplots(figsize=(6, 4))
        axs.plot(x, y, color="#9bb7d4", lw=0.8, label="raw")
        axs.plot(x, smooth(y), color="#1f4e79", lw=1.8, label="smoothed (EMA 0.9)")
        axs.set_title(title); axs.set_xlabel("PPO iteration (x 2048 env steps)")
        if scale: axs.set_yscale(scale)
        axs.grid(alpha=0.3); axs.legend()
        fig_single.tight_layout()
        fig_single.savefig(f"{out_dir}/{tag.replace('/', '_')}.png", dpi=150)
        plt.close(fig_single)
    axes.flat[0].legend()
    fig.suptitle("PPO on SO100 tracking (hw4 ex3)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f"{out_dir}/ppo_training_curves.png", dpi=130)
    pdf.savefig(fig)
print("saved to", out_dir)
