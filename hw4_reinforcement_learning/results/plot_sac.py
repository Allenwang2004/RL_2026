import sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# usage: python results/plot_sac.py logs/sac/<run> results [logs/ppo/<run>]
run_dir, out_dir = sys.argv[1], sys.argv[2]
ppo_dir = sys.argv[3] if len(sys.argv) > 3 else None

def load(d):
    f = glob.glob(f"{d}/events.out.tfevents*")[0]
    ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
    return ea

def series(ea, tag):
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])

def smooth(y, w=0.9):
    out, last = [], y[0]
    for v in y:
        last = w * last + (1 - w) * v
        out.append(last)
    return np.array(out)

ea = load(run_dir)
panels = [
    ("eval/return", "Eval return", None),
    ("eval/ee_tracking_error", "Eval EE tracking error (m)", "log"),
    ("train/critic_loss", "Critic loss", "log"),
    ("train/actor_loss", "Actor loss", None),
    ("train/alpha", "Temperature alpha", "log"),
    ("train/alpha_loss", "Alpha loss", None),
]
xlabel = "SAC iteration (x 2048 env steps)"

with PdfPages(f"{out_dir}/sac_training_curves.pdf") as pdf:
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax, (tag, title, scale) in zip(axes.flat, panels):
        x, y = series(ea, tag)
        for a in (ax,):
            a.plot(x, y, color="#e3b48f", lw=0.8, label="raw")
            a.plot(x, smooth(y), color="#9c4a12", lw=1.8, label="smoothed (EMA 0.9)")
            a.set_title(title); a.set_xlabel(xlabel)
            if scale: a.set_yscale(scale)
            a.grid(alpha=0.3)
        fig_single, axs = plt.subplots(figsize=(6, 4))
        axs.plot(x, y, color="#e3b48f", lw=0.8, label="raw")
        axs.plot(x, smooth(y), color="#9c4a12", lw=1.8, label="smoothed (EMA 0.9)")
        axs.set_title(title); axs.set_xlabel(xlabel)
        if scale: axs.set_yscale(scale)
        axs.grid(alpha=0.3); axs.legend()
        fig_single.tight_layout()
        fig_single.savefig(f"{out_dir}/sac_{tag.replace('/', '_')}.png", dpi=150)
        plt.close(fig_single)
    axes.flat[0].legend()
    fig.suptitle("SAC on SO100 tracking (hw4 ex4)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f"{out_dir}/sac_training_curves.png", dpi=130)
    pdf.savefig(fig)

    if ppo_dir:
        ea_ppo = load(ppo_dir)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, (tag, title, scale) in zip(axes, panels[:2]):
            for e, name, light, dark in [(ea_ppo, "PPO", "#9bb7d4", "#1f4e79"),
                                         (ea, "SAC", "#e3b48f", "#9c4a12")]:
                it, y = series(e, tag)
                steps = dict(zip(*series(e, "train/step")))
                x = np.array([steps.get(i, np.nan) for i in it])
                ax.plot(x, y, color=light, lw=0.6, alpha=0.7)
                ax.plot(x, smooth(y), color=dark, lw=1.8, label=f"{name} (EMA 0.9)")
            ax.set_title(title); ax.set_xlabel("Environment steps")
            if scale: ax.set_yscale(scale)
            ax.grid(alpha=0.3); ax.legend()
        fig.suptitle("PPO vs SAC (same environment steps)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{out_dir}/ppo_vs_sac.png", dpi=130)
        pdf.savefig(fig)
print("saved to", out_dir)
