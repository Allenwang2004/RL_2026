"""Evaluate every checkpoint in a folder with the official eval harness and keep the best result.

Each ``*.pt`` in ``--folder`` is evaluated exactly like ``student_eval/run_eval.py``
(same compiled harness, 100 episodes, seed 42 by default). Only the signed result
file of the best checkpoint is kept, as ``<folder>/ex{N}_result.hwresult``.
All checkpoints are left untouched.

The result file is tied to the checkpoint's hash, so submit it together with
the best checkpoint renamed to ``ex{N}.pt``.

Usage:
    python scripts/eval_folder.py --exercise 2 \\
        --folder checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger20ep_2026-10-22_14-30-00
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = PROJECT_ROOT / "student_eval"
MODEL_PY = PROJECT_ROOT / "hw3" / "model.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official eval on every checkpoint in a folder and keep the best result."
    )
    parser.add_argument(
        "--exercise", type=int, required=True, choices=[1, 2, 3], help="Exercise number (1, 2, or 3)."
    )
    parser.add_argument(
        "--folder", type=Path, required=True, help="Folder with the .pt checkpoints to compare."
    )
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Episodes per checkpoint (default: 100)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Eval seed (default: 42, the official one)."
    )
    args = parser.parse_args()

    ckpts = sorted(args.folder.glob("*.pt"))
    if not ckpts:
        print(f"ERROR: no .pt files in {args.folder}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(HARNESS_DIR))
    import eval_harness  # noqa: E402  — compiled .so

    output = args.folder / f"ex{args.exercise}_result.hwresult"
    results: list[tuple[Path, dict]] = []
    best: tuple[Path, dict] | None = None

    with tempfile.TemporaryDirectory() as tmp:
        for i, ckpt in enumerate(ckpts, start=1):
            print(f"\n═══ [{i}/{len(ckpts)}] {ckpt.name} ═══")
            tmp_out = Path(tmp) / f"{ckpt.stem}.hwresult"
            res = eval_harness.run_eval(
                exercise=args.exercise,
                model_py=str(MODEL_PY),
                checkpoint=str(ckpt.resolve()),
                output_path=str(tmp_out),
                num_episodes=args.num_episodes,
                seed=args.seed,
                verbose=False,
            )
            print(f"  success {res['successes']}/{res['total']} ({res['rate'] * 100:.1f}%)")
            results.append((ckpt, res))
            if best is None or res["rate"] > best[1]["rate"]:
                best = (ckpt, res)
                output.write_bytes(tmp_out.read_bytes())

    assert best is not None
    print(f"\n{'=' * 55}")
    print(f"  Exercise {args.exercise}, {args.num_episodes} episodes, seed {args.seed}")
    for ckpt, res in results:
        mark = "  <- best" if ckpt == best[0] else ""
        print(f"  {ckpt.name:30s} {res['successes']:3d}/{res['total']} ({res['rate'] * 100:5.1f}%){mark}")
    print(f"{'=' * 55}")
    print(f"Best: {best[0]}  score {best[1]['score']:.0f}/{best[1]['max_score']:.0f}")
    print(f"Result file written to: {output}")
    print(f"Submit it together with {best[0].name} renamed to ex{args.exercise}.pt.")


if __name__ == "__main__":
    main()
