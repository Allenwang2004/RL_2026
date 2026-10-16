"""Interactive key configuration for the SO-100 teleop recorder.

Opens a small OpenCV window and walks you through each action one-by-one.
Press the key you want to assign to each action (look for our recommended keys).  The result is saved as
a JSON file that is loaded automatically throughout the homework.

Usage:
    python scripts/configure_keys.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Default output location (next to this script)
DEFAULT_KEYMAP_PATH = Path(__file__).resolve().parent.parent / "hw3" / "keymap.json"

# Every configurable action with a human-readable description (English, shown in
# the OpenCV window; Chinese, printed in the terminal).  Movement directions are
# relative to the arm, as seen from the "operator" camera behind the base.
ACTIONS: list[tuple[str, str, str]] = [
    ("move_up", "Move EE up (+Z) (recommended: UP ARROW)",
     "夾爪往上移 (+Z)（建議：上方向鍵）"),
    ("move_down", "Move EE down (-Z) (recommended: DOWN ARROW)",
     "夾爪往下移 (-Z)（建議：下方向鍵）"),
    ("move_left", "Move EE left (+Y) (recommended: LEFT ARROW)",
     "夾爪往手臂的左邊移 (+Y)，即 operator 畫面的左邊（建議：左方向鍵）"),
    ("move_right", "Move EE right (-Y) (recommended: RIGHT ARROW)",
     "夾爪往手臂的右邊移 (-Y)，即 operator 畫面的右邊（建議：右方向鍵）"),
    ("move_forward", "Move EE forward (+X) (recommended: W)",
     "夾爪往前伸，遠離底座 (+X)，即 operator 畫面的深處（建議：W）"),
    ("move_backward", "Move EE backward (-X) (recommended: S)",
     "夾爪往後收，靠近底座 (-X)（建議：S）"),
    ("rot_x_pos", "Rotate EE +X axis", "夾爪繞自身 X 軸正向旋轉（作業用不到，但仍要指定一個沒用過的鍵）"),
    ("rot_x_neg", "Rotate EE -X axis", "夾爪繞自身 X 軸反向旋轉（作業用不到，但仍要指定一個沒用過的鍵）"),
    ("rot_y_pos", "Rotate EE +Y axis", "夾爪繞自身 Y 軸正向旋轉（作業用不到，但仍要指定一個沒用過的鍵）"),
    ("rot_y_neg", "Rotate EE -Y axis", "夾爪繞自身 Y 軸反向旋轉（作業用不到，但仍要指定一個沒用過的鍵）"),
    ("rot_z_pos", "Rotate EE +Z axis", "夾爪繞自身 Z 軸正向旋轉（作業用不到，但仍要指定一個沒用過的鍵）"),
    ("rot_z_neg", "Rotate EE -Z axis", "夾爪繞自身 Z 軸反向旋轉（作業用不到，但仍要指定一個沒用過的鍵）"),
    ("gripper_open", "Open gripper (Jaw +)", "張開夾爪"),
    ("gripper_close", "Close gripper (Jaw -)", "閉合夾爪（夾住方塊）"),
    ("reset", "Reset environment (recommended: R)",
     "重置環境，若正在錄製會丟棄這一段（建議：R）"),
    ("record", "Toggle recording on/off (recommended: Space)",
     "開始／暫停錄製（建議：空白鍵）"),
    ("end_episode", "End recorded episode (recommended: Enter)",
     "結束並儲存這一段 episode，然後重置環境（建議：Enter）"),
    ("escape", "Quit session (recommended: ESC)",
     "儲存進行中的 episode 並離開程式（建議：ESC）"),
    ("goal_cube_red", "Select RED cube as goal (multicube only)",
     "選紅色方塊當目標（只有 --multicube 模式會用到）"),
    ("goal_cube_green", "Select GREEN cube as goal (multicube only)",
     "選綠色方塊當目標（只有 --multicube 模式會用到）"),
    ("goal_cube_blue", "Select BLUE cube as goal (multicube only)",
     "選藍色方塊當目標（只有 --multicube 模式會用到）"),
]

WINDOW_W = 640
WINDOW_H = 200


def draw_prompt(
    action_name: str, description: str, index: int, total: int
) -> np.ndarray:
    """Create a prompt image asking the user to press a key."""
    img = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    cv2.putText(
        img,
        f"Key config  ({index + 1}/{total})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
    )
    cv2.putText(
        img,
        f"Action: {description}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"({action_name})",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )
    cv2.putText(
        img,
        ">>> Press the key you want to use <<<",
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return img


def draw_assigned(action_name: str, raw: int, ascii_code: int) -> np.ndarray:
    """Show confirmation after a key was captured."""
    img = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    label = chr(ascii_code) if 32 <= ascii_code <= 126 else "<special>"
    cv2.putText(
        img,
        f"Assigned: {action_name}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"Key: '{label}'  (raw={raw}, ascii={ascii_code})",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2,
    )
    cv2.putText(
        img,
        "Moving to next action...",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )
    return img


def run_configuration(output_path: Path) -> None:
    """Walk through all actions and capture keys interactively."""
    cv2.namedWindow("Key Configuration", cv2.WINDOW_AUTOSIZE)

    keymap: dict[str, dict] = {}
    total = len(ACTIONS)

    print("按鍵設定：請點一下 OpenCV 視窗讓它取得焦點，再依照下面的提示逐一按下要指定的按鍵。")
    print("移動方向都是以手臂為準，也就是站在底座後方、看 operator 畫面時的前後左右。\n")

    for i, (action_name, description, description_zh) in enumerate(ACTIONS):
        prompt = draw_prompt(action_name, description, i, total)
        cv2.imshow("Key Configuration", prompt)
        print(f"  [{i + 1}/{total}] {action_name}: {description_zh}")

        # Wait for a key press (block until a key is pressed)
        while True:
            k_raw = cv2.waitKeyEx(0)  # 0 = wait forever
            if k_raw != -1:
                break

        k_ascii = k_raw & 0xFF
        label = chr(k_ascii) if 32 <= k_ascii <= 126 else f"raw:{k_raw}"

        keymap[action_name] = {
            "raw": k_raw,
            "ascii": k_ascii,
            "label": label,
            "description": description,
        }

        print(f"           -> 已指定 '{label}' (raw={k_raw})")

        # Brief confirmation
        confirm = draw_assigned(action_name, k_raw, k_ascii)
        cv2.imshow("Key Configuration", confirm)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(keymap, f, indent=2)
    print(f"\nKey mapping saved to {output_path}")
    print(
        "You can now run record_teleop_demos.py — it will load this mapping automatically."
    )
    print(f"按鍵設定已存到 {output_path}，之後執行 record_teleop_demos.py 會自動載入。")


def load_keymap(path: Path | None = None) -> dict[str, int]:
    """Load a keymap JSON and return {action_name: raw_keycode}.

    If no file exists, returns an empty dict (caller should use defaults).
    """
    if path is None:
        path = DEFAULT_KEYMAP_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {action: entry["raw"] for action, entry in data.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configure keyboard mapping for SO-100 teleop."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_KEYMAP_PATH,
        help=f"Output JSON file (default: {DEFAULT_KEYMAP_PATH}).",
    )
    args = parser.parse_args()
    run_configuration(args.output)
