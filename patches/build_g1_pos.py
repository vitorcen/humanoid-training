"""Rewrite dependencies/humanoid-bench/humanoid_bench/assets/robots/g1_torque.xml
to use PD position actuators (matching H1's setup), keeping filename unchanged
so all 27 g1_torque_*.xml envs auto-benefit.

_Why: HumanoidBench upstream gives G1 raw <motor> actuators (torque control),
making DR.Q sample-inefficient. H1 uses <position kp=... kv=...> (PD control).
Converting G1 to the same regime is the root-cause fix._

Run once:
    python patches/build_g1_pos.py
Then:
    cd dependencies/humanoid-bench && git diff > /home/david/work/humanoid-training/patches/g1-pos-control.patch
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "dependencies/humanoid-bench/humanoid_bench/assets/robots/g1_torque.xml"

# PD constants per joint class — mirrored from H1 (h1_mjx_feet_collisions_pos.xml)
PD_PER_NAME = {
    # leg/torso explicit
    "hip_pitch":      ("g1",         200,  5,  -200,  200),
    "hip_roll":       ("g1",         200,  5,  -200,  200),
    "hip_yaw":        ("g1",         200,  5,  -200,  200),
    "knee":           ("g1",         300,  6,  -300,  300),
    "ankle_pitch":    ("g1",          40,  2,   -40,   40),
    "ankle_roll":     ("g1",          40,  2,   -40,   40),
    "torso":          ("g1",         300,  6,  -200,  200),
    # arm
    "shoulder_pitch": ("arm_joint",  100,  2,   -40,   40),
    "shoulder_roll":  ("arm_joint",  100,  2,   -40,   40),
    "shoulder_yaw":   ("arm_joint",  100,  2,   -40,   40),
    "elbow_pitch":    ("arm_joint",  100,  2,   -18,   18),
    "elbow_roll":     ("arm_joint",  100,  2,   -18,   18),
    # hand (fine motor — low gain to avoid stalling fingers)
    "hand":           ("hand_joint",  10, 0.5, -2,    2),
}


def pd_params_for(jname: str):
    """Match joint name suffix to PD class."""
    base = jname.replace("left_", "").replace("right_", "").replace("_joint", "")
    if base in PD_PER_NAME:
        return PD_PER_NAME[base]
    # hand fingers: zero/one/two/three/four/five/six
    return PD_PER_NAME["hand"]


def extract_joint_ranges(xml_text: str) -> dict[str, str]:
    """Pull range attribute from each <joint name=... range=.../> in xml."""
    out = {}
    pat = re.compile(r'<joint\s+name="([^"]+)"[^>]*?range="([^"]+)"')
    for m in pat.finditer(xml_text):
        out[m.group(1)] = m.group(2)
    return out


def rewrite():
    text = XML.read_text()
    ranges = extract_joint_ranges(text)
    print(f"found {len(ranges)} joint ranges")

    # 1) Rewrite default class blocks
    text = text.replace(
        '      <default class="arm_joint">\n        <motor ctrlrange="-20 20"/>\n      </default>',
        '      <default class="arm_joint">\n        <position kp="100" kv="2" forcerange="-40 40"/>\n      </default>',
    )
    text = text.replace(
        '      <default class="hand_joint">\n        <motor ctrlrange="-0.7 0.7"/>\n      </default>',
        '      <default class="hand_joint">\n        <position kp="10" kv="0.5" forcerange="-2 2"/>\n      </default>',
    )

    # 2) Rewrite each <motor ... /> in actuator block
    def motor_to_position(m: re.Match) -> str:
        attrs = m.group(1)
        # extract name + joint
        name = re.search(r'name="([^"]+)"', attrs).group(1)
        joint = re.search(r'joint="([^"]+)"', attrs).group(1)
        cls = re.search(r'class="([^"]+)"', attrs).group(1)
        joint_range = ranges.get(joint)
        if joint_range is None:
            print(f"  ⚠ no range for {joint}, skipping")
            return m.group(0)
        # class-level PD already set in default; here we only set ctrlrange + forcerange override if explicit
        explicit_fr = re.search(r'ctrlrange="([^"]+)"', attrs)
        if cls == "g1":
            # leg/torso: take per-joint forcerange = motor ctrlrange (e.g., "-88 88")
            fr = explicit_fr.group(1) if explicit_fr else "-200 200"
            kp_kv = pd_params_for(joint)
            kp, kv = kp_kv[1], kp_kv[2]
            return (f'<position class="{cls}" name="{name}" joint="{joint}" '
                    f'kp="{kp}" kv="{kv}" forcerange="{fr}" ctrlrange="{joint_range}"/>')
        # arm/hand: rely on default kp/kv/forcerange, only override ctrlrange
        return (f'<position class="{cls}" name="{name}" joint="{joint}" '
                f'ctrlrange="{joint_range}"/>')

    text = re.sub(r'<motor\s+(class="(?:g1|arm_joint|hand_joint)"[^/]+)/>', motor_to_position, text)

    XML.write_text(text)
    n_pos = text.count("<position")
    n_motor = text.count("<motor")
    print(f"after rewrite: <position>={n_pos}, <motor>={n_motor}")
    if n_motor > 0:
        print("⚠ some <motor> tags remain — inspect manually")


if __name__ == "__main__":
    rewrite()
