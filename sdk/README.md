# soarmMoce-sdk

Minimal Python SDK for SoarmMoce.

## Quickstart

### 1) Create venv (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2) Use config + SDK
```python
from soarmMoce_sdk.api.robot import Robot

# run from repo root so ./configs/soarm_moce.yaml is valid
robot = Robot.from_config("./configs/soarm_moce.yaml")
robot.connect()
print(robot.get_joint_state())
robot.move_pose([0.1, 0.0, 0.15], [0.0, 0.0, 0.0])
robot.disconnect()
```

### Fallback (if you can't install)
```bash
PYTHONPATH=soarmMoce_sdk/src python3 examples/00_quickstart_connect.py
```

### Simulation demo (PyBullet)
```bash
pip install pybullet
python examples/03_sim_ik1_style.py --config ./configs/soarm_moce.yaml
```
