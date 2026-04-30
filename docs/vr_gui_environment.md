# VR/GUI Environment Reproduction

This repo keeps the original Evo-RL dependency setup and adds a locked path for the newer Quest3 VR teleoperation and GUI recording modules.

The baseline is the current `evo-rl-v1` environment on Linux x86_64. It contains the original Evo-RL runtime plus the VR/GUI additions. The important VR IK detail is that `evo-rl-v1` uses the faster CasADi/IPOPT/Pinocchio/OpenBLAS native stack; a nearby environment using newer Pinocchio/CasADi and netlib BLAS was much slower in the Piper IK micro-benchmark.

## Files

- `pyproject.toml`: project dependency declarations and extras, including `quest3_vr` and `gui`.
- `environment-vr-gui.yml`: maintainable conda environment with the key VR IK and GUI native packages pinned.
- `conda-linux-64-evo-rl-v1-explicit.txt`: explicit conda lock exported from `evo-rl-v1`.
- `requirements-vr-gui-lock.txt`: pip packages installed in `evo-rl-v1`, filtered from `conda list` to `channel == pypi`.

## Recommended Install

Use this path on a new Linux x86_64 machine when you want to reproduce the current machine as closely as possible:

```bash
git clone <repo-url>
cd Evo-RL

conda create -n evo-rl-v1 --file conda-linux-64-evo-rl-v1-explicit.txt
conda activate evo-rl-v1

pip install -r requirements-vr-gui-lock.txt
pip install --no-deps -e .
```

`--no-deps` is intentional. The conda and pip lock files already define the environment. Letting pip resolve `.[all]` again can upgrade packages that were deliberately locked for this VR/GUI baseline.

`requirements-vr-gui-lock.txt` includes the PyTorch CUDA 12.8 wheel index because the baseline uses `torch==2.7.1+cu128` and `torchvision==0.22.1+cu128`.

## Fallback Install

If the explicit conda lock cannot be used, create the maintainable environment instead:

```bash
git clone <repo-url>
cd Evo-RL

conda env create -f environment-vr-gui.yml
conda activate evo-rl-vr-gui
```

This route is less strict than the explicit lock, but it keeps the performance-sensitive IK packages pinned to the known baseline.

## ADB Requirement

Quest3 input uses the system `adb` executable to launch the companion app and stream logcat data. `adb` is not a Python or conda dependency, so install it at the OS level.

Current baseline:

```text
Android Debug Bridge version 1.0.41
Version 28.0.2-debian
Installed as /usr/lib/android-sdk/platform-tools/adb
```

Ubuntu/Debian install:

```bash
sudo apt update
sudo apt install android-sdk-platform-tools
```

Verify:

```bash
adb version
adb devices
```

`adb devices` must show the Quest headset with state `device`. If it shows `unauthorized`, authorize the host inside the headset. If no device appears, check the USB cable, headset developer mode, and udev/USB permissions.

## Verification

Check the Python stack:

```bash
python - <<'PY'
import casadi
import pinocchio
from PySide6.QtWidgets import QApplication
import qt_material
import qtawesome

print("casadi", casadi.__version__)
print("pinocchio", pinocchio.__version__)
print("VR/GUI imports OK")
PY
```

Expected baseline versions:

- `casadi`: `3.6.7`
- `pinocchio`: `3.2.0`
- `PySide6`: `6.11.0`
- `qt-material`: `2.17`
- `QtAwesome`: `1.4.2`

For IK performance, run the Piper IK micro-benchmark used during environment validation. The current `evo-rl-v1` baseline had a median `solve_ms` around `1.2 ms`; large regressions usually indicate that the native IK stack drifted.
