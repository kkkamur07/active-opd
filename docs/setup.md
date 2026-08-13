# Machine setup

Target: GCP `a2-ultragpu-2g` (2x A100 SXM4 80GB, 24 vCPU, 334 GB RAM), Debian 13 trixie,
kernel `6.12.101+deb13-cloud-amd64`. Version rationale is in [versions.md](versions.md).

Path A is recommended. Path B is for when the DKMS build fails.

---

## Path A: the existing VM

### 1. Grow the boot disk

The default boot disk is 10 GB with ~4.5 GB free. This pass needs ~35 GB, and ~80 GB
once the 9B teacher and checkpoints arrive. GCP disks grow online (no reboot, no data
loss) and can never shrink, so pick a size you are happy with.

Run this from outside the VM (your laptop, Cloud Shell, or the Console). The VM's
service account has only the default scopes and no `compute` scope, so gcloud inside the
VM fails with `insufficient scopes`:

```bash
gcloud compute disks resize instance-20260813-175138 \
  --zone=us-central1-a \
  --size=200GB
```

Then, inside the VM, check whether the partition and filesystem followed:

```bash
lsblk        # is sda1 the full disk size?
df -h /      # is the filesystem the full partition size?
```

On this VM they did, with no manual step. The Debian cloud image runs `growpart` and
`resize2fs` from `cloud-init` at boot, so a resize from 10 GB to 75 GB appeared as a
74.9 GB `sda1` and a 74 GB filesystem on the next boot.

Only if `lsblk` shows a partition smaller than the disk, grow it by hand:

```bash
sudo apt-get install -y cloud-guest-utils   # provides growpart
sudo growpart /dev/sda 1
sudo resize2fs /dev/sda1                     # ext4 grows while mounted
```

`resize2fs` is already present (`/sbin/resize2fs`, e2fsprogs 1.47.2).

The two 375 GB NVMe local SSDs (`/dev/nvme0n1`, `/dev/nvme0n2`) are attached, unformatted,
and unmounted. They are not needed once the boot disk is resized. Contents are lost
whenever the VM is stopped, so nothing durable should live on them.

### 2. Install the NVIDIA driver

Compute-only, open kernel modules, no CUDA toolkit. See [versions.md](versions.md).

```bash
# Kernel headers first: DKMS cannot build the module without them.
sudo apt-get update
sudo apt-get install -y "linux-headers-$(uname -r)" dkms g++

# NVIDIA's CUDA repo for Debian 13. cuda-keyring installs both the signing key
# and the apt source, so neither is hand-rolled.
curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Pin the R595 Production Branch, then install compute userspace + open kernel module.
sudo apt-get install -y nvidia-driver-pinning-595
sudo apt-get install -y \
  nvidia-kernel-open-dkms=595.91.07-1 \
  nvidia-driver-cuda=595.91.07-1 \
  libcuda1=595.91.07-1

sudo modprobe nvidia
sudo modprobe nvidia-uvm
nvidia-smi          # must list 2x A100-SXM4-80GB and driver 595.91.07

# Keep the driver resident so it is not torn down between runs.
sudo systemctl enable --now nvidia-persistenced
```

The DKMS build takes a few minutes and signs five modules (`nvidia`, `nvidia-modeset`,
`nvidia-drm`, `nvidia-peermem`, `nvidia-uvm`). `/dev/nvidia*` nodes do not exist until the
first client attaches, so an empty `ls /dev/nvidia*` before running `nvidia-smi` is normal.

Note that the repo's default candidate is the R610 New Feature Branch, so
`nvidia-driver-pinning-595` is what makes the R595 versions install.

Skip `nvidia-driver-libs` (EGL/Vulkan/Wayland graphics stack, pointless on a headless box)
and `cuda-toolkit-*`. Do not install the system CUDA toolkit through apt. The Python
environment installs the CUDA runtime packages required by torch and vLLM.

If the DKMS build fails, check `/var/lib/dkms/nvidia/*/build/make.log`, try
`590.48.01-1` as a fallback, and only then consider Path B.

### 3. Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd ~/active-opd
uv sync --extra vllm        # torch 2.11.0+cu130, vLLM 0.26.0
```

The Python requirement and dependency constraints are in `pyproject.toml`; the resolved
Python versions are in `uv.lock`. The lockfile currently selects torch 2.11.0+cu130,
vLLM 0.26.0, transformers 5.15.0, datasets 5.0.1, and accelerate 1.14.0. Do not run
`uv pip install vllm --torch-backend=auto`: it picks the variant from the driver rather
than from the vLLM wheel, which is the wrong input. See [versions.md](versions.md) for why
this is cu130 and not cu129.

Add `--extra train` when the training phase starts; that installs TRL 1.10.0 and its
vLLM integration. Do not use `--with-teacher` in the same process as the vLLM engine.
The teacher is loaded by Hugging Face and can require GPU memory that vLLM has reserved
for its KV cache.

### 4. Verify before spending GPU hours

```bash
uv run python -c "
import torch, transformers, vllm
print('torch       ', torch.__version__, '| cuda runtime', torch.version.cuda)
print('cuda avail  ', torch.cuda.is_available(), '| devices', torch.cuda.device_count())
print('transformers', transformers.__version__)
print('vllm        ', vllm.__version__)
from vllm.model_executor.models.registry import ModelRegistry
print('qwen3.5 arch', 'Qwen3_5ForConditionalGeneration' in ModelRegistry.get_supported_archs())
a = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
print('bf16 matmul ', bool(torch.isfinite(a @ a).all()))
"
```

Verified output on this machine:

```
torch        2.11.0+cu130 | cuda runtime 13.0
cuda avail   True | devices 2
transformers 5.15.0
vllm         0.26.0
qwen3.5 arch True
bf16 matmul  True
```

Check all of it, not just the torch lines. `import vllm` is the part that catches a CUDA
variant mismatch, and torch reports a perfectly healthy CUDA setup even when vLLM cannot
load at all. If `cuda avail` is False the driver is not loaded and every later job would
silently run on CPU. The `bf16 matmul` line is the cheapest proof that a kernel actually
executes on the device rather than merely enumerating it.

---

## Path B: fallback, recreate from a Deep Learning VM image

Only worth it if the 595 DKMS build cannot be made to work. This image ships driver 580
(older than 595), PyTorch 2.9 and Python 3.10, all of which we replace with the uv
environment above. The driver is the only part we use.

```bash
gcloud compute instances create aopd-vm \
  --zone=us-central1-a \
  --machine-type=a2-ultragpu-2g \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata="install-nvidia-driver=True"
```

`a2-ultragpu-2g` bundles its own GPUs and local SSDs, so no `--accelerator` or
`--local-ssd` flags are needed. `--scopes=cloud-platform` avoids the scope problem that
blocks disk resizing from inside the current VM. Ubuntu 24.04 may be available as
`common-cu129-ubuntu-2404-nvidia-580`. Check the image family before creating the VM.

To resolve the concrete image behind a family:

```bash
gcloud compute images describe-from-family common-cu129-ubuntu-2204-nvidia-580 \
  --project deeplearning-platform-release
```

Then skip to step 3. The driver is already installed. Note the machine may land on
different A100 capacity, and recreating gives up the current instance's reservation.
