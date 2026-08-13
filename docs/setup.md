# Machine setup

Target: GCP `a2-ultragpu-2g` (2x A100 SXM4 80GB, 24 vCPU, 334 GB RAM), Debian 13 trixie,
kernel `6.12.101+deb13-cloud-amd64`. Version rationale is in [versions.md](versions.md).

Two paths. **Path A is the recommended one**; Path B exists for when the DKMS build fails.

---

## Path A — the existing VM

### 1. Grow the boot disk

The default boot disk is 10 GB with ~4.5 GB free. The stack needs ~35 GB for this pass and
~80 GB once the 9B teacher and checkpoints arrive. GCP disks grow **online** — no reboot,
no data loss — but they can never shrink, so pick a size you are happy with.

Run this **from outside the VM** (your laptop, Cloud Shell, or the Console). The VM's
service account has only the default scopes and no `compute` scope, so gcloud *inside* the
VM fails with `insufficient scopes`:

```bash
gcloud compute disks resize instance-20260813-175138 \
  --zone=us-central1-a \
  --size=200GB
```

Then, **inside** the VM, grow the partition and the filesystem:

```bash
sudo apt-get install -y cloud-guest-utils   # provides growpart
sudo growpart /dev/sda 1                     # 10.6GB partition -> whole disk
sudo resize2fs /dev/sda1                     # ext4 grows while mounted
df -h /                                      # confirm
```

`resize2fs` is already present (`/sbin/resize2fs`, e2fsprogs 1.47.2).

The two 375 GB NVMe local SSDs (`/dev/nvme0n1`, `/dev/nvme0n2`) are attached, unformatted,
and unmounted. They are not needed once the boot disk is resized, and their contents are
**lost whenever the VM is stopped**, so nothing durable should live on them.

### 2. Install the NVIDIA driver

Compute-only, open kernel modules, no CUDA toolkit — see [versions.md](versions.md).

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
nvidia-smi          # must list 2x A100-SXM4-80GB and driver 595.91.07
```

Deliberately **not** installed: `nvidia-driver-libs` (EGL/Vulkan/Wayland graphics stack,
pointless on a headless box) and `cuda-toolkit-*` (~5 GB; torch and vLLM ship their own
CUDA runtime as `nvidia-*` pip packages).

If the DKMS build fails, check `/var/lib/dkms/nvidia/*/build/make.log`, try
`590.48.01-1` as a fallback, and only then consider Path B.

### 3. Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd ~/active-opd
uv sync --extra vllm        # torch 2.11.0+cu129, vLLM 0.26.0
```

Everything — interpreter, CUDA-variant index, exact versions — is pinned in
`pyproject.toml` and locked in `uv.lock`. Do **not** run
`uv pip install vllm --torch-backend=auto`: it inspects the driver, sees a CUDA-13-capable
595, and picks cu130, silently diverging from the cu129 wheel vLLM ships on PyPI.

Add `--extra train` when the training phase starts; that pulls `trl[vllm]`.

### 4. Verify before spending GPU hours

```bash
uv run python -c "
import torch, vllm
print('torch      ', torch.__version__)
print('cuda avail ', torch.cuda.is_available())
print('devices    ', torch.cuda.device_count())
print('vllm       ', vllm.__version__)
"
```

Expect `torch 2.11.0+cu129`, `True`, `2`, `0.26.0`. If `cuda avail` is False the driver is
not loaded and every later job would silently run on CPU.

---

## Path B — fallback: recreate from a Deep Learning VM image

Only worth it if the 595 DKMS build cannot be made to work. This image ships driver **580**
(older than 595), PyTorch 2.9 and Python 3.10 — all of which we replace with the uv
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
blocks disk resizing from inside the current VM. Ubuntu 24.04 is available as
`common-cu129-ubuntu-2404-nvidia-580`.

To resolve the concrete image behind a family:

```bash
gcloud compute images describe-from-family common-cu129-ubuntu-2204-nvidia-580 \
  --project deeplearning-platform-release
```

Then skip to step 3 — the driver is already installed. Note the machine may land on
different A100 capacity, and recreating gives up the current instance's reservation.
