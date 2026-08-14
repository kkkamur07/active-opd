# Project guide

This is the single reference for machine setup, dependency choices, dataset limits, and rollout findings. The root [README](../README.md) keeps the experiment commands and project layout. The accepted training decision remains in [ADR 0001](adr/0001-trl-gkd-trainer-for-opd.md).

## Quick start

The rollout environment uses Python 3.12, PyTorch 2.11.0 with CUDA 13.0, and vLLM 0.26.0.

```bash
uv sync --extra vllm
```

Add the training dependencies only when that phase begins:

```bash
uv sync --extra vllm --extra train
```

Do not install vLLM with `uv pip install vllm --torch-backend=auto`. The vLLM wheel chooses the CUDA variant. The driver only sets the supported ceiling.

## Machine setup

The tested target is GCP `a2-ultragpu-2g`: two A100 SXM4 80 GB GPUs, 24 vCPUs, and 334 GB RAM on Debian 13 trixie with kernel `6.12.101+deb13-cloud-amd64`.

### 1. Grow the boot disk

The original boot disk was 10 GB, with about 4.5 GB free. This pass needs about 35 GB and about 80 GB after the 9B teacher and checkpoints arrive. Resize the disk outside the VM:

```bash
gcloud compute disks resize instance-20260813-175138 \
  --zone=us-central1-a \
  --size=200GB
```

Inside the VM, check the result:

```bash
lsblk
df -h /
```

The Debian cloud image normally grows the partition and filesystem at boot. If the partition is still smaller than the disk, run:

```bash
sudo apt-get install -y cloud-guest-utils
sudo growpart /dev/sda 1
sudo resize2fs /dev/sda1
```

The two 375 GB local NVMe disks are temporary. They are not needed after the boot disk is resized, and their contents disappear when the VM stops.

### 2. Install the NVIDIA driver

Install the compute-only open kernel module. Do not install the system CUDA toolkit or the graphics libraries.

```bash
sudo apt-get update
sudo apt-get install -y "linux-headers-$(uname -r)" dkms g++

curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

sudo apt-get install -y nvidia-driver-pinning-595
sudo apt-get install -y \
  nvidia-kernel-open-dkms=595.91.07-1 \
  nvidia-driver-cuda=595.91.07-1 \
  libcuda1=595.91.07-1

sudo modprobe nvidia
sudo modprobe nvidia-uvm
nvidia-smi
sudo systemctl enable --now nvidia-persistenced
```

`nvidia-smi` should show two A100-SXM4-80GB GPUs and driver 595.91.07. If the DKMS build fails, inspect `/var/lib/dkms/nvidia/*/build/make.log`, try `590.48.01-1`, and use the fallback below only if that fails too.

### 3. Install and verify Python dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/active-opd
uv sync --extra vllm
```

Run the full smoke test before spending GPU time:

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

The verified environment reported torch 2.11.0+cu130, CUDA runtime 13.0, two available devices, transformers 5.15.0, vLLM 0.26.0, Qwen3.5 architecture support, and a successful bf16 matmul. The important checks are `import vllm` and `bf16 matmul`, not just `torch.cuda.is_available()`.

### Fallback VM image

Use this only if the R595 DKMS build cannot work on Debian 13. The image supplies driver 580, which is older than the selected 595 driver. It also supplies PyTorch 2.9 and Python 3.10, both of which this project replaces with the uv environment.

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

Check the image family before using it. To resolve its current image:

```bash
gcloud compute images describe-from-family common-cu129-ubuntu-2204-nvidia-580 \
  --project deeplearning-platform-release
```

## Dependency choices

The lockfile is the source of truth for Python package versions. The NVIDIA driver is installed outside uv and must be checked separately.

| Component | Version | Reason |
| --- | --- | --- |
| Python | 3.12 | Supported by vLLM and TRL, and selected by the project. |
| NVIDIA driver | R595, 595.91.07 | Production branch; the driver supports the CUDA 13.x wheels. |
| CUDA wheel variant | cu130 | vLLM 0.26.0 requires a CUDA 13 runtime. |
| torch | 2.11.0+cu130 | vLLM 0.26.0 pins torch 2.11.0. |
| vLLM | 0.26.0 | TRL 1.10.0 declares `vllm>=0.17.0,<=0.26.0`. |
| transformers | 5.15.0 | Meets the vLLM and Qwen3.5 requirements. |
| TRL | 1.10.0 | Used by the later training phase. |
| datasets | 5.0.1 | Meets TRL's requirement. |
| accelerate | 1.14.0 | Meets TRL's requirement. |
| math-verify | 0.9.0 | Meets the required math-verify extra floor. |

vLLM 0.26.0 supports the `Qwen3_5ForConditionalGeneration` architecture used by `Qwen/Qwen3.5-2B`. Moving to vLLM 0.27 or later would break the planned TRL integration until TRL raises its version ceiling.

### Why cu130

The vLLM wheel declares `nvidia-cutlass-dsl[cu13]==4.6.0` and its compiled extension links `libcudart.so.13`. A cu129 torch installs `libcudart.so.12`, so torch can pass its CUDA checks while `import vllm` fails. The earlier cu126 pin came from a driver assumption that does not apply to this machine.

CUDA minor versions do not need to match. This machine has a 13.2 driver, a 13.0 runtime, and some 13.3 toolkit components. CUDA 13.x minor-version compatibility covers that combination above the 580.65.06 driver floor. The major version must match.

### Sampling defaults

The collection defaults follow the Qwen3.5-2B model card for thinking-mode math tasks:

```text
temperature=1.0
top_p=0.95
top_k=20
presence_penalty=1.5
```

The presence penalty means traces are not sampled from the unmodified student policy. Record the settings in `outputs/trajectories/manifest.json` when collecting data.

## CUDA troubleshooting

### `ImportError: libcudart.so.13`

This means torch was installed as a CUDA 12 build while vLLM 0.26.0 expects CUDA 13. Recreate the environment from the pinned project configuration:

```bash
rm -rf .venv uv.lock
uv sync --extra vllm
```

Confirm the wheel's requirement and the installed runtime:

```bash
uv run python -c "import importlib.metadata as m; print([r for r in m.requires('vllm') if 'cutlass' in r])"
ls .venv/lib/python3.12/site-packages/nvidia/
find .venv -name 'libcudart.so.*'
```

A working setup includes `cu13` and `libcudart.so.13`. Do not diagnose this with `torch.cuda.is_available()` alone. That check can pass in the broken setup.

### FlashInfer compiler error

The wheel set contains nvcc 13.3.73 and CUDA runtime headers 13.0.96. Runtime compatibility does not make those compiler inputs interchangeable. FlashInfer can fail with:

```text
CUDA compiler and CUDA toolkit headers are incompatible
```

`apod.models.generate_vllm.build_llm` points `CUDA_HOME` at the nvcc wheel and disables the FlashInfer sampler when the versions differ. The native sampler uses the same top-k and top-p distribution. A per-request seed also selects the native path in this experiment.

Do not try to fix this by changing one CUDA package in isolation. The runtime version is pinned through torch, and the alternative nvcc pin still leaves the wheel layout incompatible with FlashInfer's `lib64` assumption. The native sampler is the supported path here.

## Dataset contract and limits

### OpenThoughts math

The loader reads `siyanzhao/Openthoughts_math_30k_opsd`, split `train`. The split has 29,434 rows and 11 columns. The loader uses an explicit mapping:

| Loader field | Dataset column |
| --- | --- |
| `problem` | `problem` |
| `answer` | `Answer` |
| `solution` | `solution` |
| `cot` | `COT_Reason` |
| `source` | `source` |
| `dataset_correct` | `correct` |

It filters first to non-empty `problem` and `Answer`, leaving 29,427 usable rows. Seven olympiad rows have an empty answer because their reference solution ends in an empty `\boxed{}`. The loader samples from the usable pool with `random.Random(seed).sample`, keeps the original row index in the id, and raises instead of returning fewer rows than requested.

`Question` is just `problem` with the dataset's answer instruction prepended. The loader builds the same prompt from `problem` so OpenThoughts and MATH-500 share the format. The `messages` and `conversations` columns are redundant encodings and are not used.

Important audit findings:

- `correct` is `True` for every row. It describes the stored dataset generation, not a new rollout, so it is not a quality filter.
- A sample of 800 reference solutions agreed with their own `Answer` only 98.2% of the time. Treat roughly 97 to 98% as the practical grading ceiling for this pool, not 100%.
- Some answers are prose, intervals, lists, inequalities, or choice letters. Math-Verify can parse them while still comparing a generated answer differently.
- Twenty-two AoPS problems contain a leaked solution block. On 13 of those, the gold answer appears in the prompt. They remain in the pool because the loader's contract is only the non-empty problem and answer filter.
- The source mix is 72.4% olympiads, 18.2% math, 7.8% AoPS forum, and 1.6% AMC/AIME. Per-source results from a 512-example run are noisy, especially for AMC/AIME.

### MATH-500

`HuggingFaceH4/MATH-500`, split `test`, has 500 rows and no blank answers. Its gold column is lowercase `answer`, and it has no source or chain-of-thought column. Since the default request is 512 examples, `load_examples("math500")` raises instead of silently returning a short sample.

## Rollout findings

These findings explain the current implementation. They are not generic tuning advice.

### Presence penalty was the throughput bottleneck

With `presence_penalty=1.5`, the original vLLM path rebuilt a CPU penalty mask from the full generated history on every decode step. The cost grew with trace length and caused throughput to decay while GPU utilization stayed near 20%.

`apod/models/presence_penalty.py` keeps a boolean token-presence mask on the GPU and scatters only new tokens into it. It also clears and moves rows when vLLM reuses batch slots. The native vLLM penalty field stays at zero, while the incremental processor carries the experiment's penalty through `SamplingParams.extra_args`.

The optimized path measured about 4,600 generated tokens per second per shard, or about 9,200 combined, at 70 to 73% GPU utilization for 128 prompts per A100 with uncapped traces. A greedy 8-sequence by 256-token comparison produced identical output through both penalty paths.

### Chunk size and engine settings

Larger `llm.generate` chunks keep more sequences in flight and reduce the tail where the last long traces run alone. They also delay writes, so a crash loses the current chunk. Individual completion durability would require driving `LLMEngine.step()` directly.

The relevant vLLM defaults were already appropriate: `max_num_seqs=256`, `max_num_batched_tokens=8192`, prefix caching enabled, CUDA graphs enabled, and no meaningful preemption risk for the planned workload. Raising the token budget was flagged as an A100 regression in vLLM's source. FP8 KV cache was not needed because KV capacity was not the bottleneck.

FlashInfer is not worth forcing back on. Its fused sampler was about 1.86 times faster for the sampling operation alone, but sampling was only about 0.7 ms of a 10 to 15 ms decode step, and seeded requests use the native sampler anyway.

## Sources

- [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [vLLM 0.26.0 model registry](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/registry.py)
- [NVIDIA driver branches](https://docs.nvidia.com/datacenter/tesla/drivers/supported-drivers-and-cuda-toolkit-versions.html)
- [CUDA minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [Deep Learning VM release notes](https://docs.cloud.google.com/deep-learning-vm/docs/release-notes)
- [Qwen/Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)
