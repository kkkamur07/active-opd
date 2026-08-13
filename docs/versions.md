# Version decisions

Every version below was checked against package metadata or vendor documentation on
2026-08-13, not from memory. Sources are at the bottom.

## The stack

The table separates requirements from the versions currently resolved in `uv.lock`.
The lockfile is the source of truth for Python packages. The NVIDIA driver is installed
outside uv and must be checked against the repository before installation.

| Component | Requirement or choice | Current resolved version | Why this one |
| --- | --- | --- | --- |
| Python | 3.12 | 3.12 | vLLM allows `>=3.10,<3.15`; TRL allows `>=3.10`. 3.12 sits inside both. |
| NVIDIA driver | R595, open kernel modules | **595.91.07** (installed) | R595 is the Production Branch, with EOL listed as Mar 2027. See below. Confirm that this exact package is still available before installing it. |
| CUDA wheel variant | **cu130** | **cu130** | vLLM 0.26.0's wheel is a CUDA 13 build, so torch must match it. See below. |
| torch | **2.11.0** | **2.11.0+cu130** | vLLM 0.26.0 requires `torch==2.11.0`; the uv source selects the cu130 build. |
| vLLM | **0.26.0** | **0.26.0** | TRL 1.10.0 caps its vLLM extra at this version. |
| transformers | `>=5.5.3` | 5.15.0 | vLLM 0.26.0's floor. Qwen3.5 needs `>=5.2`. |
| TRL | **1.10.0** | 1.10.0 | Used by the later training phase. It is not installed by the default rollout extra. |
| datasets | `>=4.7.0` | 5.0.1 | TRL 1.10.0's floor. |
| accelerate | `>=1.4.0` | 1.14.0 | TRL 1.10.0's floor. |
| math-verify | `>=0.5.2` | 0.9.0 | TRL's `math-verify` extra floor. |

## TRL is the binding constraint, not vLLM

TRL 1.10.0 declares:

```
vllm<=0.26.0,>=0.17.0 ; extra == "vllm"
```

vLLM 0.27.1, the current release, is incompatible with TRL's vLLM integration.
Installing the newest vLLM now means either downgrading it later or giving up
`trl[vllm]` when we move to the online/on-policy phase.

The chain resolves top-down:

```
TRL 1.10.0  --requires-->  vllm <= 0.26.0
vLLM 0.26.0 --requires-->  torch == 2.11.0,  transformers >= 5.5.3
torch 2.11.0 -------------> +cu130 build exists on download.pytorch.org
```

Capping at 0.26.0 costs nothing for this experiment. Qwen3.5 support landed in vLLM
0.17, and v0.26.0's model registry contains `Qwen3_5ForConditionalGeneration` (verified
by reading `vllm/model_executor/models/registry.py` at tag `v0.26.0`), which is the
architecture `Qwen/Qwen3.5-2B` actually declares. The engine kwargs the collection code
uses, `language_model_only` and `max_cudagraph_capture_size`, are both real `EngineArgs`
fields at that tag.

## Why driver 595.91.07

The machine had no NVIDIA driver at all. `lspci` sees both `GA100 [A100 SXM4 80GB]`,
but there is no kernel module, no `/dev/nvidia*`, and no `nvidia-smi`. The driver was a
free choice, not an inherited constraint.

NVIDIA's CUDA repo for `debian13` offers three branches:

| Branch | Type | EOL | Verdict |
| --- | --- | --- | --- |
| R590 (`590.44.01`, `590.48.01`) | New Feature | Dec 2026 | Short-lived, no benefit over 595 |
| **R595 (`595.45.04` … `595.91.07`)** | **Production** | **Mar 2027** | **Chosen** |
| R610 (`610.43.02`, `610.57.04`) | New Feature | ~Aug 2026 | Effectively at EOL already |

R535 (LTS) is CUDA 12 only. R580 (LTS, EOL Jun 2028) would be the longest-lived option but
is not published for debian13. It is what the GCP Deep Learning images ship.

Driver 595 reports CUDA 13.2, comfortably above the 580.65.06 floor for CUDA 13.x, so it
runs the cu130 wheels used here.

Open kernel modules, and compute-only. `nvidia-kernel-open-dkms` conflicts with the
proprietary `nvidia-kernel-dkms` and needs `dkms>=3.1.8` (trixie has 3.2.2), `g++` (14.2),
and `firmware-nvidia-gsp`. Ampere GA100 supports the open modules. We install
`nvidia-driver-cuda` (which Provides `nvidia-smi`) and `libcuda1` (which Provides
`libcuda.so.1`, the thing torch actually links) but not `nvidia-driver-libs`, which
pulls the whole EGL/Vulkan/Wayland graphics stack onto a headless box.

Do not install the system CUDA toolkit through apt. The Python environment installs the
CUDA runtime and related packages required by torch and vLLM. Installing
`cuda-toolkit-13-x` at the system level would add about 5 GB that this project does not
need.

## Why cu130 and not cu129 or cu126

vLLM 0.26.0's PyPI wheel is a CUDA 13 build. Its metadata requires
`nvidia-cutlass-dsl[cu13]==4.6.0`, and its compiled extension links `libcudart.so.13`:

```
$ ldd .venv/.../vllm/_C_stable_libtorch*.so | grep cudart
        libcudart.so.13 => not found
```

A cu129 torch installs `libcudart.so.12` instead, so `import vllm` fails outright with
`ImportError: libcudart.so.13: cannot open shared object file`. This was hit on this
machine: the stack was first synced as cu129, torch itself worked (`cuda avail True`, two
A100s, bf16 matmul fine), and only `import vllm` failed. Switching the index to cu130
moved the whole `nvidia-*` set to cu13 and resolved it.

An earlier revision of this document recommended cu129 on the grounds that it was vLLM's
default and best-tested variant. That was true of older vLLM releases and is not true of
0.26.0. The wheel decides the variant; the driver only sets a ceiling.

Driver 595 reports `CUDA Version: 13.2`, and the wheels carry a 13.3 runtime. That is fine:
the CUDA 13.x driver floor is 580.65.06, and minor version compatibility covers a newer
13.x runtime on an older 13.x driver.

cu126 was the original pin, justified by a comment reading "Driver here is 550.x (CUDA 12.4
max)". That driver never existed on this machine, and cu126 is two variants away from what
vLLM needs. It is removed.

Do not use `uv pip install vllm --torch-backend=auto`. It infers the variant from the
driver rather than from the vLLM wheel, which is the wrong input. The variant is pinned
explicitly in `pyproject.toml` instead.

## Why not a Deep Learning VM image

The latest DLVM release (M132, Apr 2026) is Ubuntu 22.04/24.04 with driver 580, CUDA 12.9,
PyTorch 2.9, Python 3.10. Debian families are discontinued.

Of that, the only piece we would actually consume is the driver, and 580 is older than
the 595 we can install. torch 2.9 is wrong (vLLM 0.26.0 needs 2.11.0) and Python 3.10 is
irrelevant because uv provides its own interpreter. Recreating also means surrendering this
`a2-ultragpu-2g` instance, and A100-80GB capacity in `us-central1-a` is frequently
constrained.

The command is kept in [setup.md](setup.md) as a fallback for the one scenario that
justifies it: the DKMS build of the 595 open kernel module failing against Debian 13's
`6.12.101+deb13-cloud-amd64` kernel. Driver package versions and image-family names can
change, so check the linked repositories before using the commands.

## Sampling defaults

The [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B) recommends
`temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` for thinking-mode text
tasks. It also warns that "Qwen3.5-2B is more prone to entering thinking loops compared to
other Qwen3.5 models." Those are the collection defaults in this repository.

That deviation is not free. Sampling with a presence penalty means traces are not drawn
from π_S, which weakens the "on-policy" claim that reverse-KL distillation rests on. The
length sweep exists to measure the effect of the cap and sampling settings. Record the
settings used for collection in `outputs/trajectories/manifest.json`.

## Sources

- [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/), wheel variants and `--torch-backend`
- vLLM 0.26.0 wheel metadata (`Requires-Dist: nvidia-cutlass-dsl[cu13]==4.6.0`) and `ldd` of its compiled extension, both read on this machine
- TRL 1.10.0 and vLLM 0.26.0 PyPI metadata (`requires_dist`), the version caps
- [vLLM v0.26.0 model registry](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/registry.py), Qwen3.5 support
- [NVIDIA driver branch matrix](https://docs.nvidia.com/datacenter/tesla/drivers/supported-drivers-and-cuda-toolkit-versions.html) and [driver lifecycle](https://docs.nvidia.com/datacenter/tesla/drivers/driver-lifecycle.html)
- [CUDA minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- NVIDIA `debian13` repo `Packages` index, the versions actually installable
- [Deep Learning VM release notes](https://docs.cloud.google.com/deep-learning-vm/docs/release-notes) and [image families](https://docs.cloud.google.com/deep-learning-vm/docs/images)
- [Qwen/Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [transformers Qwen3.5 docs](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), requires transformers >= 5.2
