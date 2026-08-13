# Version decisions

Every version below was checked against package metadata or vendor documentation on
2026-08-13, not from memory. Sources are at the bottom.

## The stack

| Component | Pinned | Why this one |
| --- | --- | --- |
| Python | 3.12 | vLLM allows `>=3.10,<3.15`; TRL allows `>=3.10`. 3.12 sits inside both. |
| NVIDIA driver | **595.91.07** (open kernel modules) | R595 is the *Production Branch*, EOL Mar 2027. See below. |
| CUDA wheel variant | **cu129** | vLLM's default and best-tested wheel variant. |
| torch | **2.11.0+cu129** | Exactly what vLLM 0.26.0 pins (`torch==2.11.0`). |
| vLLM | **0.26.0** | The ceiling TRL imposes. See below. |
| transformers | **>=5.5.3** | vLLM 0.26.0's floor. Qwen3.5 needs `>=5.2`, so this covers it. |
| TRL | **1.10.0** | Latest; only needed for the later training phase. |
| datasets | **>=4.7.0** | TRL 1.10.0's floor (was `>=3.0`, too low). |
| accelerate | **>=1.4.0** | TRL 1.10.0's floor. |
| math-verify | **>=0.5.2** | TRL's `math-verify` extra floor (was `>=0.5`). |

## The binding constraint is TRL, not vLLM

This is the one that would have bitten us. TRL 1.10.0 declares:

```
vllm<=0.26.0,>=0.17.0 ; extra == "vllm"
```

So **vLLM 0.27.1 — the current release — is incompatible with TRL's vLLM integration.**
Installing the newest vLLM now means either downgrading it later or giving up
`trl[vllm]` when we move to the online/on-policy phase.

The chain resolves top-down:

```
TRL 1.10.0  --requires-->  vllm <= 0.26.0
vLLM 0.26.0 --requires-->  torch == 2.11.0,  transformers >= 5.5.3
torch 2.11.0 -------------> +cu129 build exists on download.pytorch.org
```

Costs of capping at 0.26.0: none for this experiment. Qwen3.5 support landed in vLLM
0.17, and v0.26.0's model registry contains `Qwen3_5ForConditionalGeneration` (verified
by reading `vllm/model_executor/models/registry.py` at tag `v0.26.0`) — which is the
architecture `Qwen/Qwen3.5-2B` actually declares. The engine kwargs the collection code
uses, `language_model_only` and `max_cudagraph_capture_size`, are both real `EngineArgs`
fields at that tag.

## Why driver 595.91.07

The machine had **no NVIDIA driver at all** — `lspci` sees both `GA100 [A100 SXM4 80GB]`,
but there is no kernel module, no `/dev/nvidia*`, and no `nvidia-smi`. So the driver was a
free choice, not an inherited constraint.

NVIDIA's CUDA repo for `debian13` offers three branches:

| Branch | Type | EOL | Verdict |
| --- | --- | --- | --- |
| R590 (`590.44.01`, `590.48.01`) | New Feature | Dec 2026 | Short-lived, no benefit over 595 |
| **R595 (`595.45.04` … `595.91.07`)** | **Production** | **Mar 2027** | **Chosen** |
| R610 (`610.43.02`, `610.57.04`) | New Feature | ~Aug 2026 | Effectively at EOL already |

R535 (LTS) is CUDA 12 only. R580 (LTS, EOL Jun 2028) would be the longest-lived option but
**is not published for debian13** — it is what the GCP Deep Learning images ship.

Driver 595 supports up to CUDA 13, so it runs cu129 wheels trivially: a newer driver
running an older CUDA runtime is plain backward compatibility.

**Open kernel modules, and compute-only.** `nvidia-kernel-open-dkms` conflicts with the
proprietary `nvidia-kernel-dkms` and needs `dkms>=3.1.8` (trixie has 3.2.2), `g++` (14.2),
and `firmware-nvidia-gsp`. Ampere GA100 supports the open modules. We install
`nvidia-driver-cuda` (which *Provides* `nvidia-smi`) and `libcuda1` (which *Provides*
`libcuda.so.1`, the thing torch actually links) but deliberately **not** `nvidia-driver-libs`
— that pulls the whole EGL/Vulkan/Wayland graphics stack onto a headless box.

**No CUDA toolkit.** torch and vLLM ship their own CUDA runtime as `nvidia-*` pip
packages. Installing `cuda-toolkit-13-x` would waste ~5 GB for nothing.

## Why cu129 and not cu126 or cu130

The old `pyproject.toml` pinned torch to a **cu126** index, justified by a comment reading
*"Driver here is 550.x (CUDA 12.4 max)"*. That driver never existed on this machine. Worse,
vLLM's PyPI wheel is compiled against CUDA 12.9, so a cu126 torch next to a cu129-built
vLLM is a genuine ABI mismatch — precisely the dependency hell to avoid. **The cu126 pin
is removed.**

cu130 is also self-consistent (torch 2.11.0+cu130 and a vLLM cu130 variant both exist, and
driver 595 supports CUDA 13). We do not take it: cu129 is vLLM's default variant and
therefore the most-tested path, and nothing about A100 (sm_80) benefits from CUDA 13.

Do **not** use `uv pip install vllm --torch-backend=auto` here. Auto-detection inspects the
driver, sees a CUDA-13-capable 595, and would likely pick cu130 — silently diverging from
the cu129 wheel vLLM actually ships on PyPI. The variant is pinned explicitly instead.

## Why we are not switching to a Deep Learning VM image

The latest DLVM release (M132, Apr 2026) is Ubuntu 22.04/24.04 with **driver 580, CUDA 12.9,
PyTorch 2.9, Python 3.10**. Debian families are discontinued.

Of that, the only piece we would actually consume is the driver — and 580 is *older* than
the 595 we can install. torch 2.9 is wrong (vLLM 0.26.0 needs 2.11.0) and Python 3.10 is
irrelevant because uv provides its own interpreter. Recreating also means surrendering this
`a2-ultragpu-2g` instance, and A100-80GB capacity in `us-central1-a` is frequently
constrained.

The command is kept in [setup.md](setup.md) as a fallback for the one scenario that
justifies it: the DKMS build of the 595 open kernel module failing against Debian 13's
`6.12.101+deb13-cloud-amd64` kernel.

## An experimental note, not a version note

The [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B) recommends
`temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0` for thinking mode, and warns
that *"Qwen3.5-2B is more prone to entering thinking loops compared to other Qwen3.5
models."* The collection defaults here are `temperature=1.0, presence_penalty=1.5`, chosen
to break those loops.

That deviation is not free. Sampling with a presence penalty means traces are **not** drawn
from π_S, which weakens the "on-policy" claim that reverse-KL distillation rests on. The
length sweep exists to decide this empirically; whichever setting is frozen for collection
should be recorded in `outputs/trajectories/manifest.json`.

## Sources

- [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) — wheel variants, cu129 default, `--torch-backend`
- TRL 1.10.0 and vLLM 0.26.0 PyPI metadata (`requires_dist`) — the version caps
- [vLLM v0.26.0 model registry](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/registry.py) — Qwen3.5 support
- [NVIDIA driver branch matrix](https://docs.nvidia.com/datacenter/tesla/drivers/supported-drivers-and-cuda-toolkit-versions.html) and [driver lifecycle](https://docs.nvidia.com/datacenter/tesla/drivers/driver-lifecycle.html)
- [CUDA minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- NVIDIA `debian13` repo `Packages` index — the versions actually installable
- [Deep Learning VM release notes](https://docs.cloud.google.com/deep-learning-vm/docs/release-notes) and [image families](https://docs.cloud.google.com/deep-learning-vm/docs/images)
- [Qwen/Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [transformers Qwen3.5 docs](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5) — requires transformers >= 5.2
