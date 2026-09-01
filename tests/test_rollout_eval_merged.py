"""The merged eval+rollout generate stream produces exactly what two streams did.

Runs ``apod.stages.rollout_eval.main`` against a deterministic fake vLLM
engine (outputs are a pure function of the rendered text and the request's
SamplingParams, so batch composition cannot change them) twice per scenario:
once with the pre-merge stage and generator (``git show OLD_REV:...``, the
last commit before the merge) and once with the current code. Both must hand
the engine the same request sequence (texts and SamplingParams, seeds
included) and leave byte-identical eval rows, trajectory rows, npz arrays
and markers behind. Scenarios: full round, ``--eval-only``, ``--eval-only
--eval-dataset`` on a stamped set with ``--eval-num-problems``, resume after
a finished eval (rollouts only, no eval regeneration), resume mid-rollouts,
and two shards. The merge itself is checked too: fewer generate calls, a
chunk holding both kinds, and the eval marker written only once every eval
request is graded.

Needs vLLM importable (CPU is fine) for the real SamplingParams; runs under
pytest or as ``python -m tests.test_rollout_eval_merged``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
OLD_REV = "c7e7ef0"  # last commit with separate eval / rollout generate streams
EOS_ID = 7
EVAL_SEED_OFFSET = 100000
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # collect_eos_ids must not hit the Hub


# --- fake engine --------------------------------------------------------------


class FakeTokenizer:
    eos_token_id = EOS_ID
    pad_token_id = None  # exercises the pad = eos fallback

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt
        return f"<user>{messages[0]['content']}<assistant>" + ("<think>" if enable_thinking else "")


class FakeCompletion:
    def __init__(self, text, token_ids, finish_reason):
        self.text, self.token_ids, self.finish_reason = text, token_ids, finish_reason


class FakeRequestOutput:
    num_cached_tokens = 0

    def __init__(self, prompt_token_ids, outputs):
        self.prompt_token_ids, self.outputs = prompt_token_ids, outputs


class FakeLLM:
    """Deterministic per (text, SamplingParams); records every generate call."""

    def __init__(self, calls: list, watch: dict[str, Path]):
        self.calls = calls
        self.watch = watch

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, texts, params):
        params = list(params) if isinstance(params, (list, tuple)) else [params] * len(texts)
        assert len(params) == len(texts)
        self.calls.append(
            {
                "requests": [(text, _fields(p)) for text, p in zip(texts, params)],
                "markers": {name: path.exists() for name, path in self.watch.items()},
            }
        )
        return [self._output(text, p) for text, p in zip(texts, params)]

    @staticmethod
    def _output(text, p):
        prompt_ids = [100 + (ord(c) % 50) for c in text]
        outputs = []
        for i in range(p.n):
            rng = random.Random(f"{text}|{p.seed + i}")
            length = rng.randint(1, p.max_tokens)
            finish = "length" if length == p.max_tokens else "stop"
            ids = [rng.randint(1, 999) for _ in range(length)]
            if finish == "stop" and rng.random() < 0.5:
                ids[-1] = EOS_ID  # the other half of the stops need EOS repair
            answer = rng.choice([r"\boxed{70}", r"\boxed{71}", "70", "no answer"])
            outputs.append(FakeCompletion(f"sample {i}: {answer}", ids, finish))
        return FakeRequestOutput(prompt_ids, outputs)


def _fields(p) -> tuple:
    return (p.n, p.temperature, p.top_p, p.top_k, p.presence_penalty, p.max_tokens, p.seed, p.extra_args)


# --- fixtures -----------------------------------------------------------------

CONFIG = {
    "model": {"student_id": "/nonexistent/fake-student", "teacher_id": "x", "enable_thinking": True},
    "data": {"dataset": "openthoughts", "split": None, "pool_seed": 42},
    "rollout": {"num_prompts": 5, "num_rollouts": 2},
    "sampling": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0,
        "fast_presence_penalty": False, "max_new_tokens": 6,
    },
    # budget 8: eval chunks of 4 problems (x2), rollout chunks of 4 prompts (x2)
    "engine": {"gpu_memory_utilization": 0.9, "max_model_len": 64, "target_concurrent_sequences": 8},
    "eval": {
        "dataset": "math500", "num_problems": 7, "num_samples": 2,
        "intermediate_num_problems": 7, "eval_seed_offset": EVAL_SEED_OFFSET,
    },
    "eval_sets": {
        "tiny": {
            "dataset": "tiny", "num_problems": 3, "num_samples": 2,
            "intermediate_num_problems": 3, "eval_seed_offset": 200000,
        }
    },
    "resume": True,
    "seed": 42,
    "num_gpus": 2,
}


def make_run_dir(root: Path) -> Path:
    run_dir = root / "run"
    (run_dir / "pool").mkdir(parents=True)
    OmegaConf.save(config=OmegaConf.create(CONFIG), f=run_dir / "resolved_config.yaml")
    with (run_dir / "pool" / "prompts.jsonl").open("w") as f:
        for i in range(10):
            f.write(json.dumps({
                "example_index": i, "id": f"p{i}", "prompt": f"Compute problem {i}.",
                "reference": "70", "round": i // 5,
            }) + "\n")
    for name, count in (("eval_problems.jsonl", 7), ("eval_problems_tiny.jsonl", 3)):
        with (run_dir / "pool" / name).open("w") as f:
            for i in range(count):
                f.write(json.dumps({"id": f"{name}:{i}", "prompt": f"Eval {name} {i}?", "answer": "70"}) + "\n")
    return run_dir


def load_old_modules(tmp: Path):
    """The pre-merge stage with the pre-merge generator wired in."""

    tmp.mkdir(parents=True, exist_ok=True)
    modules = {}
    for name, path in (("old_generate_vllm", "apod/models/generate_vllm.py"),
                       ("old_rollout_eval", "apod/stages/rollout_eval.py")):
        source = subprocess.run(
            ["git", "show", f"{OLD_REV}:{path}"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout
        target = tmp / f"{name}.py"
        target.write_text(source)
        spec = importlib.util.spec_from_file_location(name, target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    old_gv, old_re = modules["old_generate_vllm"], modules["old_rollout_eval"]
    old_re.generate_trajectories_vllm = old_gv.generate_trajectories_vllm
    old_re.build_sampling_params = old_gv.build_sampling_params
    old_re.render_prompt = old_gv.render_prompt
    return old_re


class Harness:
    def __init__(self, stage_module, label: str, root: Path):
        self.stage = stage_module
        self.label = label
        self.run_dir = make_run_dir(root / label)
        self.calls: list = []
        stage_module.build_llm = self.build_llm

    def build_llm(self, model_path, **kwargs):
        round_dir = self.run_dir / "arms" / "a" / "rounds" / "round_00"
        return FakeLLM(self.calls, {
            "eval": round_dir / "eval" / "done.shard0",
            "rollouts": round_dir / "rollouts" / "done.shard0",
        })

    def run(self, *extra: str, shard: int = 0, num_shards: int = 1) -> list:
        self.calls.clear()
        argv = ["--run-dir", str(self.run_dir), "--arm", "a", "--round", "0",
                "--shard", str(shard), "--num-shards", str(num_shards), *extra]
        assert self.stage.main(argv) == 0
        return list(self.calls)

    def round_dir(self) -> Path:
        return self.run_dir / "arms" / "a" / "rounds" / "round_00"


# --- comparison ---------------------------------------------------------------


def flat_requests(calls: list) -> list:
    return [request for call in calls for request in call["requests"]]


def assert_same_artifacts(old: Path, new: Path) -> None:
    old_files = sorted(p.relative_to(old) for p in old.rglob("*") if p.is_file())
    new_files = sorted(p.relative_to(new) for p in new.rglob("*") if p.is_file())
    assert old_files == new_files, (old_files, new_files)
    for rel in old_files:
        a, b = old / rel, new / rel
        if rel.suffix == ".npz":
            # savez_compressed stamps zip entry mtimes: compare the arrays
            with np.load(a, allow_pickle=True) as x, np.load(b, allow_pickle=True) as y:
                assert sorted(x.files) == sorted(y.files), rel
                for key in x.files:
                    assert np.array_equal(x[key], y[key]), (rel, key)
        else:
            assert a.read_bytes() == b.read_bytes(), rel


def assert_same(old_calls: list, new_calls: list, old_dir: Path, new_dir: Path) -> None:
    assert flat_requests(old_calls) == flat_requests(new_calls)
    assert_same_artifacts(old_dir, new_dir)


def is_eval(request) -> bool:
    return request[1][6] >= EVAL_SEED_OFFSET  # seed field


# --- the tests ----------------------------------------------------------------


def run_all(root: Path) -> None:
    from apod.stages import rollout_eval as new_re

    old_re = load_old_modules(root / "old_src")
    old, new = Harness(old_re, "old", root), Harness(new_re, "new", root)

    # 1. full round, one shard
    old_calls, new_calls = old.run(), new.run()
    assert_same(old_calls, new_calls, old.round_dir(), new.round_dir())
    assert len(old_calls) == 4, len(old_calls)  # eval [4][3] + rollouts [4][1]
    assert len(new_calls) == 3, len(new_calls)  # eval [4] [3 + 1 rollout] [4]
    mixed = [c for c in new_calls if any(map(is_eval, c["requests"])) and not all(map(is_eval, c["requests"]))]
    assert len(mixed) == 1 and len(mixed[0]["requests"]) == 4
    assert sum(r[1][0] for r in mixed[0]["requests"]) == 8  # the chunk is full
    # rollout seeds keep the rollout-only chunking (5 prompts at chunk 4 -> base, base + 4)
    rollout_seeds = [r[1][6] for r in flat_requests(new_calls) if not is_eval(r)]
    assert rollout_seeds == [42, 42, 42, 42, 46], rollout_seeds
    # eval marker only after the last eval request is graded: absent during
    # every generate that carries an eval request, present at the end
    for call in new_calls:
        if any(map(is_eval, call["requests"])):
            assert not call["markers"]["eval"]
        assert not call["markers"]["rollouts"]
    assert (new.round_dir() / "eval" / "done.shard0").exists()
    assert (new.round_dir() / "rollouts" / "done.shard0").exists()
    rows = [json.loads(line) for line in (new.round_dir() / "rollouts" / "trajectories.shard0.jsonl").open()]
    assert len(rows) == 10 and any(r["finish_reason"] == "stop" for r in rows)

    # 2. resume mid-rollouts: keep two complete prompt groups plus a torn third
    for h in (old, new):
        path = h.round_dir() / "rollouts" / "trajectories.shard0.jsonl"
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines[:5]))
        (h.round_dir() / "rollouts" / "done.shard0").unlink()
    old_calls, new_calls = old.run(), new.run()
    assert_same(old_calls, new_calls, old.round_dir(), new.round_dir())
    assert not any(map(is_eval, flat_requests(new_calls)))  # eval rows complete: untouched
    assert len(flat_requests(new_calls)) == 3  # the torn group and the two missing prompts

    # 3. eval-only into fresh dirs, then 4. resume after eval: rollouts only
    for h in (old, new):
        shutil.rmtree(h.round_dir())
    old_calls, new_calls = old.run("--eval-only"), new.run("--eval-only")
    assert_same(old_calls, new_calls, old.round_dir(), new.round_dir())
    assert len(new_calls) == 2 and all(map(is_eval, flat_requests(new_calls)))
    assert not (new.round_dir() / "rollouts").exists()
    old_calls, new_calls = old.run(), new.run()
    assert_same(old_calls, new_calls, old.round_dir(), new.round_dir())
    assert not any(map(is_eval, flat_requests(new_calls)))
    assert len(new_calls) == 2 and len(flat_requests(new_calls)) == 5
    assert [r[1][6] for r in flat_requests(new_calls)] == [42, 42, 42, 42, 46]

    # 5. named eval set (stamped protocol) with a problem prefix, eval-only
    old_calls = old.run("--eval-only", "--eval-dataset", "tiny", "--eval-num-problems", "2")
    new_calls = new.run("--eval-only", "--eval-dataset", "tiny", "--eval-num-problems", "2")
    assert_same(old_calls, new_calls, old.round_dir(), new.round_dir())
    assert len(flat_requests(new_calls)) == 2
    assert all(r[1][6] >= 200000 for r in flat_requests(new_calls))
    assert (new.round_dir() / "eval_tiny" / "done.shard0").exists()

    # 6. two shards, fresh dirs
    for h in (old, new):
        shutil.rmtree(h.round_dir())
    for shard in (0, 1):
        old_calls = old.run(shard=shard, num_shards=2)
        new_calls = new.run(shard=shard, num_shards=2)
        assert flat_requests(old_calls) == flat_requests(new_calls)
    assert_same_artifacts(old.round_dir(), new.round_dir())
    for shard in (0, 1):
        assert (new.round_dir() / "eval" / f"done.shard{shard}").exists()
        assert (new.round_dir() / "rollouts" / f"done.shard{shard}").exists()
    assert len(list((new.round_dir() / "rollouts" / "tokens").glob("*.npz"))) == 5


def test_merged_stream_matches_separate_streams():
    with tempfile.TemporaryDirectory() as tmp:
        run_all(Path(tmp))


if __name__ == "__main__":
    test_merged_stream_matches_separate_streams()
    print("tests/test_rollout_eval_merged.py passed")
