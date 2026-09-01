"""The question bank on CPU: labels, scheduling, and the build loop on a fake engine.

Pure parts: the strict grade -> C/W/M label -> bucket mapping on hand-built
grade patterns (truncation rule included) and ``next_step``'s narrow teacher
scheduling and stopping rule. The build loop runs ``apod.bank.main`` end to
end with the deterministic FakeLLM of tests/test_rollout_eval_merged.py
(vLLM importable, no GPU), the workers launched in-process instead of as
subprocesses, and a fake entropy scorer: layout, sharding, resume (a rerun
issues no requests; a torn shard file regenerates one question), extending
student_questions, the override guard, and ``--report``.

Runs under pytest or as ``python -m tests.test_bank``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from apod import bank  # noqa: E402
from tests.test_rollout_eval_merged import FakeLLM  # noqa: E402

THRESHOLDS = {"correct_min": 3, "wrong_max": 1}


# --- labels -------------------------------------------------------------------


def _row(correct=True, boxed=True, truncated=False):
    return {"correct": correct, "has_boxed": boxed, "truncated": truncated}


def test_strict_correct():
    assert bank.strict_correct(_row())
    assert not bank.strict_correct(_row(correct=False))
    assert not bank.strict_correct(_row(boxed=False))  # Math-Verify fallback on a bare number: not strict
    assert not bank.strict_correct(_row(truncated=True))  # cap-hit is wrong even with a boxed answer


def test_label_thresholds():
    assert bank.label([True] * 4, **THRESHOLDS) == "C"
    assert bank.label([True, True, True, False], **THRESHOLDS) == "C"
    assert bank.label([True, True, False, False], **THRESHOLDS) == "M"
    assert bank.label([True, False, False, False], **THRESHOLDS) == "W"
    assert bank.label([False] * 4, **THRESHOLDS) == "W"


def test_bucket_of():
    assert bank.bucket_of("W", "C") == "teacher_right_student_wrong"
    assert bank.bucket_of("C", "C") == "both_right"
    assert bank.bucket_of("W", "W") == "both_wrong"
    assert bank.bucket_of("C", "W") == "mixed"  # teacher-wrong / student-correct cell
    assert bank.bucket_of("C", "M") == "mixed"
    assert bank.bucket_of("W", "M") == "mixed"
    assert bank.bucket_of("M", None) == "mixed"  # student-M: mixed without a teacher sweep
    assert bank.bucket_of("M", "C") == "mixed"
    assert bank.bucket_of("C", None) == "unlabelled"
    assert bank.bucket_of("W", None) == "unlabelled"


# --- scheduling ---------------------------------------------------------------


def _cfg(**bank_overrides):
    values = {"chunk_questions": 4, "student_questions": 8, "target_per_bucket": 2, "num_rollouts": 4}
    values.update(bank_overrides)
    return SimpleNamespace(bank=SimpleNamespace(**values))


def _q(index, student, teacher=None, entropy=1.0):
    return {
        "example_index": index, "student_label": student, "teacher_label": teacher,
        "bucket": bank.bucket_of(student, teacher), "question_entropy": entropy,
    }


def test_next_step_student_then_entropy_then_teacher():
    cfg = _cfg()
    assert bank.next_step([], 100, [], cfg) == ("student", 0, [0, 1, 2, 3])
    rows = [_q(i, "W") for i in range(4)]
    assert bank.next_step(rows, 100, [], cfg) == ("student", 1, [4, 5, 6, 7])
    rows += [_q(i, "W", entropy=None) for i in range(4, 8)]
    assert bank.next_step(rows, 100, [], cfg) == ("entropy", 1, [4, 5, 6, 7])
    rows = [_q(i, "W") for i in range(8)]
    assert bank.next_step(rows, 100, [], cfg) == ("teacher", 0, [0, 1, 2, 3])
    # the student limit is clamped to the pool and the last chunk may be short
    assert bank.next_step([], 6, [], cfg) == ("student", 0, [0, 1, 2, 3])
    assert bank.next_step([_q(i, "W") for i in range(4)], 6, [], cfg) == ("student", 1, [4, 5])


def test_next_step_teacher_eligibility_and_stop():
    cfg = _cfg(student_questions=12, chunk_questions=12)
    # student-M never swept; C and W taken in pool order, chunk_questions at a time
    rows = [_q(0, "M"), _q(1, "C"), _q(2, "W"), _q(3, "M"), _q(4, "W")]
    rows += [_q(i, "W") for i in range(5, 12)]
    step = bank.next_step(rows, 12, [], cfg)
    assert step[:2] == ("teacher", 0) and step[2][:3] == [1, 2, 4] and len(step[2]) == 10
    # a persisted plan that is not complete is rerun before any new plan
    assert bank.next_step(rows, 12, [[2, 4]], cfg) == ("teacher", 0, [2, 4])
    rows[2]["teacher_label"], rows[4]["teacher_label"] = "C", "C"
    for i in (2, 4):
        rows[i]["bucket"] = bank.bucket_of("W", "C")
    assert bank.next_step(rows, 12, [[2, 4]], cfg)[1] == 1  # plan 0 complete: next chunk
    # TC/SW full (2): W questions stay eligible while both_wrong / mixed are short
    step = bank.next_step(rows, 12, [[2, 4]], cfg)
    assert 5 in step[2] and 1 in step[2]
    # fill both_wrong and mixed from W questions: no W is eligible any more, C still is (both_right)
    for i, t in ((5, "W"), (6, "W"), (7, "M"), (8, "M")):
        rows[i]["teacher_label"], rows[i]["bucket"] = t, bank.bucket_of("W", t)
    step = bank.next_step(rows, 12, [[2, 4]], cfg)
    assert step[2] == [1]
    # both_right full too (2 C questions with teacher C): nothing eligible -> done
    rows[1]["teacher_label"], rows[1]["bucket"] = "C", "both_right"
    rows.append(_q(12, "C", "C"))
    assert bank.next_step(rows, 13, [[2, 4]], cfg) is None
    # every bucket full stops even with eligible questions left
    rows.append(_q(13, "W"))
    assert bank.next_step(rows, 14, [[2, 4]], cfg) is None
    # both_right short with no C left and W buckets full: the pool is exhausted for the teacher
    rows[12]["teacher_label"], rows[12]["bucket"] = "W", "mixed"
    assert bank.next_step(rows, 14, [[2, 4]], cfg) is None


# --- the build loop on a fake engine ------------------------------------------

CONFIG = {
    "model": {"student_id": "/nonexistent/fake-student", "teacher_id": "/nonexistent/fake-teacher",
              "enable_thinking": True},
    "data": {"dataset": "openthoughts", "split": None, "pool_seed": 42},
    "rollout": {"num_prompts": 8, "num_rollouts": 4},
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0,
                 "fast_presence_penalty": False, "max_new_tokens": 6},
    "engine": {"gpu_memory_utilization": 0.9, "max_model_len": 64, "target_concurrent_sequences": 8},
    "selection": {"logit_chunk": 4},
    "bank": {"name": "bank-test", "num_rollouts": 4, "max_new_tokens": 6, "correct_min": 3, "wrong_max": 1,
             "chunk_questions": 8, "student_questions": 16, "target_per_bucket": 2,
             "student_trajectories_per_min": 118, "teacher_slowdown": 2.5, "throughput_gpus": 2},
    "resume": True,
    "seed": 42,
    "num_gpus": 2,
}
POOL_SIZE = 24


class Harness:
    """apod.bank with in-process workers, a FakeLLM per engine build, a counting entropy scorer."""

    def __init__(self, root: Path):
        self.bank_dir = root / "bank-test"
        self.bank_dir.mkdir(parents=True)
        OmegaConf.save(config=OmegaConf.create(CONFIG), f=self.bank_dir / "resolved_config.yaml")
        self.calls: list = []
        self.scored = 0
        self.engines: list[str] = []
        bank.build_llm = self.build_llm
        bank.load_entropy_scorer = self.load_scorer
        bank.pool_examples = self.pool_examples
        bank.launch_shards = self.launch_shards

    def build_llm(self, model_path, **kwargs):
        self.engines.append(model_path)
        return FakeLLM(self.calls, {})

    def load_scorer(self, model_path, logit_chunk):
        def score(ids, prompt_length, response_length):
            self.scored += 1
            return {"entropy": 0.5 + response_length / 10, "mean_logprob": -1.0, "scored_tokens": response_length}
        return score

    @staticmethod
    def pool_examples(cfg):
        return [{"id": f"openthoughts:{i}", "prompt": f"Compute problem {i}.", "answer": "70"} for i in range(POOL_SIZE)]

    def launch_shards(self, bank_dir, worker_argv, gpus):
        for shard in range(len(gpus)):
            argv = ["--bank-dir", str(bank_dir), *worker_argv, "--shard", str(shard), "--num-shards", str(len(gpus))]
            assert bank.main(argv) == 0

    def run(self, *argv: str) -> None:
        self.calls.clear()
        self.engines.clear()
        self.scored = 0
        assert bank.main(["--bank-dir", str(self.bank_dir), *argv]) == 0

    def rows(self) -> list[dict]:
        return bank.load_bank(self.bank_dir)


def requested_texts(calls: list) -> list[str]:
    return [text for call in calls for text, _ in call["requests"]]


def run_build(root: Path) -> None:
    h = Harness(root)
    h.run("--gpus", "0,1")
    rows = h.rows()
    cfg = OmegaConf.load(h.bank_dir / "resolved_config.yaml")

    # student: student_questions = 16 of a 24-question pool, in two chunks, 4 rollouts each
    assert [r["example_index"] for r in rows] == list(range(16))
    assert all(len(r["student_grades"]) == 4 and len(r["student_lengths"]) == 4 for r in rows)
    assert all(r["question_entropy"] is not None for r in rows)
    assert {r["chunk"] for r in rows} == {0, 1}
    assert len(list((h.bank_dir / "student" / "tokens").glob("example_*.npz"))) == 16
    for chunk in (0, 1):
        for shard in (0, 1):
            assert (h.bank_dir / "student" / f"done.chunk{chunk:03d}.shard{shard}").exists()
            assert (h.bank_dir / "student" / "entropy" / f"done.chunk{chunk:03d}.shard{shard}").exists()
    # sharding by example_index % 2, in both models
    for model in ("student", "teacher"):
        for shard in (0, 1):
            indices = {r["example_index"] for r in bank.iter_jsonl(h.bank_dir / model / f"trajectories.shard{shard}.jsonl")}
            assert indices and all(i % 2 == shard for i in indices), (model, shard, indices)
    # labels and buckets are consistent with the raw rows
    groups = bank.complete_groups(h.bank_dir / "student", 4, bank.ROLLOUT_FIELDS)
    for r in rows:
        grades = [bank.strict_correct(x) for x in groups[r["example_index"]]]
        assert r["student_grades"] == grades
        assert r["student_label"] == bank.label(grades, **THRESHOLDS)
        assert r["bucket"] == bank.bucket_of(r["student_label"], r["teacher_label"])
    labels = Counter(r["student_label"] for r in rows)
    assert labels["W"] > 0 and labels["M"] > 0, labels  # the fake grades cover both
    # H(q) = mean of the 4 trajectory entropies from the fake scorer
    ent = bank.entropy_by_question(h.bank_dir / "student" / "entropy")
    for r in rows:
        assert abs(r["question_entropy"] - sum(ent[r["example_index"]].values()) / 4) < 1e-9
    # narrow teacher sweep: every C/W question swept (target 2 is not reached for both_right
    # or is, either way no student-M question ever is), student-M never
    teacher_swept = {r["example_index"] for r in rows if r["teacher_label"] is not None}
    assert all(r["example_index"] not in teacher_swept for r in rows if r["student_label"] == "M")
    counts = bank.bucket_counts(rows)
    full = {b: counts[b] >= 2 for b in bank.BUCKETS[:4]}
    for r in rows:
        if r["student_label"] in ("C", "W") and any(not full[b] for b in bank.REACHABLE[r["student_label"]]):
            assert r["example_index"] in teacher_swept, r
    assert counts["unlabelled"] == 0 or all(full[b] for b in bank.BUCKETS[:4])
    plans = bank.teacher_plans(h.bank_dir)
    assert plans and all(len(p) <= 8 for p in plans)
    assert sorted(i for p in plans for i in p) == sorted(teacher_swept)
    # engines: student before teacher, one build per (step, shard)
    assert h.engines[:4] == ["/nonexistent/fake-student"] * 4
    assert set(h.engines[4:]) == {"/nonexistent/fake-teacher"}
    # the teacher's requests: 4 samples each, only the planned questions
    teacher_texts = [t for c in h.calls for t, p in c["requests"] if p[0] == 4]
    assert len(teacher_texts) == 16 + len(teacher_swept)  # student 16 + teacher

    # rerun = no new requests, no scoring, byte-identical bank
    before = (h.bank_dir / "questions.jsonl").read_bytes()
    h.run("--gpus", "0,1")
    assert h.calls == [] and h.scored == 0 and h.engines == []
    assert (h.bank_dir / "questions.jsonl").read_bytes() == before

    # resume: a crash mid-sweep leaves a student shard file with a complete group lost
    # and a torn one (entropy for that chunk has not run yet: the loop scores a chunk
    # only once its sweep is complete). Exactly those questions are re-requested and
    # scored; nothing else is.
    path = h.bank_dir / "student" / "trajectories.shard0.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    lost = {json.loads(l)["example_index"] for l in lines[-6:]}
    path.write_text("".join(lines[:-6]) + '{"example_index": 999, "rollout_in')
    for epath in (h.bank_dir / "student" / "entropy").glob("entropy.shard*.jsonl"):
        kept = [r for r in bank.iter_jsonl(epath) if r["example_index"] not in lost]
        epath.write_text("".join(json.dumps(r) + "\n" for r in kept))
    h.run("--gpus", "0,1")
    assert sorted(requested_texts(h.calls)) == sorted(f"<user>Compute problem {i}.<assistant><think>" for i in lost)
    assert h.scored == 4 * len(lost)  # the regenerated rollouts are rescored
    assert h.engines == ["/nonexistent/fake-student"]
    assert [r["example_index"] for r in h.rows()] == list(range(16))
    assert bank.complete_indices(path, 4) >= lost

    # a torn entropy shard resumes too: one scored row lost -> one rescored
    epath = h.bank_dir / "student" / "entropy" / "entropy.shard1.jsonl"
    elines = epath.read_text().splitlines(keepends=True)
    epath.write_text("".join(elines[:-1]) + '{"example_index": 1, "rollout_index": 3, "ent')
    h.run("--gpus", "0,1")
    assert h.calls == [] and h.scored == 1

    # extending the sweep: a mutable override adds the third chunk; the regime is frozen
    h.run("bank.student_questions=24", "--gpus", "0,1")
    rows = h.rows()
    assert [r["example_index"] for r in rows] == list(range(24))
    assert OmegaConf.load(h.bank_dir / "resolved_config.yaml").bank.student_questions == 24
    student_texts = [t for c in h.calls for t, p in c["requests"]]
    assert all(any(f"problem {i}." in t for i in range(16, 24)) for t in student_texts[:8])
    try:
        h.run("bank.max_new_tokens=8192")
    except SystemExit as exc:
        assert "bank.max_new_tokens" in str(exc)
    else:
        raise AssertionError("regime override accepted on an existing bank")

    # a GPU-count change is safe: one shard sees the other shard's complete groups
    h.run("--gpus", "0")
    assert h.calls == [] and h.scored == 0

    # --report relabels and prints every bucket plus the cost lines
    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out):
        h.run("--report")
    text = out.getvalue()
    for b in bank.BUCKETS:
        assert b in text
    assert "remaining generation" in text and "teacher:" in text and "student:" in text
    est = bank.remaining_generation(rows, POOL_SIZE, cfg)
    assert est["student_questions"] == 0 and est["student_hours"] == 0.0
    # rate scaling: 2 GPUs at the measured 118/min; a 1-GPU config halves it
    assert abs(est["trajectories_per_min"] - 118.0) < 1e-9
    # bucket helpers
    assert sum(len(bank.bucket_questions(rows, b)) for b in bank.BUCKETS) == len(rows)


def test_bank_build_on_fake_engine():
    with tempfile.TemporaryDirectory() as tmp:
        run_build(Path(tmp))


if __name__ == "__main__":
    test_strict_correct()
    test_label_thresholds()
    test_bucket_of()
    test_next_step_student_then_entropy_then_teacher()
    test_next_step_teacher_eligibility_and_stop()
    test_bank_build_on_fake_engine()
    print("tests/test_bank.py passed")
