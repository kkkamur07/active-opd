from .generate_vllm import build_llm, generate_trajectories_vllm, render_prompt
from .load import load_lm
from .student import generate_trajectories
from .teacher import generate_teacher, teacher_logits

STUDENT_ID = "Qwen/Qwen3.5-2B"
TEACHER_ID = "Qwen/Qwen3.5-9B"


def load_student(model_id: str = STUDENT_ID, device_map: str = "auto"):
    return load_lm(model_id, frozen=False, device_map=device_map)


def load_teacher(model_id: str = TEACHER_ID, device_map: str = "auto"):
    return load_lm(model_id, frozen=True, device_map=device_map)
