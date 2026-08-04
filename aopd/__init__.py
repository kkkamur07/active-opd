"""Active on-policy distillation research prototype."""

from .models.student import StudentModel
from .models.teacher import TeacherModel

__all__ = ["StudentModel", "TeacherModel"]
