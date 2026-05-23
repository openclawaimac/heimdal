"""Manual file-fed teacher provider.

Reads teacher answers the operator wrote by hand from
``storage/mirror/teacher_outputs/<case_id>.txt`` (or ``.md``). Useful for:

- testing the diff engine with a known frontier answer copied from a
  chat session,
- mirror-mode dry-runs where the operator wants to control the teacher
  output rather than call a real cloud API.

If no manual file exists for a case, ``generate`` returns ``status="skipped"``.
"""

from __future__ import annotations

import os

from heimdal.mirror.provider import TeacherInput, TeacherProvider, TeacherResult

MANUAL_MODEL = "heimdal-mirror-manual"


class ManualTeacher(TeacherProvider):
    name = "manual"

    def __init__(self, teacher_outputs_dir: str, model: str = MANUAL_MODEL):
        self.teacher_outputs_dir = teacher_outputs_dir
        self.model = model

    def _candidate_paths(self, case_id: str) -> list[str]:
        return [
            os.path.join(self.teacher_outputs_dir, f"{case_id}.md"),
            os.path.join(self.teacher_outputs_dir, f"{case_id}.txt"),
        ]

    def generate(self, input_: TeacherInput) -> TeacherResult:
        for path in self._candidate_paths(input_.case_id):
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    output = fh.read()
            except OSError:
                continue
            return TeacherResult(
                provider=self.name,
                model=self.model,
                status="pass",
                output=output,
                usage={
                    "input_tokens": len((input_.local_output or "").split()),
                    "output_tokens": len(output.split()),
                    "estimated_cost": 0.0,
                },
                metadata={"source_path": os.path.basename(path)},
            )
        return TeacherResult(
            provider=self.name,
            model=self.model,
            status="skipped",
            output="",
            usage={"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0},
            metadata={"reason": f"no manual teacher output for case {input_.case_id!r}"},
        )
