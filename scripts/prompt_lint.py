"""Advisory prompt checks after Törnberg (2024), sections 3 and 5.

The run is finished, so these checks never change the prompt. They document how the
archived prompt relates to the recommended structure. Every check is a deterministic
text rule; results are pass / warn / n/a.
"""
from __future__ import annotations

import re
from typing import Any

UNCERTAIN = re.compile(r"\b(uncertain|unsure|unclear|don'?t know|do not know|cannot (?:be )?determine"
                       r"|not sure|undecidable|can'?t tell)\b", re.IGNORECASE)
CONSTRAINT = re.compile(r"\b(answer with|respond with|reply with|return|output|only (?:the|one|a)|"
                        r"json|format|choose one|select one|exactly one)\b", re.IGNORECASE)
AMBIGUOUS = re.compile(r"\b(ambiguous|borderline|unclear|if (?:you are )?(?:unsure|uncertain|in doubt)|"
                       r"edge case|when in doubt)\b", re.IGNORECASE)


def _check(name: str, ok: bool | None, detail: str) -> dict:
    status = "n/a" if ok is None else ("pass" if ok else "warn")
    return {"check": name, "status": status, "detail": detail}


def lint(system: str, user_template: str, labels: list[str], label_specs: list[Any] | None,
         few_shot: list | None, rendered_roles: list[str] | None, multi_field_output: bool = False) -> list[dict]:
    text = f"{system}\n{user_template}"
    out = []
    lab_lower = [str(x).lower() for x in labels]
    listed = all(lab in text.lower() for lab in lab_lower) or "{label_list}" in text or "{codebook}" in text
    out.append(_check("options_enumerated", listed if labels else None,
                      "every label appears in the prompt" if listed else "some labels are not listed in the prompt"))
    if labels and listed and "{label_list}" not in text and "{codebook}" not in text:
        lines = text.lower().splitlines()
        one_per_line = all(any(re.search(rf"^\s*[-*\d.)]*\s*{re.escape(lab)}\b", ln) for ln in lines)
                           for lab in lab_lower)
        out.append(_check("options_one_per_line", one_per_line, "labels on separate lines"
                          if one_per_line else "labels are not each on their own line"))
    has_unc = any(UNCERTAIN.search(lab) for lab in lab_lower) or bool(UNCERTAIN.search(text))
    out.append(_check("uncertain_option", has_unc,
                      "an uncertain / don't-know option or instruction is present" if has_unc
                      else "no uncertain or don't-know option"))
    out.append(_check("output_constraint", bool(CONSTRAINT.search(text)),
                      "output format is constrained" if CONSTRAINT.search(text) else "no output-format constraint found"))
    if multi_field_output:
        js = "json" in text.lower()
        out.append(_check("json_for_multiple_fields", js, "JSON requested" if js else "several fields without JSON"))
    if label_specs is not None:
        defs = all(isinstance(s, dict) and s.get("definition") for s in label_specs)
        out.append(_check("codebook_definitions", defs, "every label has a definition" if defs
                          else "some labels lack a definition"))
        exs = all(isinstance(s, dict) and s.get("examples") for s in label_specs)
        out.append(_check("codebook_examples", exs, "every label has examples" if exs
                          else "not every label has examples (codebook should give examples per category)"))
    out.append(_check("ambiguous_case_rule", bool(AMBIGUOUS.search(text)),
                      "instructions for ambiguous cases present" if AMBIGUOUS.search(text)
                      else "no instruction for ambiguous cases"))
    if rendered_roles is not None:
        sys_ok = "system" in rendered_roles
        out.append(_check("system_instruction", sys_ok, "instructions sent as a system message" if sys_ok
                          else "no system message in the rendered inputs"))
    if few_shot:
        out.append(_check("few_shot_as_dialogue", True, f"{len(few_shot)} examples sent as user/assistant turns"))
    return out
