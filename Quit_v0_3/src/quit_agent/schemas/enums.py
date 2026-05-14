from __future__ import annotations

from enum import StrEnum


class WorkflowState(StrEnum):
    PLAN = "PLAN"
    VALIDATE_BRIEF = "VALIDATE_BRIEF"
    RETRIEVE = "RETRIEVE"
    READ = "READ"
    IDEATE = "IDEATE"
    IDEA_EVAL = "IDEA_EVAL"
    BUILD_SPEC = "BUILD_SPEC"
    CODE = "CODE"
    CODE_EVAL = "CODE_EVAL"
    CODE_LLM_EVAL = "CODE_LLM_EVAL"
    CODE_REVISE = "CODE_REVISE"
    WRITE = "WRITE"
    WRITE_EVAL = "WRITE_EVAL"
    WRITE_LLM_EVAL = "WRITE_LLM_EVAL"
    WRITE_REVISE = "WRITE_REVISE"
    WRITE_FINAL_EVAL = "WRITE_FINAL_EVAL"
    EXTRACT = "EXTRACT"
    STOP = "STOP"


class ValidationStatus(StrEnum):
    PASS = "PASS"
    REPAIR = "REPAIR"
    FAIL = "FAIL"


class IdeaDecisionType(StrEnum):
    PASS = "PASS"
    REVISE = "REVISE"
    REJECT = "REJECT"


class FallbackTarget(StrEnum):
    PLAN = "PLAN"
    RETRIEVE = "RETRIEVE"
    READ = "READ"
    IDEATE = "IDEATE"
    BUILD_SPEC = "BUILD_SPEC"
    CODE = "CODE"
    WRITE = "WRITE"
    STOP = "STOP"


class PaperStatus(StrEnum):
    FOUND = "found"
    DOWNLOADED = "downloaded"
    FAILED = "failed"
    READ = "read"
