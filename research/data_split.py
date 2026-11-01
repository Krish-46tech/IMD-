"""Deterministic problem-group split shared by model training and evaluation."""
import hashlib


def problem_key(problem: str) -> str:
    normalized = " ".join(problem.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def split_for(problem: str) -> str:
    bucket = int(problem_key(problem)[:8], 16) % 10
    return "test" if bucket < 2 else "validation" if bucket == 2 else "train"
