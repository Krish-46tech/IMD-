"""Pure-Python inference for the trained TF-IDF logistic ProcessBench model."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"(?u)\b\w\w+\b")


class PonrModel:
    def __init__(self, model_path: Path):
        data = json.loads(model_path.read_text(encoding="utf-8"))
        self.vocabulary: dict[str, int] = data["vocabulary"]
        self.idf: list[float] = data["idf"]
        self.coefficients: list[float] = data["coefficients"]
        self.intercept: float = data["intercept"]
        self.selected_threshold: float = data["selected_threshold"]
        self.analyzer: str = data["feature_config"]["analyzer"]
        self.ngram_range: tuple[int, int] = tuple(data["feature_config"]["ngram_range"])

    def risk(self, text: str) -> float:
        lowered = text.lower()
        counts = Counter()
        if self.analyzer == "word":
            tokens = TOKEN.findall(lowered)
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                counts.update(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
        elif self.analyzer == "char_wb":
            for token in lowered.split():
                padded = f" {token} "
                for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                    counts.update(padded[i:i + n] for i in range(len(padded) - n + 1))
        else:
            raise ValueError(f"Unsupported trained-model analyzer: {self.analyzer}")
        values = {}
        for term, count in counts.items():
            index = self.vocabulary.get(term)
            if index is not None:
                values[index] = (1.0 + math.log(count)) * self.idf[index]
        norm = math.sqrt(sum(value * value for value in values.values()))
        logit = self.intercept
        if norm:
            logit += sum(self.coefficients[index] * value / norm for index, value in values.items())
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
