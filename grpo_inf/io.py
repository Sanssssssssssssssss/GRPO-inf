from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "dev", "test_locked", "test_redteam")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class DatasetReader:
    def __init__(self, dataset_path: str | Path):
        self.path = Path(dataset_path)
        self._zip: zipfile.ZipFile | None = None
        if self.path.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.path)
            self.names = [name for name in self._zip.namelist() if not name.endswith("/")]
        else:
            self.names = [str(item.relative_to(self.path)).replace("\\", "/") for item in self.path.rglob("*") if item.is_file()]

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "DatasetReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def exists(self, suffix: str) -> bool:
        return self.resolve(suffix) is not None

    def resolve(self, suffix: str) -> str | None:
        suffix = suffix.replace("\\", "/").lstrip("/")
        for name in self.names:
            if name == suffix or name.endswith("/" + suffix):
                return name
        return None

    def read_text(self, name: str) -> str:
        if self._zip is not None:
            return self._zip.read(name).decode("utf-8")
        return (self.path / name).read_text(encoding="utf-8")

    def read_jsonl_suffix(self, suffix: str) -> list[dict[str, Any]]:
        name = self.resolve(suffix)
        if not name:
            return []
        return [json.loads(line) for line in self.read_text(name).splitlines() if line.strip()]


def find_split_rows(reader: DatasetReader, split: str) -> list[dict[str, Any]]:
    candidates = [
        f"records/{split}.jsonl",
        f"grpo/prompts_{split}.jsonl",
        f"sft/reviewer_{split}.jsonl",
    ]
    for candidate in candidates:
        rows = reader.read_jsonl_suffix(candidate)
        if rows:
            return rows
    return []


def document_text(document: dict[str, Any]) -> str:
    if document.get("ocr_text"):
        return str(document["ocr_text"])
    ocr = document.get("ocr")
    if isinstance(ocr, dict):
        lines = ocr.get("lines")
        if isinstance(lines, list):
            return "\n".join(str(line.get("text", "")) for line in lines if isinstance(line, dict))
    if document.get("raw_text"):
        return str(document["raw_text"])
    if document.get("text"):
        return str(document["text"])
    return ""


def prompt_text(row: dict[str, Any]) -> str:
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    if isinstance(row.get("prompt"), list):
        return "\n".join(str(item.get("content", "")) for item in row["prompt"] if isinstance(item, dict))
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
    return str(row.get("reviewer_prompt", ""))


def oracle_from_row(row: dict[str, Any]) -> dict[str, Any]:
    oracle = row.get("oracle") or row.get("reviewer_oracle")
    return oracle if isinstance(oracle, dict) else {}
