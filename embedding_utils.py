from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# Detect whether CUDA is available so the embedder can choose a sane default device.
def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# Normalize embedding vectors so FAISS inner-product search behaves like cosine similarity.
def l2_normalize(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms = np.sqrt((x * x).sum(axis=1, keepdims=True))
    norms = np.maximum(norms, eps)
    return x / norms


# Load a local sentence-transformers model directory for embedding generation.
def load_sentence_embedder(model_path: str, device: str | None = None):
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed in the current environment."
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    resolved_path = Path(str(model_path or "").strip())
    if not resolved_path.is_dir():
        raise FileNotFoundError(
            f"Embedding model directory does not exist: {resolved_path}"
        )

    resolved_device = device or ("cuda" if torch_cuda_available() else "cpu")
    return SentenceTransformer(str(resolved_path), device=resolved_device)


# 只加载当前 embedding 模型的 tokenizer，供对话切片使用，避免为切片提前加载完整权重。
def load_embedding_tokenizer(model_path: str):
    if AutoTokenizer is None:
        raise RuntimeError("transformers is not installed in the current environment.")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    resolved_path = Path(str(model_path or "").strip())
    if not resolved_path.is_dir():
        raise FileNotFoundError(
            f"Embedding model directory does not exist: {resolved_path}"
        )
    return AutoTokenizer.from_pretrained(str(resolved_path), local_files_only=True)
