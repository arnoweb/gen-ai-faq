"""Minimal ONNX Runtime inference for the FAQ models — no torch / transformers / sentence-transformers.

Why this module exists: on Streamlit Community Cloud (~1 GB RAM), going through
sentence-transformers' ONNX backend cost far more memory than the models themselves:
importing torch + transformers + optimum is ~550 MB before any model is loaded, and the
"fast" (Rust) XLM-R tokenizer takes ~285 MB per model for its 250k-token vocabulary.
Here the same ONNX files run directly on onnxruntime, tokenized with the original
SentencePiece model (~55 MB), which brings the whole app well under the limit.

Each ONNX file is also re-saved once (on first start) with its weights in a separate
"external data" file, which onnxruntime memory-maps instead of copying: weights then live
in reclaimable page cache rather than private memory, only the pages actually used (e.g. the
embedding rows of tokens seen) are read, and the ~2x load-time spike of a plain .onnx file
disappears. This is what lets the compare app keep two encoders loaded at once.

Outputs match sentence-transformers exactly: same token ids as XLMRobertaTokenizer, same
mean pooling for the bi-encoders, same raw logits for the cross-encoder.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download


class XLMRobertaTokenizer:
    """SentencePiece-based tokenizer producing the same ids as transformers' XLMRobertaTokenizer.

    Every model used here (fine-tuned XLM-R, its paraphrase-multilingual-mpnet base and the
    mMiniLMv2 cross-encoder) shares the XLM-R vocabulary, so one instance can serve them all.
    """

    BOS, PAD, EOS, UNK = 0, 1, 2, 3
    FAIRSEQ_OFFSET = 1  # XLM-R ids = SentencePiece ids + 1 (fairseq layout), spm 0 (<unk>) -> 3

    def __init__(self, sentencepiece_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=sentencepiece_path)

    def _ids(self, text: str) -> list[int]:
        return [i + self.FAIRSEQ_OFFSET if i else self.UNK for i in self.sp.encode(text)]

    def encode(self, text: str, max_length: int) -> list[int]:
        return [self.BOS] + self._ids(text)[: max_length - 2] + [self.EOS]

    def encode_pair(self, text_a: str, text_b: str, max_length: int) -> list[int]:
        a, b = self._ids(text_a), self._ids(text_b)
        # "longest_first" truncation, as in transformers: trim the longer sequence one token at a time.
        while len(a) + len(b) > max_length - 4:
            if len(a) > len(b):
                a.pop()
            else:
                b.pop()
        return [self.BOS] + a + [self.EOS, self.EOS] + b + [self.EOS]

    def pad(self, sequences: list[list[int]]) -> dict[str, np.ndarray]:
        width = max(len(s) for s in sequences)
        input_ids = np.full((len(sequences), width), self.PAD, dtype=np.int64)
        attention_mask = np.zeros((len(sequences), width), dtype=np.int64)
        for row, seq in enumerate(sequences):
            input_ids[row, : len(seq)] = seq
            attention_mask[row, : len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_tokenizer(repo_id: str) -> XLMRobertaTokenizer:
    return XLMRobertaTokenizer(hf_hub_download(repo_id, "sentencepiece.bpe.model"))


CACHE_DIR = Path(os.getenv("FAQ_ONNX_CACHE", Path.home() / ".cache" / "faq-onnx"))
EXTERNAL_DATA_NAME = "model.onnx.data"


def _convert_to_external_data(source: str, target_dir: str) -> None:
    import onnx  # only needed for this one-off conversion, run in a child process

    model = onnx.load(source)
    onnx.save_model(
        model,
        os.path.join(target_dir, "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=EXTERNAL_DATA_NAME,
        size_threshold=1024,
    )


def _external_data_model(repo_id: str, onnx_file: str) -> str:
    """Path of `onnx_file` from `repo_id` re-saved with external (memory-mappable) weights."""
    target_dir = CACHE_DIR / repo_id.replace("/", "--") / Path(onnx_file).stem
    target = target_dir / "model.onnx"
    if not (target.exists() and (target_dir / EXTERNAL_DATA_NAME).exists()):
        source = hf_hub_download(repo_id, onnx_file)
        tmp_dir = target_dir.with_name(f"{target_dir.name}.tmp{os.getpid()}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        # Child process: loading the protobuf costs ~2x the file size in RAM, and running it
        # in a subprocess guarantees all of that is returned to the OS before the app goes on.
        subprocess.run([sys.executable, __file__, source, str(tmp_dir)], check=True)
        shutil.rmtree(target_dir, ignore_errors=True)
        tmp_dir.rename(target_dir)
        # The downloaded file is no longer needed once converted: delete it so the
        # container doesn't keep a second copy of the weights in page cache.
        _drop_page_cache(source)
        _remove_hub_file(source)
    for path in target_dir.iterdir():
        _drop_page_cache(path)
    return str(target)


def _drop_page_cache(path) -> None:
    """Evict a file from the page cache.

    Streamlit Community Cloud counts the container's whole memory, page cache included,
    against its ~1 GB limit: downloading and converting the models leaves ~1 GB of
    file pages behind even though the app itself uses ~250 MB. Pages of the memory-mapped
    weights are read back on demand afterwards.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def _remove_hub_file(path: str) -> None:
    """Delete a hf_hub_download file (a symlink to a blob in the Hub cache) and its blob."""
    try:
        blob = os.path.realpath(path)
        os.remove(path)
        if blob != path:
            os.remove(blob)
    except OSError:
        pass


def _session(repo_id: str, onnx_file: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # Prepacking would copy the memory-mapped weights into private memory, for no measurable
    # speed-up on short FAQ queries.
    options.add_session_config_entry("session.disable_prepacking", "1")
    return ort.InferenceSession(
        _external_data_model(repo_id, onnx_file), options, providers=["CPUExecutionProvider"]
    )


class OnnxSentenceEncoder:
    """Bi-encoder: mean-pooled last_hidden_state (the models' 1_Pooling config), like SentenceTransformer.encode."""

    def __init__(self, repo_id: str, onnx_file: str, tokenizer: XLMRobertaTokenizer, batch_size: int = 32):
        self.session = _session(repo_id, onnx_file)
        self.tokenizer = tokenizer
        self.max_length = json.load(open(hf_hub_download(repo_id, "sentence_bert_config.json")))["max_seq_length"]
        self.batch_size = batch_size

    def encode(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        # Same batching as sentence-transformers (longest texts first, then restored order):
        # with dynamically-quantized int8 weights, activations are quantized per batch, so
        # batch composition slightly affects the output — this keeps it identical.
        order = np.argsort([-len(t) for t in texts])
        sorted_texts = [texts[i] for i in order]
        chunks = []
        for start in range(0, len(sorted_texts), self.batch_size):
            chunk = sorted_texts[start : start + self.batch_size]
            batch = self.tokenizer.pad([self.tokenizer.encode(t, self.max_length) for t in chunk])
            hidden = self.session.run(["last_hidden_state"], batch)[0]
            mask = batch["attention_mask"][..., None].astype(np.float32)
            chunks.append((hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None))
        embeddings = np.empty((len(texts), chunks[0].shape[1]), dtype=np.float32)
        embeddings[order] = np.concatenate(chunks)
        return embeddings[0] if single else embeddings


class OnnxCrossEncoder:
    """Cross-encoder: raw relevance logit per (query, passage) pair, like CrossEncoder.predict."""

    def __init__(self, repo_id: str, onnx_file: str, tokenizer: XLMRobertaTokenizer, max_length: int = 512):
        self.session = _session(repo_id, onnx_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def predict(self, pairs) -> np.ndarray:
        batch = self.tokenizer.pad([self.tokenizer.encode_pair(a, b, self.max_length) for a, b in pairs])
        return self.session.run(["logits"], batch)[0][:, 0]


def cos_sim(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query embedding against each row of `embeddings`."""
    q = query_embedding / np.linalg.norm(query_embedding)
    m = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return m @ q


if __name__ == "__main__":
    _convert_to_external_data(sys.argv[1], sys.argv[2])
