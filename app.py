from __future__ import annotations

import os
import pickle
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )
except ImportError:
    AutoModelForSeq2SeqLM = None
    AutoModelForSequenceClassification = None
    AutoModelForTokenClassification = None
    AutoTokenizer = None
    pipeline = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    load_model = None
    pad_sequences = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

try:
    from sqlalchemy import create_engine, text as sql_text
except ImportError:
    create_engine = None
    sql_text = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------------------------------------------------
# Paths: relative to this app.py, so it works in VS Code, EC2, or Spaces.
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CLS_ML_DIR = BASE_DIR / "Classification task" / "Classification ML & DL Model"
CLS_TRANS_DIR = BASE_DIR / "Classification task" / "classification_trans"

NER_DL_DIR = BASE_DIR / "NER Task" / "NER_DL" / "NER_DL"
NER_TRANS_DIR = BASE_DIR / "NER Task" / "NER_TRANS" / "NER_TRANS" / "NER_BERT_MODEL"

SUM_EXT_DIR = (
    BASE_DIR
    / "Summarization_task"
    / "Summarization_Extractive Baseline"
)
SUM_DL_DIR = BASE_DIR / "Summarization_task" / "summarization_dl"
SUM_TRANS_DIR = (
    BASE_DIR
    / "Summarization_task"
    / "Pretrained Transformer Summarizer"
    / "saved_models"
)

HF_CACHE_DIR = BASE_DIR / "hf_model_cache"
HF_MODEL_REPOS = {
    "hf-distilbert-news-classifier": "PUT_YOUR_USERNAME/news-classification-distilbert",
    "hf-bert-news-ner": "PUT_YOUR_USERNAME/news-ner-bert",
    "hf-bart-news-summarizer": "PUT_YOUR_USERNAME/news-summarization-bart",
}

DEVICE = 0 if torch is not None and torch.cuda.is_available() else -1


st.set_page_config(page_title="News Intelligence System", layout="wide")


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_pickle(path: Path) -> Any:
    require(path.exists(), f"Missing file: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


def load_joblib(path: Path) -> Any:
    require(joblib is not None, "joblib is not installed. Run: pip install joblib")
    require(path.exists(), f"Missing file: {path}")
    return joblib.load(path)


def decode_label(raw_label: str, label_encoder: Any | None) -> str:
    if label_encoder is None:
        return raw_label

    try:
        if raw_label.startswith("LABEL_"):
            index = int(raw_label.split("_")[-1])
        else:
            index = int(raw_label)
        return str(label_encoder.inverse_transform([index])[0])
    except Exception:
        return raw_label


def read_uploaded_text(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""
    raw = uploaded_file.read()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def split_sentences(text_value: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text_value.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def available_dirs(parent: Path, required_file: str) -> list[str]:
    if not parent.exists():
        return []
    return sorted(
        path.name
        for path in parent.iterdir()
        if path.is_dir() and (path / required_file).exists()
    )


def available_hf_models(task: str) -> list[str]:
    if task == "classification":
        names = ["hf-distilbert-news-classifier"]
    elif task == "ner":
        names = ["hf-bert-news-ner"]
    elif task == "summarization":
        names = ["hf-bart-news-summarizer"]
    else:
        names = []
    return [
        name
        for name in names
        if HF_MODEL_REPOS.get(name, "").strip()
        and "PUT_YOUR_USERNAME" not in HF_MODEL_REPOS[name]
    ]


def find_in_nested_dict(obj: Any, names: set[str]) -> Any | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in names:
                return value
        for value in obj.values():
            found = find_in_nested_dict(value, names)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------
# Optional RDS logging. Set DATABASE_URL in EC2 to enable.
# Example: postgresql+psycopg2://user:password@host:5432/dbname
# ---------------------------------------------------------------------

@st.cache_resource
def get_db_engine():
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        return None
    require(create_engine is not None, "sqlalchemy is not installed.")
    return create_engine(database_url, pool_pre_ping=True)


def log_inference(
    task_type: str,
    model_family: str,
    model_name: str,
    input_text: str,
    output_text: str,
    error_flag: bool = False,
) -> None:
    engine = get_db_engine()
    if engine is None or sql_text is None:
        return

    create_query = sql_text(
        """
        CREATE TABLE IF NOT EXISTS inference_logs (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP NOT NULL,
            task_type VARCHAR(64),
            model_family VARCHAR(64),
            model_name VARCHAR(128),
            input_length INTEGER,
            output_preview TEXT,
            error_flag BOOLEAN
        );
        """
    )

    insert_query = sql_text(
        """
        INSERT INTO inference_logs (
            created_at, task_type, model_family, model_name,
            input_length, output_preview, error_flag
        )
        VALUES (
            :created_at, :task_type, :model_family, :model_name,
            :input_length, :output_preview, :error_flag
        );
        """
    )

    with engine.begin() as connection:
        connection.execute(create_query)
        connection.execute(
            insert_query,
            {
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "task_type": task_type,
                "model_family": model_family,
                "model_name": model_name,
                "input_length": len(input_text),
                "output_preview": output_text[:1000],
                "error_flag": error_flag,
            },
        )


# ---------------------------------------------------------------------
# Classification models
# ---------------------------------------------------------------------

def predict_classification_keyword(input_text: str) -> tuple[str, float]:
    text_lower = input_text.lower()
    keyword_map = {
        "finance": [
            "stock", "market", "bank", "finance", "investment", "shares",
            "revenue", "profit", "inflation", "economy", "loan", "rbi",
        ],
        "technology": [
            "technology", "software", "ai", "artificial intelligence", "cloud",
            "microsoft", "google", "startup", "data", "cybersecurity",
        ],
        "sports": [
            "match", "football", "soccer", "cricket", "nba", "nfl", "tournament",
            "team", "player", "coach", "league",
        ],
        "politics": [
            "government", "minister", "election", "senate", "president",
            "policy", "law", "regulation", "parliament", "campaign",
        ],
        "entertainment": [
            "movie", "film", "actor", "music", "celebrity", "hollywood",
            "television", "show", "album",
        ],
    }

    scores = {
        label: sum(1 for keyword in keywords if keyword in text_lower)
        for label, keywords in keyword_map.items()
    }
    best_label = max(scores, key=scores.get)
    total = sum(scores.values())
    if scores[best_label] == 0:
        return "news", 0.50
    return best_label, max(0.50, min(0.99, scores[best_label] / max(total, 1)))


@st.cache_resource
def load_classification_ml():
    model_path = CLS_ML_DIR / "best_ml_model.pkl"
    vectorizer_path = CLS_ML_DIR / "best_text_ML_model.pkl"
    label_path = CLS_ML_DIR / "label_encoder.pkl"

    model = load_joblib(model_path)
    vectorizer_or_pipeline = load_joblib(vectorizer_path)
    label_encoder = load_joblib(label_path) if label_path.exists() else None
    return model, vectorizer_or_pipeline, label_encoder


def predict_classification_ml(input_text: str) -> tuple[str, float | None]:
    model, vectorizer_or_pipeline, label_encoder = load_classification_ml()

    if hasattr(vectorizer_or_pipeline, "predict"):
        prediction = vectorizer_or_pipeline.predict([input_text])[0]
        probabilities = (
            vectorizer_or_pipeline.predict_proba([input_text])[0]
            if hasattr(vectorizer_or_pipeline, "predict_proba")
            else None
        )
    else:
        features = vectorizer_or_pipeline.transform([input_text])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None

    label = (
        str(label_encoder.inverse_transform([prediction])[0])
        if label_encoder is not None and isinstance(prediction, (int, np.integer))
        else str(prediction)
    )
    confidence = float(np.max(probabilities)) if probabilities is not None else None
    return label, confidence


@st.cache_resource
def load_classification_dl():
    require(load_model is not None, "tensorflow is not installed. Run: pip install tensorflow")
    require(pad_sequences is not None, "tensorflow preprocessing is unavailable.")

    model_path = CLS_ML_DIR / "best_dl_model.h5"
    tokenizer_path = CLS_ML_DIR / "tokenizer.pkl"
    label_path = CLS_ML_DIR / "label_encoder.pkl"

    model = load_model(model_path, compile=False)
    tokenizer = load_joblib(tokenizer_path)
    label_encoder = load_joblib(label_path) if label_path.exists() else None
    max_len = int(model.input_shape[1]) if isinstance(model.input_shape, tuple) else 200
    return model, tokenizer, label_encoder, max_len


def predict_classification_dl(input_text: str) -> tuple[str, float]:
    model, tokenizer, label_encoder, max_len = load_classification_dl()
    sequence = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)[0]
    index = int(np.argmax(probabilities))
    label = (
        str(label_encoder.inverse_transform([index])[0])
        if label_encoder is not None
        else str(index)
    )
    return label, float(np.max(probabilities))


@st.cache_resource
def load_classification_transformer(model_name: str):
    require(
        AutoTokenizer is not None
        and AutoModelForSequenceClassification is not None
        and pipeline is not None,
        "transformers is not installed.",
    )
    model_source = HF_MODEL_REPOS.get(model_name, CLS_TRANS_DIR / model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    label_path = CLS_TRANS_DIR / model_name / "label_encoder.pkl"
    fallback_label_path = CLS_ML_DIR / "label_encoder.pkl"
    label_encoder = None
    if label_path.exists():
        label_encoder = load_joblib(label_path)
    elif fallback_label_path.exists():
        label_encoder = load_joblib(fallback_label_path)

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
    )
    return classifier, label_encoder


def predict_classification_transformer(input_text: str, model_name: str) -> tuple[str, float]:
    classifier, label_encoder = load_classification_transformer(model_name)
    result = classifier(input_text, truncation=True, max_length=512)[0]
    return decode_label(str(result["label"]), label_encoder), float(result["score"])


# ---------------------------------------------------------------------
# NER models
# ---------------------------------------------------------------------

def predict_ner_rule_based(input_text: str) -> list[dict[str, Any]]:
    patterns = [
        ("DATE", r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b"),
        ("ORG", r"\b[A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*)*\s+(?:Inc|Ltd|LLC|Corp|Corporation|Company|University|Bank|Agency|Committee)\b"),
        ("LOC", r"\b(?:India|USA|United States|United Kingdom|New York|Washington|London|California|Delhi|Mumbai)\b"),
        ("PER", r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"),
    ]

    entities = []
    used_spans: list[tuple[int, int]] = []
    for label, pattern in patterns:
        for match in re.finditer(pattern, input_text):
            span = (match.start(), match.end())
            if any(not (span[1] <= old[0] or span[0] >= old[1]) for old in used_spans):
                continue
            used_spans.append(span)
            entities.append(
                {
                    "word": match.group(0),
                    "entity_group": label,
                    "start": match.start(),
                    "end": match.end(),
                    "score": None,
                }
            )
    return sorted(entities, key=lambda item: item["start"])


@st.cache_resource
def load_ner_transformer():
    require(
        AutoTokenizer is not None
        and AutoModelForTokenClassification is not None
        and pipeline is not None,
        "transformers is not installed.",
    )
    if (NER_TRANS_DIR / "config.json").exists():
        model_source = NER_TRANS_DIR
    else:
        model_source = HF_MODEL_REPOS.get("hf-bert-news-ner")
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForTokenClassification.from_pretrained(model_source)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=DEVICE,
    )


def predict_ner_transformer(input_text: str) -> list[dict[str, Any]]:
    ner_pipeline = load_ner_transformer()
    return ner_pipeline(input_text)


@st.cache_resource
def load_ner_dl():
    require(load_model is not None, "tensorflow is not installed. Run: pip install tensorflow")
    require(pad_sequences is not None, "tensorflow preprocessing is unavailable.")

    model = load_model(NER_DL_DIR / "final_best_model.keras", compile=False)
    metadata_candidates = [
        NER_DL_DIR / "encoded_data.pkl",
        NER_DL_DIR / "processed_data.pkl",
    ]
    metadata = None
    for candidate in metadata_candidates:
        if candidate.exists():
            try:
                metadata = load_pickle(candidate)
                break
            except Exception:
                if joblib is not None:
                    metadata = load_joblib(candidate)
                    break

    require(metadata is not None, "Missing NER metadata pickle with word/tag mappings.")
    word2idx = find_in_nested_dict(metadata, {"word2idx", "word_to_idx", "word_index"})
    char2idx = find_in_nested_dict(metadata, {"char2idx", "char_to_idx"})
    idx2tag = find_in_nested_dict(metadata, {"idx2tag", "idx_to_tag", "index_to_tag"})
    tag2idx = find_in_nested_dict(metadata, {"tag2idx", "tag_to_idx"})

    if idx2tag is None and tag2idx is not None:
        idx2tag = {value: key for key, value in tag2idx.items()}

    require(word2idx is not None, "Could not find word2idx in NER metadata.")
    require(idx2tag is not None, "Could not find idx2tag/tag2idx in NER metadata.")

    max_len = int(model.input_shape[0][1] if isinstance(model.input_shape, list) else model.input_shape[1])
    max_char_len = 20
    if isinstance(model.input_shape, list) and len(model.input_shape) > 1:
        max_char_len = int(model.input_shape[1][2])
    return model, word2idx, char2idx, idx2tag, max_len, max_char_len


def predict_ner_dl(input_text: str) -> list[dict[str, Any]]:
    model, word2idx, char2idx, idx2tag, max_len, max_char_len = load_ner_dl()
    tokens = re.findall(r"\w+|[^\w\s]", input_text)

    pad_id = word2idx.get("<PAD>", 0)
    oov_id = word2idx.get("<OOV>", word2idx.get("UNK", 1))
    encoded = [word2idx.get(token.lower(), oov_id) for token in tokens]
    word_input = pad_sequences([encoded], maxlen=max_len, padding="post", truncating="post", value=pad_id)

    if isinstance(model.input_shape, list) and len(model.input_shape) > 1:
        require(char2idx is not None, "This NER DL model needs char2idx metadata.")
        char_input = []
        for token in tokens[:max_len]:
            chars = [char2idx.get(char.lower(), 1) for char in token[:max_char_len]]
            chars += [0] * (max_char_len - len(chars))
            char_input.append(chars)
        char_input += [[0] * max_char_len] * (max_len - len(char_input))
        predictions = model.predict([word_input, np.array([char_input])], verbose=0)[0]
    else:
        predictions = model.predict(word_input, verbose=0)[0]

    tag_ids = np.argmax(predictions, axis=-1)[: len(tokens)]
    token_tags = [(token, str(idx2tag.get(int(tag_id), "O"))) for token, tag_id in zip(tokens, tag_ids)]
    return bio_tags_to_entities(token_tags)


def bio_tags_to_entities(token_tags: list[tuple[str, str]]) -> list[dict[str, Any]]:
    entities = []
    current_tokens = []
    current_label = None

    def flush_entity():
        nonlocal current_tokens, current_label
        if current_tokens and current_label:
            entities.append(
                {
                    "word": " ".join(current_tokens),
                    "entity_group": current_label,
                    "score": None,
                }
            )
        current_tokens = []
        current_label = None

    for token, tag in token_tags:
        if tag.startswith("B-"):
            flush_entity()
            current_label = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_label == tag[2:]:
            current_tokens.append(token)
        else:
            flush_entity()
    flush_entity()
    return entities


# ---------------------------------------------------------------------
# Summarization models
# ---------------------------------------------------------------------

def summarize_extractive(input_text: str, top_k: int = 3) -> str:
    sentences = split_sentences(input_text)
    if len(sentences) <= top_k:
        return " ".join(sentences)

    if TfidfVectorizer is None:
        stopwords = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "to", "of", "in", "on", "for", "with", "as", "by", "from", "at",
            "this", "that", "it", "its", "will", "be", "has", "have", "had",
        }
        words = re.findall(r"[a-zA-Z]{3,}", input_text.lower())
        frequencies = {}
        for word in words:
            if word not in stopwords:
                frequencies[word] = frequencies.get(word, 0) + 1

        scored = []
        for index, sentence in enumerate(sentences):
            sentence_words = re.findall(r"[a-zA-Z]{3,}", sentence.lower())
            score = sum(frequencies.get(word, 0) for word in sentence_words)
            scored.append((score, index))
        selected = sorted(index for _, index in sorted(scored, reverse=True)[:top_k])
        return " ".join(sentences[index] for index in selected)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(sentences)
    scores = np.asarray(matrix.sum(axis=1)).ravel()
    selected = sorted(np.argsort(scores)[-top_k:])
    return " ".join(sentences[index] for index in selected)


@st.cache_resource
def load_summarization_transformer(model_name: str):
    require(
        AutoTokenizer is not None and AutoModelForSeq2SeqLM is not None,
        "transformers is not installed.",
    )
    model_source = HF_MODEL_REPOS.get(model_name, SUM_TRANS_DIR / model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
    return tokenizer, model


def summarize_transformer(input_text: str, model_name: str) -> str:
    tokenizer, model = load_summarization_transformer(model_name)
    prefix = "summarize: " if model_name.startswith("t5") else ""
    inputs = tokenizer(
        prefix + input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    outputs = model.generate(
        inputs["input_ids"],
        max_length=120,
        min_length=25,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@st.cache_resource
def load_summarization_dl(model_name: str):
    require(load_model is not None, "tensorflow is not installed. Run: pip install tensorflow")
    require(pad_sequences is not None, "tensorflow preprocessing is unavailable.")

    model_path = SUM_DL_DIR / ("gru_model.h5" if model_name == "GRU" else "lstm_model.h5")
    model = load_model(model_path, compile=False)
    src_tokenizer = load_joblib(SUM_DL_DIR / "src_tokenizer.pkl.pkl")
    tgt_tokenizer = load_joblib(SUM_DL_DIR / "tgt_tokenizer.pkl")
    config = load_joblib(SUM_DL_DIR / "config.pkl")
    return model, src_tokenizer, tgt_tokenizer, config


def summarize_dl(input_text: str, model_name: str) -> str:
    model, src_tokenizer, tgt_tokenizer, config = load_summarization_dl(model_name)
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        max_src_len = int(input_shape[0][1] or 512)
        max_tgt_len = int(input_shape[1][1] or 60)
    else:
        max_src_len = int(input_shape[1] or 512)
        max_tgt_len = int(config.get("max_summary_len", config.get("max_tgt_len", 60))) if isinstance(config, dict) else 60

    src_seq = src_tokenizer.texts_to_sequences([input_text])
    src_pad = pad_sequences(src_seq, maxlen=max_src_len, padding="post", truncating="post")
    target_word_index = getattr(tgt_tokenizer, "word_index", {})
    index_word = getattr(tgt_tokenizer, "index_word", {})
    start_id = (
        target_word_index.get("sostok")
        or target_word_index.get("start")
        or target_word_index.get("<start>")
        or target_word_index.get("startseq")
        or 1
    )
    end_ids = {
        token_id
        for word, token_id in target_word_index.items()
        if word in {"eostok", "end", "<end>", "endseq"}
    }

    # Training-time seq2seq models usually need both encoder input and decoder input.
    # For app inference, decode autoregressively by feeding each prediction back in.
    if isinstance(input_shape, list) and len(input_shape) == 2:
        decoder_input = np.zeros((1, max_tgt_len), dtype="int32")
        decoder_input[0, 0] = int(start_id)
        generated_ids = []

        for step in range(max_tgt_len - 1):
            predictions = model.predict([src_pad, decoder_input], verbose=0)
            if isinstance(predictions, list):
                predictions = predictions[0]

            next_id = int(np.argmax(predictions[0, step, :]))
            if next_id == 0 or next_id in end_ids:
                break

            generated_ids.append(next_id)
            decoder_input[0, step + 1] = next_id
    else:
        predictions = model.predict(src_pad, verbose=0)
        if isinstance(predictions, list):
            predictions = predictions[0]
        generated_ids = np.argmax(predictions[0], axis=-1)[:max_tgt_len]

    words = []
    for token_id in generated_ids:
        word = index_word.get(int(token_id), "")
        if word in {"", "<pad>", "start", "sostok", "<start>", "startseq"}:
            continue
        if word in {"end", "eostok", "<end>", "endseq"}:
            break
        words.append(word)
    summary = " ".join(words).strip()
    require(summary, "DL summarizer returned an empty summary. Check decoder inference code.")
    return summary


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("News Intelligence System")
st.caption("Classification, Named Entity Recognition, and Summarization")

with st.sidebar:
    st.header("Model Options")
    task = st.selectbox(
        "Task",
        ["Text Classification", "Named Entity Recognition", "Summarization"],
    )

    if task == "Text Classification":
        cls_families = ["Keyword Baseline"]
        if (CLS_ML_DIR / "best_ml_model.pkl").exists() and (CLS_ML_DIR / "best_text_ML_model.pkl").exists():
            cls_families.append("From-Scratch ML")
        if (CLS_ML_DIR / "best_dl_model.h5").exists():
            cls_families.append("From-Scratch DL")
        cls_transformers = available_dirs(CLS_TRANS_DIR, "config.json") + available_hf_models("classification")
        if cls_transformers:
            cls_families.append("Pretrained Transformer")
        if not cls_families:
            cls_families = ["Pretrained Transformer"]

        model_family = st.selectbox(
            "Model Family",
            cls_families,
        )
        model_name = "BERT_model"
        if model_family == "Pretrained Transformer":
            model_name = st.selectbox("Model", cls_transformers)
        elif model_family == "From-Scratch ML":
            model_name = "best_ml_model.pkl"
        elif model_family == "From-Scratch DL":
            model_name = "best_dl_model.h5"
        else:
            model_name = "keyword_baseline"

    elif task == "Named Entity Recognition":
        ner_families = ["Rule-Based Baseline"]
        if (NER_DL_DIR / "final_best_model.keras").exists():
            ner_families.append("From-Scratch DL")
        if (NER_TRANS_DIR / "config.json").exists() or available_hf_models("ner"):
            ner_families.append("Pretrained Transformer")

        model_family = st.selectbox(
            "Model Family",
            ner_families,
        )
        model_name = {
            "Pretrained Transformer": "NER_BERT_MODEL",
            "From-Scratch DL": "final_best_model.keras",
            "Rule-Based Baseline": "regex_heuristic",
        }[model_family]

    else:
        sum_families = ["Extractive Baseline"]
        if (SUM_DL_DIR / "gru_model.h5").exists() or (SUM_DL_DIR / "lstm_model.h5").exists():
            sum_families.append("From-Scratch DL")
        sum_transformers = available_dirs(SUM_TRANS_DIR, "config.json") + available_hf_models("summarization")
        if sum_transformers:
            sum_families.append("Pretrained Transformer")

        model_family = st.selectbox(
            "Model Family",
            sum_families,
        )
        if model_family == "Pretrained Transformer":
            model_name = st.selectbox("Model", sum_transformers)
        elif model_family == "From-Scratch DL":
            dl_options = []
            if (SUM_DL_DIR / "gru_model.h5").exists():
                dl_options.append("GRU")
            if (SUM_DL_DIR / "lstm_model.h5").exists():
                dl_options.append("LSTM")
            model_name = st.selectbox("Model", dl_options)
        else:
            model_name = "TF-IDF"

    st.write("Device:", "GPU" if DEVICE == 0 else "CPU")
    st.write("Database logging:", "enabled" if os.getenv("DATABASE_URL") else "disabled")


uploaded_file = st.file_uploader("Upload a .txt article", type=["txt"])
uploaded_text = read_uploaded_text(uploaded_file)

text_input = st.text_area(
    "Paste news article text",
    value=uploaded_text,
    height=260,
    placeholder="Paste or upload a news article here...",
)


if st.button("Run", type="primary"):
    if not text_input.strip():
        st.warning("Please paste text or upload a .txt file.")
        st.stop()

    try:
        with st.spinner(f"Running {task} using {model_family}..."):
            if task == "Text Classification":
                if model_family == "Keyword Baseline":
                    label, confidence = predict_classification_keyword(text_input)
                elif model_family == "From-Scratch ML":
                    label, confidence = predict_classification_ml(text_input)
                elif model_family == "From-Scratch DL":
                    label, confidence = predict_classification_dl(text_input)
                else:
                    label, confidence = predict_classification_transformer(text_input, model_name)

                st.subheader("Classification Result")
                st.metric("Predicted Label", label)
                if confidence is not None:
                    st.metric("Confidence", f"{confidence:.4f}")
                output_text = f"label={label}; confidence={confidence}"

            elif task == "Named Entity Recognition":
                if model_family == "Rule-Based Baseline":
                    entities = predict_ner_rule_based(text_input)
                elif model_family == "From-Scratch DL":
                    entities = predict_ner_dl(text_input)
                else:
                    entities = predict_ner_transformer(text_input)

                st.subheader("Entities")
                if not entities:
                    st.info("No entities found.")
                else:
                    st.dataframe(
                        [
                            {
                                "entity": entity.get("word", ""),
                                "label": entity.get("entity_group", entity.get("entity", "")),
                                "score": entity.get("score", ""),
                            }
                            for entity in entities
                        ],
                        use_container_width=True,
                    )
                output_text = str(entities[:20])

            else:
                if model_family == "Extractive Baseline":
                    summary = summarize_extractive(text_input)
                elif model_family == "From-Scratch DL":
                    summary = summarize_dl(text_input, model_name)
                else:
                    summary = summarize_transformer(text_input, model_name)

                st.subheader("Summary")
                st.write(summary)
                output_text = summary

        log_inference(
            task_type=task,
            model_family=model_family,
            model_name=model_name,
            input_text=text_input,
            output_text=output_text,
            error_flag=False,
        )

    except Exception as exc:
        st.error("This path failed. The message below tells you exactly what to fix.")
        st.exception(exc)
        log_inference(
            task_type=task,
            model_family=model_family,
            model_name=model_name,
            input_text=text_input,
            output_text=str(exc),
            error_flag=True,
        )


with st.expander("Debug paths"):
    st.write("BASE_DIR", str(BASE_DIR))
    st.write("Classification ML/DL", str(CLS_ML_DIR))
    st.write("Classification Transformers", str(CLS_TRANS_DIR))
    st.write("NER DL", str(NER_DL_DIR))
    st.write("NER Transformer", str(NER_TRANS_DIR))
    st.write("Summarization DL", str(SUM_DL_DIR))
    st.write("Summarization Transformers", str(SUM_TRANS_DIR))
