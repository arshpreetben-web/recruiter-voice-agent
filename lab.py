"""
Lab Activity 4: NLP for Human Resources — Bias Detection in Job Descriptions
- Detect gender/race/age bias using ALBERT + LDA topic modeling + T5 rewrite
- Usage: python lab4_bias_detection.py
- If you have your own CSV of job ads, set JOB_CSV variable to its path.
  CSV should contain at least: 'id', 'title', 'description' columns (or will use demo data).
"""

import re
import torch

import os
import random
import pandas as pd
import numpy as np
from collections import Counter

# --- NLP & modeling ---
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate

# --- Topic modeling ---
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords

# --- rewrite model ---
from transformers import AutoTokenizer as ATok, AutoModelForSeq2SeqLM

# --- sklearn helpers ---
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ------- CONFIG -------
JOB_CSV = "job_ads.csv"   # if you have a CSV, put it here (id,title,description)
RANDOM_SEED = 42
ALBERT_CHECKPOINT = "albert-base-v2"      # compact ALBERT
REWRITE_MODEL = "t5-small"                # small T5 for inclusive rewrites
N_LDA_TOPICS = 6
MAX_LEN = 128
EPOCHS = 2
BATCH = 8

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------- DOWNLOAD NLTK -------
nltk.download("stopwords")
nltk.download("punkt")
STOPWORDS = set(stopwords.words("english"))

# ------- Utilities -------
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|@\S+|#\S+"," ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# simple heuristic labeler for bias: returns 1 if biased, 0 if not (for demo only)
GENDER_WORDS_MALE = {"he","him","his","man","men","father","son","brother","young man","young"}
GENDER_WORDS_FEMALE = {"she","her","hers","woman","women","mother","daughter","sister","young woman"}
AGE_WORDS = {"young","senior","seniority","junior","experienced","freshers","fresh graduate"}

def heuristic_bias_label(text):
    """
    Heuristic: mark bias if explicitly gendered/ageed phrases appear.
    This is only a weak-supervision for demo. For real projects annotate data.
    """
    t = text.lower()
    # gendered terms: "young", "energetic" combined with "male/female" etc.
    gender_match = any(w in t for w in GENDER_WORDS_MALE.union(GENDER_WORDS_FEMALE))
    age_match = any(w in t for w in AGE_WORDS)
    # naive rules: explicit gender words OR age qualifiers are considered biased here
    return int(gender_match or age_match)

# ------- Load dataset (fallback) -------
print("Loading dataset (CSV fallback if available)...")
if os.path.exists(JOB_CSV):
    df = pd.read_csv(JOB_CSV)
    required_cols = {"description","title"}
    if not required_cols.intersection(set(df.columns)):
        raise ValueError(f"CSV must contain at least one of these columns: {required_cols}")
    # unify text column
    if "description" in df.columns:
        df["text"] = df["description"].astype(str)
    else:
        df["text"] = df["title"].astype(str)
else:
    print("No CSV found. Using tiny demo job ads dataset (for lab demo).")
    df = pd.DataFrame({
        "id":[1,2,3,4,5,6],
        "title":[
            "Senior Software Engineer - male preferred",
            "Marketing Associate - energetic young person required",
            "Data Analyst - open to all genders",
            "Frontend Developer - fresh graduate",
            "HR Manager - experienced woman preferred",
            "Customer Support Representative - no bias"
        ],
        "description":[
            "We are seeking a senior software engineer. He should have 8+ years of experience and be a strong team player.",
            "Looking for a young and energetic marketing associate to join our startup. Fresh minds encouraged.",
            "Responsible for data analysis and reporting. Equal opportunity employer and open to all.",
            "Front-end role for fresh graduates. Will train and mentor junior candidates.",
            "We prefer an experienced woman for managing HR activities. Ability to handle sensitive issues.",
            "Serve customers and ensure satisfaction. Open to applicants of any background."
        ]
    })
    df["text"] = df["title"] + ". " + df["description"]

# basic cleaning
df["clean_text"] = df["text"].apply(clean_text)

# create weak labels via heuristic (for demo/training). In real work use human labels.
df["label"] = df["clean_text"].apply(heuristic_bias_label)

print(f"Loaded {len(df)} job ads. Label distribution: {Counter(df['label'])}")

# ------- Topic Modeling (LDA) -------
print("\nRunning topic modeling (LDA) to group job ads ...")
tokenized = [ [w for w in nltk.word_tokenize(t.lower()) if w not in STOPWORDS and len(w)>2] for t in df["clean_text"] ]
dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(t) for t in tokenized]

# guard small-data cases
num_topics = min(N_LDA_TOPICS, max(2, len(dictionary)//5))
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=RANDOM_SEED)

print(f"Identified {num_topics} topics. Top words per topic:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# assign each doc to dominant topic
def get_doc_topic(bow):
    dist = lda_model.get_document_topics(bow)
    if not dist:
        return -1
    return max(dist, key=lambda x: x[1])[0]

df["bow"] = corpus
df["topic"] = df["bow"].apply(get_doc_topic)

print("\nSample docs with assigned topics (id,topic,label):")
print(df[["id","topic","label"]].head())

# ------- Prepare dataset for ALBERT fine-tuning -------
print("\nPreparing data for ALBERT training...")

# Ensure text is string
df["text_for_model"] = df["clean_text"].astype(str)

# handle small-class cases (if any class has <2 samples, do not stratify later)
label_counts = Counter(df["label"])
min_count = min(label_counts.values())
print("Label counts:", label_counts)

# Dataset split
if min_count >= 2 and len(label_counts) > 1:
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"])
else:
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# Create HuggingFace datasets
train_ds = Dataset.from_pandas(train_df[["text_for_model","label"]])
val_ds = Dataset.from_pandas(val_df[["text_for_model","label"]])

# tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(ALBERT_CHECKPOINT)
def tokenize_fn(batch):
    return tokenizer(batch["text_for_model"], truncation=True, padding="max_length", max_length=MAX_LEN)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

# make labels int32 and set format
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in ["input_ids","attention_mask","label"]])
val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in ["input_ids","attention_mask","label"]])
train_ds = train_ds.rename_column("label","labels")
val_ds = val_ds.rename_column("label","labels")
train_ds.set_format(type="torch")
val_ds.set_format(type="torch")

# compute class weights for loss if imbalance
y_train = train_df["label"].to_numpy()
if len(np.unique(y_train))>1:
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i,w in enumerate(class_weights)}
else:
    class_weights = None

# load model
model = AutoModelForSequenceClassification.from_pretrained(ALBERT_CHECKPOINT, num_labels=len(np.unique(df["label"])))
if class_weights:
    # set custom loss: modify model.forward not trivial here; simplest is to pass class weights to trainer loss via compute_loss wrapper or use CrossEntropyLoss inside compute_metrics — keep simple for lab: print weights for info
    print("Class weights (train):", class_weights)

# ------- Metrics & Trainer -------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1m = f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1_macro": f1m["f1"]}

training_args = TrainingArguments(
    output_dir="lab4_albert_out",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    learning_rate=2e-5,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    seed=RANDOM_SEED,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# Train (on CPU might be slow but for small demo it's fine)
print("\nStarting ALBERT fine-tuning (this may be slow on CPU)...")
trainer.train()

print("\nEvaluation on validation set:")
metrics = trainer.evaluate()
print(metrics)

# ------- Inference: predict bias scores for all job ads -------
print("\nRunning bias prediction for entire corpus...")
def predict_texts(texts_list):
    enc = tokenizer(texts_list, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        out = model(**enc.to(model.device))
    logits = out.logits.cpu().numpy()
    preds = np.argmax(logits, axis=-1)
    scores = np.max(logits, axis=-1)  # not calibrated probabilities but useful
    return preds, scores

preds_all, scores_all = predict_texts(df["text_for_model"].tolist())
df["pred_label"] = preds_all
df["pred_score"] = scores_all

print("\nPredictions (id,label,pred):")
print(df[["id","label","pred_label"]])

# ------- Inclusive Rewrite using T5 -------
print("\nGenerating inclusive rewrites (T5 small)...")
rewriter_tok = ATok.from_pretrained(REWRITE_MODEL)
rewriter = AutoModelForSeq2SeqLM.from_pretrained(REWRITE_MODEL)

def rewrite_inclusive(text, max_len=128):
    prompt = "Rewrite this job description to be inclusive, remove gender/age mentions, and keep required skills: " + text
    enc = rewriter_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = rewriter.generate(**enc, max_length=max_len, num_beams=4, early_stopping=True)
    return rewriter_tok.decode(out[0], skip_special_tokens=True)

# Show three examples: biased -> rewritten
sample_idx = df[df["pred_label"]==1].index.tolist()[:3]
if not sample_idx:
    sample_idx = [0,1]  # fallback
for i in sample_idx:
    print("\n--- ORIGINAL ---")
    print(df.loc[i,"text"][:400])
    print("\n--- REWRITE ---")
    try:
        print(rewrite_inclusive(df.loc[i,"text"][:400]))
    except Exception as e:
        print("Rewrite failed (maybe model download):", e)

# ------- Save outputs -------
OUT_CSV = "lab4_outputs.csv"
df_out = df[["id","text","label","pred_label","pred_score","topic"]]
df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved outputs to {OUT_CSV}")

# ------- Simple fairness analysis (counts) -------
print("\nFairness snapshot:")
counts = df_out.groupby(["label","pred_label"]).size().unstack(fill_value=0)
print(counts)

print("\nLab Activity 4 complete. NOTE: This is a demo pipeline — for real deployment, use human labels, careful fairness metrics (e.g., equalized odds), and legal review.")