"""
app.py
------
Streamlit demo for the Mini-Transformer-LLM.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud (share.streamlit.io) or
Hugging Face Spaces — see README.md for step-by-step instructions.
"""

import os
import time

import streamlit as st
import torch

import config
from model import MiniGPT
from tokenizer import CharTokenizer

st.set_page_config(page_title="Mini-Transformer-LLM", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_tokenizer(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        return None, None

    vocab_path = os.path.join(os.path.dirname(checkpoint_path), "vocab.json")
    tokenizer = CharTokenizer(vocab_path=vocab_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = MiniGPT(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


# ----------------------------- UI -----------------------------

st.title("Mini-Transformer-LLM")
st.caption(
    "A GPT-style decoder-only Transformer, built entirely from scratch in PyTorch "
    "(no pretrained weights, no external LLM APIs) and trained on AI/ML text from Wikipedia."
)

model, tokenizer = load_model_and_tokenizer(config.CHECKPOINT_PATH)

if model is None:
    st.error(
        f"No trained checkpoint found at `{config.CHECKPOINT_PATH}`.\n\n"
        "Train the model first with `python train.py`, then make sure "
        "`checkpoints/model.pt` and `checkpoints/vocab.json` are committed "
        "to the repo before deploying."
    )
    st.stop()

with st.sidebar:
    st.header("Generation settings")
    max_new_tokens = st.slider("Max new tokens", 50, 1000, 300, step=50)
    temperature = st.slider(
        "Temperature", 0.1, 1.5, 0.6, step=0.05,
        help="Lower = more predictable/repetitive. Higher = more random/creative.",
    )
    top_k = st.slider(
        "Top-k", 1, 100, 40, step=1,
        help="Only sample from the k most likely next characters at each step.",
    )
    st.divider()
    st.metric("Model parameters", f"{model.num_params() / 1e6:.2f}M")
    st.metric("Vocabulary size", tokenizer.vocab_size)
    st.metric("Context length", config.BLOCK_SIZE)

prompt = st.text_area(
    "Prompt",
    value="Artificial intelligence is",
    height=100,
    help="The model continues from whatever text you give it — try "
         "'Machine learning', 'A neural network', 'Deep learning models', etc.",
)

if st.button("Generate", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Enter a prompt first.")
    else:
        with st.spinner("Generating..."):
            start = time.time()
            context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            with torch.no_grad():
                out = model.generate(
                    context,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
            text = tokenizer.decode(out[0].tolist())
            elapsed = time.time() - start

        st.text_area("Output", value=text, height=300)
        st.caption(f"Generated {max_new_tokens} tokens in {elapsed:.1f}s (CPU)")

st.divider()
with st.expander("New to AI/ML? Start here"):
    st.markdown(
        """
**Artificial Intelligence (AI)** is the broad field of building systems
that perform tasks which normally require human intelligence — recognizing
images, understanding language, making decisions.

**Machine Learning (ML)** is a way of building AI: instead of hand-coding
rules, you show the system lots of examples and let it learn patterns from
data. This app's model learned to predict text purely from reading
Wikipedia articles about AI and ML, without anyone writing grammar rules
for it.

**Deep Learning** is ML using *neural networks* — layered mathematical
functions loosely inspired by neurons — stacked deep enough to learn
complex patterns. This project's neural network has 6 stacked layers.

**Large Language Models (LLMs)** — like GPT, Claude, or this project (at a
much smaller scale) — are deep learning models trained to predict the next
word/character in text, over and over, on huge amounts of text. Do that
enough times on enough data and the model ends up implicitly learning
grammar, facts, and reasoning patterns.

**Transformer** is the specific neural network architecture nearly all
modern LLMs are built on (introduced in the 2017 paper *"Attention Is All
You Need"*). Its key idea is **self-attention**: for every word, the model
learns which other words in the sentence matter most for understanding it.
That's exactly what's implemented from scratch in this project.

**How this specific model was trained:** it reads a text corpus (here,
Wikipedia articles on AI/ML topics) one chunk at a time, and for every
position is asked "given everything before this point, predict the next
character." It's wrong at first (random guesses), but after seeing
millions of these examples and adjusting its internal weights via
gradient descent, it gets progressively better at predicting
realistic-looking text.
        """
    )

st.divider()
with st.expander("About this project"):
    st.markdown(
        """
This is a **from-scratch implementation** of a GPT-style Transformer — every
component (multi-head causal self-attention, feed-forward layers, positional
embeddings, autoregressive sampling) is hand-written in PyTorch, not loaded
from a pretrained model.

**Architecture:** decoder-only Transformer, pre-layernorm residual blocks,
character-level tokenization, trained with next-token-prediction /
cross-entropy loss on a corpus of AI/ML Wikipedia articles.

Source code: [add your GitHub link here]
        """
    )