# Mini-Transformer-LLM

A complete, from-scratch implementation of a GPT-style decoder-only
Transformer in PyTorch — built for learning, not for production scale.
This is the same architecture family behind GPT-2/GPT-3 (simplified),
so once you understand this, you understand the core of every modern LLM.

## Project structure

```
mini-transformer-llm/
├── config.py       # all hyperparameters
├── tokenizer.py    # simple character-level tokenizer
├── dataset.py       # data loading + batching
├── model.py         # the Transformer architecture itself
├── train.py          # training loop
├── generate.py       # text generation / inference
├── requirements.txt
└── data/
    └── input.txt     # put your training text here
```

## How the model works

1. **Tokenization** — text is split into characters (or swap in a
   real BPE tokenizer later) and mapped to integer ids.
2. **Embeddings** — each token id is mapped to a learned vector
   (`token_embedding`), and a `position_embedding` is added so the
   model knows *where* each token sits in the sequence.
3. **Transformer blocks** (`model.py::TransformerBlock`) — stacked
   `N_LAYER` times, each containing:
   - **Causal multi-head self-attention**: every token looks back at
     itself and all previous tokens (never future ones — that's what
     makes it a valid autoregressive language model) and decides how
     much to "attend" to each, via `softmax(QK^T / sqrt(d_k))V`.
   - **Feed-forward network**: a 2-layer MLP applied independently to
     every token position, expanding to 4x width and back.
   - Both sublayers use **pre-layernorm** + **residual connections**,
     which is what makes deep transformers trainable.
4. **Output head** — a final linear layer projects back to vocabulary
   size, producing logits over "what's the next token?" at every
   position. Trained with cross-entropy loss against the actual next
   token (this is *next-token prediction*, the core LLM training
   objective).
5. **Generation** (`model.py::generate`) — at inference time, tokens
   are sampled one at a time from the predicted distribution
   (with temperature + top-k filtering) and fed back in autoregressively.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get training data

Any plain text file works. To try the classic tiny-shakespeare dataset:

```bash
curl -L -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Or use your own corpus — book text, your notes, code, chat logs, anything.
More data + more diverse data = better results.

## 2. Train

```bash
python train.py
```

This will print training/validation loss periodically and save the
best checkpoint to `checkpoints/model.pt`. On a single GPU this default
config (≈6-layer, ≈256-dim model, ~10-20M params) trains in a few
minutes; on CPU it'll be much slower — reduce `MAX_ITERS`, `N_LAYER`,
`N_EMBED`, and `BLOCK_SIZE` in `config.py` if you're CPU-only.

## 3. Generate text

```bash
python generate.py --prompt "ROMEO:" --max_new_tokens 300
```

## Tuning knobs (`config.py`)

| Parameter | What it controls |
|---|---|
| `BLOCK_SIZE` | context window / max sequence length |
| `N_EMBED` | model width (d_model) |
| `N_HEAD` | number of attention heads |
| `N_LAYER` | depth (number of transformer blocks) |
| `DROPOUT` | regularization strength |
| `BATCH_SIZE`, `LEARNING_RATE` | training dynamics |

Scaling up `N_EMBED`/`N_LAYER`/`N_HEAD` gets you closer to real LLM
scale (GPT-2 small is 12 layers, 768 dim, 12 heads, ~124M params) —
just needs more data and compute.

## Try it in the browser (Streamlit demo)

The project includes `app.py`, a Streamlit UI for typing a prompt and
generating text interactively, with sliders for temperature, top-k, and
output length.

Run it locally:

```bash
streamlit run app.py
```

This opens a local page at `http://localhost:8501`.

## Deploying as a portfolio project

You have a trained checkpoint at `checkpoints/model.pt` +
`checkpoints/vocab.json` — these **are the model**, so they need to be
committed to your repo (unlike your data file, which doesn't need to be
deployed). This checkpoint is small (a few MB for this config), so
that's fine.

### 1. Push to GitHub

```bash
git init
git add app.py model.py train.py generate.py dataset.py tokenizer.py config.py requirements.txt README.md
git add -f checkpoints/model.pt checkpoints/vocab.json   # -f because .gitignore ignores checkpoints by default
git commit -m "Mini-Transformer-LLM: from-scratch GPT in PyTorch"
git branch -M main
git remote add origin https://github.com/<your-username>/mini-transformer-llm.git
git push -u origin main
```

(Create the empty repo on GitHub first if you haven't already.)

### 2. Deploy on Streamlit Community Cloud (same flow as your Spam/Ham project)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**, pick your `mini-transformer-llm` repo, branch `main`,
   and set the main file path to `app.py`.
3. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and
   launches `app.py` automatically.
4. You'll get a public URL like `https://<your-app>.streamlit.app` —
   put this in your portfolio/resume.

### 3. Alternative: Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space),
   choosing **Streamlit** as the SDK.
2. Either push your GitHub repo to the Space's git remote, or upload the
   files directly through the Spaces web UI (same file list as above).
3. HF Spaces auto-installs `requirements.txt` and runs `app.py`.
4. You get a URL like `https://huggingface.co/spaces/<you>/mini-transformer-llm`.

### Portfolio framing tips

Since this is a from-scratch implementation (not a call to an existing
LLM API), it's worth being explicit about that in your portfolio/resume
bullet — it demonstrates you understand attention, positional encoding,
and autoregressive training at the code level, not just how to prompt
a model. Suggested framing:

> Built a GPT-style Transformer language model from scratch in PyTorch
> (custom multi-head causal self-attention, positional embeddings,
> autoregressive decoding), trained on Shakespeare text, and deployed
> an interactive demo via Streamlit Community Cloud.

## Natural next steps once this works

- Swap the character tokenizer for a real BPE tokenizer (`tiktoken` or
  HuggingFace `tokenizers`) — much better sample efficiency.
- Add **KV-caching** in `generate()` for faster inference.
- Try **RoPE** (rotary positional embeddings) instead of learned
  absolute position embeddings — what most modern LLMs use.
- Add **mixed-precision training** (`torch.cuda.amp`) and gradient
  accumulation to train bigger models faster.
- Fine-tune on instruction-following data to make it chat-like.
