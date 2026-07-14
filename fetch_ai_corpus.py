"""
fetch_ai_corpus.py
-------------------
Downloads a set of AI/ML-related Wikipedia articles and concatenates
them into data/input.txt, so you can train the transformer on
AI/ML text instead of Shakespeare.

Usage:
    pip install requests
    python fetch_ai_corpus.py
"""

import os
import re
import time
import unicodedata

import requests

# Feel free to add/remove article titles to change what the model learns from.
# More articles = more data = generally better results (aim for at least
# a few hundred KB of text; more is better, similar in spirit to tiny-shakespeare's ~1MB).
ARTICLES = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Artificial neural network",
    "Transformer (deep learning architecture)",
    "Attention (machine learning)",
    "Large language model",
    "Natural language processing",
    "Supervised learning",
    "Unsupervised learning",
    "Reinforcement learning",
    "Convolutional neural network",
    "Recurrent neural network",
    "Generative adversarial network",
    "Backpropagation",
    "Gradient descent",
    "Overfitting",
    "Feature (machine learning)",
    "Computer vision",
    "Speech recognition",
    "GPT-3",
    "GPT-4",
    "ChatGPT",
    "History of artificial intelligence",
    "AI winter",
    "Turing test",
    "Expert system",
    "Symbolic artificial intelligence",
    "Machine translation",
    "Word embedding",
    "Self-supervised learning",
    "Transfer learning",
    "Data science",
    "Artificial general intelligence",
    "Ethics of artificial intelligence",
    "AI alignment",
    "Explainable artificial intelligence",
    "Robotics",
    "Autonomous robot",
    "Support vector machine",
    "Decision tree learning",
    "Random forest",
    "Bayesian network",
    "Genetic algorithm",
    "Perceptron",
    "Long short-term memory",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"
OUTPUT_PATH = os.path.join("data", "input.txt")

# Wikipedia's API rejects requests without a descriptive User-Agent
# (returns 403 Forbidden otherwise) — see https://meta.wikimedia.org/wiki/User-Agent_policy
HEADERS = {
    "User-Agent": "mini-transformer-llm-corpus-fetcher/1.0 (educational project; contact: none)"
}

# Only keep common, learnable characters. Wikipedia's plaintext extracts of
# AI/ML articles are full of math notation, IPA pronunciation, citation
# markers, and foreign scripts (from equations like "attention(Q,K,V)" or
# formulas rendered as Unicode symbols e.g. ∀, ⟩, ≤, α). A character-level
# model has to learn a separate embedding for every distinct character it
# sees, so a bloated "long tail" of rare symbols makes training much harder
# and produces garbled, symbol-heavy output. Stripping down to plain
# English text + basic punctuation makes the learning problem far easier.
ALLOWED_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " \n.,;:!?'\"()-–—/%$&"
)


def clean_text(text: str) -> str:
    # Remove Wikipedia citation markers like [12] or [citation needed] FIRST,
    # while the brackets are still present to match against.
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)

    # Normalize accented letters to their closest plain-ASCII form (e.g. "café" -> "cafe")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Drop any character not in our allowed set (strips leftover math/rare symbols)
    text = "".join(c for c in text if c in ALLOWED_CHARS)

    # Collapse excessive blank lines / repeated spaces left behind by stripping
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def fetch_article_text(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": title,
        "redirects": 1,
    }
    resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    for page in pages.values():
        text = page.get("extract", "")
        if text:
            return text
    return ""


def main():
    os.makedirs("data", exist_ok=True)
    chunks = []
    total_chars = 0

    for i, title in enumerate(ARTICLES, 1):
        print(f"[{i}/{len(ARTICLES)}] Fetching: {title}")
        try:
            text = fetch_article_text(title)
        except requests.RequestException as e:
            print(f"  -> failed ({e}), skipping")
            continue

        if not text:
            print("  -> no content found, skipping")
            continue

        text = clean_text(text)
        if not text:
            print("  -> nothing left after cleaning, skipping")
            continue

        chunks.append(f"\n\n=== {title} ===\n\n{text}")
        total_chars += len(text)
        time.sleep(0.5)  # be polite to Wikipedia's API and avoid 429 rate limiting

    corpus = "".join(chunks)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"\nDone. Wrote {len(corpus):,} characters ({len(corpus)/1e6:.2f} MB) to {OUTPUT_PATH}")
    print("You can now run: python train.py")


if __name__ == "__main__":
    main()