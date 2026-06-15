# AI vs Human Text Detection

A machine learning project that classifies text as either **human-written** or **AI-generated**, with a focus on interpretability — understanding *why* the model makes the decisions it does.

Built as an independent research project to explore LLM behaviour detection techniques, directly relevant to ongoing research in NLP interpretability and AI safety.

---

## Motivation

As a software engineer who uses LLMs daily (GitHub Copilot, Claude), I repeatedly noticed that AI-generated text often *sounds* authoritative while being subtly or significantly wrong. This raised a question I couldn't easily answer: **what are the actual linguistic signals that distinguish AI-generated text from human writing?**

This project is an attempt to answer that question empirically.

---

## Research Questions

1. Can classical ML classifiers reliably distinguish AI-generated from human-written text?
2. Which **linguistic and stylistic features** are most predictive — and what does this tell us about how LLMs write?
3. Where does the model **fail**, and what do those failures reveal about the limits of surface-level detection?

---

## Dataset

- **52 samples** (27 human, 25 AI-generated)
- Human texts: casual writing, student writing, professional writing, opinions — representing the natural variation and imperfection of human language
- AI texts: typical LLM outputs including explanatory text, list-making, hedged opinions, formal academic style, and common LLM response patterns
- Balanced classes to ensure fair evaluation

---

## Features

### TF-IDF (n-grams 1–2)
Captures word and phrase patterns at the surface level.

### Linguistic / Stylistic Features (Hand-crafted)
These are designed to capture deeper stylistic signals:

| Feature | Rationale |
|---|---|
| `avg_word_len` | LLMs tend to use longer, more formal vocabulary |
| `avg_sent_len` | AI text often has longer, more structured sentences |
| `contractions` | Humans use contractions far more (it's, I've, can't) |
| `filler_words` | Humans use fillers (honestly, basically, tbh, lol) |
| `formal_connectors` | AI overuses (furthermore, consequently, in conclusion) |
| `has_numbering` | AI frequently structures lists with numbered items |
| `hedge_words` | AI uses hedging (may, might, suggest, appear) |
| `vocab_richness` | Type-token ratio — humans often more varied in short texts |
| `em_dash` | Human writers use em-dashes more naturally |
| `filler_words` | Strong negative predictor of AI text |

---

## Models Compared

| Model | Macro F1 | Accuracy |
|---|---|---|
| Logistic Regression (TF-IDF) | 1.000 | 1.000 |
| Logistic Regression (Combined) | 1.000 | 1.000 |
| Random Forest (Combined) | 1.000 | 1.000 |
| Gradient Boosting (Combined) | 1.000 | 1.000 |
| LinearSVC (TF-IDF) | 1.000 | 1.000 |

All models achieved perfect scores on this dataset — which is itself an interesting finding worth interrogating (see Limitations).

---

## Key Findings

### What predicts AI-generated text
- **Longer average word length** — strongest positive predictor
- **More sentences** — AI tends toward structured, multi-sentence responses
- **Numbered lists and formal connectors** — AI structures information formulaically
- **Longer average sentence length**

### What predicts human text
- **Contractions** — strongest negative predictor of AI text (humans write *it's*, AI writes *it is*)
- **Filler words** — humans write *honestly*, *basically*, *tbh*, *lol*
- **Em-dashes** — humans use these more naturally and idiomatically
- **Hedge words in informal contexts** — humans hedge differently than AI

### Error Analysis
The model made **zero errors** on the test set. This is a double-edged result:
- It confirms that current LLM outputs have strong and consistent stylistic signatures
- However, it also raises a concern: **the dataset may be too easy**. Human and AI samples were deliberately distinct in style. A harder test would include AI text that deliberately mimics informal human writing, or human text that happens to be formal and structured.

This is precisely the challenge facing real-world AI detection systems — as models improve at mimicking human writing styles, surface-level detection becomes unreliable.

---

## Limitations and Open Questions

1. **Small dataset (n=52)**: Results cannot be generalised. A production system would require thousands of samples from diverse sources.

2. **Dataset difficulty**: The clean separation between human and AI styles may not reflect real-world conditions where humans write formally or AI is prompted to write casually.

3. **Model brittleness**: These models rely on surface stylistic features. An LLM instructed to "write casually with contractions and filler words" would likely evade detection.

4. **The deeper question**: Even when detection works, we don't fully understand *why* LLMs produce the stylistic patterns they do. This connects to broader questions in neural network interpretability — what internal representations lead to these output patterns?

5. **Adversarial robustness**: As detection systems improve, LLMs will likely be fine-tuned to evade them. This is an arms race that surface-level detection alone cannot win.

---

## Implications for LLM Interpretability Research

The most interesting finding is not that detection works — it is **why** it works, and where it might stop working. The stylistic signals identified here (formal connectors, lack of contractions, structured numbering) are symptoms of something deeper: **the way LLMs represent and generate language is fundamentally different from how humans do**, even when the surface output appears similar.

Understanding these internal differences — through attention analysis, probing experiments, and mechanistic interpretability — is the next research frontier beyond surface-level detection.

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn

# Generate dataset
python generate_dataset.py

# Train and evaluate
python train_and_evaluate.py
```

---

## Project Structure

```
ai-text-detection/
├── generate_dataset.py      # Dataset creation
├── train_and_evaluate.py    # Training, evaluation, error analysis
├── data/
│   └── dataset.csv          # 52 labelled samples
├── outputs/
│   ├── results.png          # Visualisations
│   └── summary.csv          # Results summary
└── README.md
```

---

## Author

Ammar Yousuf Abrahani  
MSc Cloud Computing, National College of Ireland  
[LinkedIn](https://linkedin.com/in/ammar-abrahani) | [GitHub](https://github.com/ammarabrahani)
