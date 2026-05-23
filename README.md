# CS779: Statistical Natural Language Processing

**Dhruv Gupta · 240354 **

The course comprised of six assignments and a capstone project covering the breadth of NLP, from basic linguisticson large corpora and tokenizers built from scratch to a Transformer built entirely from scratch that produced **42,757 English to Indian language translations**.

---

## Highlights at a glance

| | |
|--|--|
| Transformer trained from scratch (PyTorch, no library wrappers) | EN→Hindi loss: **8.999 → 2.602** over 21 epochs |
| 4 subword tokenizers implemented from scratch (BPE, WP, SP, Unigram) | EN→Bengali loss: **8.999 → 4.125** over 21 epochs |
| Zipf's law validated across **11 Indian languages** | TF-IDF + KNN topic classifier: **63.5% accuracy** (10 classes) |
| 149,646 training pairs; 42,757 test translations generated | Unsupervised K-Means baseline: **55% accuracy** with no labels |

---

## Table of Contents

- [CS779: Statistical Natural Language Processing](#cs779-statistical-natural-language-processing)
	- [Highlights at a glance](#highlights-at-a-glance)
	- [Table of Contents](#table-of-contents)
	- [Assignment 1: Corpus Analysis \& Zipf's Law](#assignment-1-corpus-analysis--zipfs-law)
		- [Zipf's Law across 11 Indian Languages](#zipfs-law-across-11-indian-languages)
		- [RegEx on the Enron Corpus](#regex-on-the-enron-corpus)
		- [Wikipedia Corpus Analysis (~4.6M tokens)](#wikipedia-corpus-analysis-46m-tokens)
		- [N-Gram POS Tag Models](#n-gram-pos-tag-models)
		- [Topic Classification: TF-IDF + KNN vs. K-Means](#topic-classification-tf-idf--knn-vs-k-means)
		- [Word Embeddings via SVD](#word-embeddings-via-svd)
	- [Assignment 2: Subword Tokenization](#assignment-2-subword-tokenization)
	- [Assignment 3: Probabilistic Models \& EM](#assignment-3-probabilistic-models--em)
	- [Assignment 4: N-Gram Models \& Text Classification](#assignment-4-n-gram-models--text-classification)
		- [POS Tag N-Gram Models](#pos-tag-n-gram-models)
		- [Multi-Model Text Classification Pipeline](#multi-model-text-classification-pipeline)
	- [Assignment 6: Named Entity Recognition](#assignment-6-named-entity-recognition)
	- [Capstone: Neural Machine Translation for India](#capstone-neural-machine-translation-for-india)
		- [Data](#data)
		- [EDA drove the architecture](#eda-drove-the-architecture)
		- [Preprocessing](#preprocessing)
		- [Architecture progression](#architecture-progression)
		- [Final Model Spec](#final-model-spec)
		- [Training Results](#training-results)
		- [Vocabulary](#vocabulary)
		- [Files](#files)
	- [Tech Stack](#tech-stack)

---

## Assignment 1: Corpus Analysis & Zipf's Law

### Zipf's Law across 11 Indian Languages

Ran Zipf's law validation on the Indic NLP corpus (~27,000 Wikipedia articles, 11 languages). Used spaCy for English and `indic-nlp` for the Indian language tokenization, then fit log-log rank-frequency curves via linear regression to get the Zipfian exponent `k` per language.

Results clustered around the theoretical ideal (k=1):

| Language | k |
|----------|---|
| Kannada | 1.037 |
| Tamil | 1.036 |
| Telugu | 1.012 |
| English | 0.686 |
| Punjabi | 0.774 |
| **Average (11 languages)** | **0.84** |

Dravidian languages tracked Zipf's law most tightly. English and Punjabi deviated more, likely a corpus size and tokenization artefact rather than a linguistic one.

### RegEx on the Enron Corpus

Extracted phone numbers, email addresses, URLs, and domain-specific patterns from the Enron email corpus using handcrafted regex. Visualized frequency distributions, phone number format clusters, email domain breakdowns, and URL structure analysis using Plotly.

### Wikipedia Corpus Analysis (~4.6M tokens)

Sampled 1,000 articles from 27,000 and ran a full linguistic pipeline using spaCy:
- Article heading counts: min 2, max 745; longest article 63,374 tokens
- Punctuation accounted for **16.17%** of all tokens
- Built unigram probability models for tokens and POS tags; used log-probabilities throughout to avoid underflow
- Computed per-token POS tag entropy (avg ~2.2 bits) to measure syntactic versatility. Tokens with high entropy tend to be function words that appear across many grammatical contexts.
- NER tagging across all articles; computed per-document NER entropy to quantify entity diversity

### N-Gram POS Tag Models

Computed unigram/bigram/trigram POS transition probabilities from the same corpus. Top unigrams: PROPN (20.2%), NOUN (15.6%), PUNCT (15.5%), reflecting Wikipedia's density of proper nouns. Next-tag prediction from the three models often disagrees, which makes a clean argument for why context windows matter.

### Topic Classification: TF-IDF + KNN vs. K-Means

10-class Wikipedia topic classifier (~8,000 articles):

| Method | Config | Accuracy |
|--------|--------|----------|
| KNN (from scratch) | cosine similarity, k=7, 30K TF-IDF features | **63.5%** |
| KNN (from scratch) | Euclidean distance, k=7 | **63.5%** |
| KNN (from scratch) | Manhattan distance, k=7 | 27% |
| K-Means | k=10, majority-vote assignment | **55%** |

Manhattan distance collapses on high-dimensional sparse vectors, a useful failure mode to document. K-Means recovering 55% accuracy with zero label supervision shows that TF-IDF clusters naturally along topic lines.

### Word Embeddings via SVD

Built a word-context co-occurrence matrix and applied Truncated SVD to get dense word embeddings. Computed cosine similarities between word vectors to verify that semantically related words ended up close in the reduced space.

---

## Assignment 2: Subword Tokenization

Four tokenization algorithms written from scratch (no external tokenization libraries, plain Python and NumPy only) for mixed English-Hindi text with a 5,000-token target vocabulary.

| Algorithm | Core idea |
|-----------|-----------|
| **BPE** | Iteratively merge the most frequent adjacent symbol pair |
| **WordPiece** | Merge based on likelihood gain rather than raw frequency |
| **SentencePiece** | Treat the full Unicode byte stream as input; no pre-tokenization step |
| **Unigram LM** | Start with a large vocab and prune using an EM-based log-likelihood objective |

Each produced three output files: vocabulary, encoded training corpus, and a tokenized sample. All ran within the 10-minute execution constraint on a standard corpus.

---

## Assignment 3: Probabilistic Models & EM

**Naive Bayes:** Worked through the conditional independence assumption, where it's a reasonable approximation and where it clearly breaks (adjacent word dependencies in natural language mean it breaks almost everywhere, but the classifier is often still competitive in practice).

**Expectation-Maximization:** Implemented the EM algorithm from the ground up and applied it to a latent variable model on Wikipedia corpus data. The E-step computes expected sufficient statistics given the current parameters; the M-step re-estimates parameters to maximize expected log-likelihood. Iterated until convergence.

---

## Assignment 4: N-Gram Models & Text Classification

### POS Tag N-Gram Models

Unigram, bigram, and trigram transition probability tables from 1,000 Wikipedia articles. Used for next-POS prediction on held-out sentences; bigram and trigram models consistently predicted different (and more contextually appropriate) tags than unigram alone.

### Multi-Model Text Classification Pipeline

Full classification pipeline on a document dataset, evaluated across four embedding types and four model families:

**Embeddings:** TF-IDF · Word2Vec · FastText · pre-trained vectors

**Models tested:**
- Classical ML: multiple sklearn classifiers per embedding
- LSTM, GRU, CNN (with pooling variants)
- BiLSTM with Attention (best-performing DL model)
- BERT fine-tuned end-to-end

**Preprocessing:**
```
HTML decode → normalise backslashes → lowercase
→ strip punctuation/numbers → tokenise → remove stopwords → lemmatise
```

Evaluated on Accuracy, Precision, Recall, F1, and ROC-AUC across all combinations. BiLSTM with Attention hit the highest accuracy among DL models; BERT fine-tuning brought it further up at substantially higher compute cost.

---

## Assignment 6: Named Entity Recognition

NER system using BIO (Begin-Inside-Outside) sequence labeling on CoNLL-format data. Covered the full arc of approaches:

- **Rule-based:** gazetteers and handcrafted patterns, brittle but interpretable
- **CRF:** models label-label dependencies that an HMM's Markov assumption misses
- **BiLSTM-CRF:** bidirectional context + CRF decoding layer, the standard before Transformers took over
- **BERT-based:** contextual embeddings as input to the sequence labeler, current standard

Output files in `conll_output/`.

---

## Capstone: Neural Machine Translation for India

**English to Bengali and English to Hindi.** Built entirely from scratch in PyTorch (no Hugging Face, no pre-built seq2seq modules). Three architectures explored; a Transformer ended up as the final submission.

### Data

| | English to Bengali | English to Hindi | Total |
|-|-------------------|-----------------|-------|
| Training pairs | 68,849 | 80,797 | **149,646** |
| Test sentences | 19,672 | 23,085 | **42,757** |

### EDA drove the architecture

Before touching a model, a dedicated EDA pass on the full dataset surfaced a few things that mattered:

- **>90% of sentences in all three languages are under 25 tokens.** Set `MAX_LENGTH = 25`, so no wasted capacity on padding and no information lost from truncation.
- **Bengali vocabulary is nearly twice the size of Hindi's** (102,935 vs 78,109 unique types) with a type-token ratio of 0.108 vs 0.053. Bengali is morphologically richer, meaning the model has to predict from a much larger output space, which explains the slower and higher-loss convergence.
- ~2,791 duplicate sentence pairs in training data identified and noted.

### Preprocessing

```
raw text → HTML entity decode → normalise backslashes → lowercase
→ remove punctuation + numbers → (target only: strip stray English chars)
→ NLTK word tokenise → insert <SOS>/<EOS> → pad/truncate to 25
```

### Architecture progression

**LSTM Seq2Seq (baseline):** 2-layer encoder-decoder LSTM, teacher forcing at 50%, embedding 256, hidden 512. The fixed-size context vector is a hard bottleneck (the entire source sentence gets compressed into a single vector). Hindi training loss after 10 epochs: ~4.625. Translations looked repetitive on longer inputs.

**Bi-GRU with Attention:** Replaced LSTMs with bidirectional GRUs, added a soft attention mechanism so the decoder can look back at all encoder hidden states rather than just the final one. Long-range dependencies handled much better; the improvement is most visible on sentences where the key content word appears early in English but late in the target language.

**Transformer (final):** Full *Attention Is All You Need* architecture, every component written from scratch: sinusoidal positional encoding, scaled dot-product attention, multi-head attention, position-wise FFN, encoder/decoder stacks with residual connections and layer norm, padding masks, and causal mask. No library wrappers at any level.

### Final Model Spec

| | |
|--|--|
| Encoder / Decoder layers | 3 each |
| Attention heads | 8 |
| d_model | 256 |
| Feed-forward dim | 512 |
| Dropout | 0.1 |
| Gradient clip | 1.0 |
| Max length | 25 tokens |
| Weight init | Xavier uniform |

**Weight tying:** The decoder's embedding matrix and the final output projection share weights. Fewer parameters, and it forces the model to keep output representations aligned with the input embedding space, which generally helps translation quality.

**Noam LR schedule** (directly from the paper):
```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```
4,000 warmup steps, then inverse square root decay. No manual LR tuning needed.

**Inference:** Greedy decoding, argmax at each step fed back as the next input. Stops on `<EOS>` or max length.

### Training Results

**English to Bengali** (21 epochs, 1,076 batches/epoch, ~82 min training, 32 min inference)

| Epoch | Loss |
|-------|------|
| 1 | 8.999 |
| 5 | 6.075 |
| 10 | 4.875 |
| 15 | 4.474 |
| **21** | **4.125** |

**English to Hindi** (21 epochs, 1,263 batches/epoch, ~28 min training, 36 min inference)

| Epoch | Loss |
|-------|------|
| 1 | 7.678 |
| 5 | 4.162 |
| 10 | 3.200 |
| 15 | 2.850 |
| **21** | **2.602** |

Hindi converged faster and to a lower final loss, directly attributable to its smaller output vocabulary (75,562 vs 105,528 for Bengali). At each decoder step, Bengali is predicting over ~40% more possible tokens.

Final output: `answersB.csv` + `answersH.csv` merged into `answer.csv` and zipped as `submission.zip`. **42,757 translations total.**

### Vocabulary

| | English (source) | Target |
|-|-----------------|--------|
| Bengali model | 53,923 | 105,528 |
| Hindi model | 57,278 | 75,562 |

### Files

```
Capstone Project/
├── FINAL SUBMIT/
│   ├── CS779-CP-Dhruv-Gupta-240354.ipynb    <- final Transformer (Bengali + Hindi)
│   ├── eda_traintest.ipynb                   <- EDA notebook
│   └── CS799-CP-Dhruv-Gupta-240354-report.pdf
├── simple_lstm.ipynb                         <- LSTM Seq2Seq prototype
├── Bi-GRU Attention.ipynb                    <- Bi-GRU + attention prototype
├── optimized_transformer_wth_bpe.ipynb       <- Transformer variant with BPE
├── train_data1.json                          <- 149,646 training pairs
├── val_data1.json
└── test_data1.json                           <- 42,757 test sentences
```

---

## Tech Stack

PyTorch · spaCy · NLTK · indic-nlp-library · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · Plotly

Trained on Kaggle (GPU) and Google Colab (GPU).
