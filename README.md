# ML-Mini-Project — Latent Feature Extraction for Musical Genres from Raw Audio

Team: Harsha and Sunil (UE24CS352A Machine Learning mini-project).

This project reproduces the paper *Latent Feature Extraction for Musical Genres from Raw Audio* (Sawhney,
Vasavada & Wang, NIPS 2018 workshop; [references/musical_genre.pdf](references/musical_genre.pdf)). It then builds
improved models on top of it. Everything runs in one notebook:
[musical_genre_paper_reproduction.ipynb](musical_genre_paper_reproduction.ipynb).

For a plain-language walkthrough of everything we did and found, read **[summary.md](summary.md)**.

## What is in the notebook

**Part 1: paper reproduction.** The notebook rebuilds the paper's method: GTZAN audio (classical, jazz, metal, pop),
20,000-sample segments average-pooled to 500 values, a random 6,000/1,000/1,000 split, and the paper's three models:
- a two-layer neural network;
- a vanilla autoencoder;
- the Deep Softmax Autoencoder.

It then recreates the paper's Tables 1–2 and Figures 1, 3, 4, 5 and 6. Where the paper gives no value, the
authors' linked code is used, and the first cell lists the source of every setting. An appendix compares our numbers
with the paper's and tests why they differ.

**Part 2: improvements.** Four models are built on top of the reproduction, one change at a time:
1. the paper's model with its bugs fixed;
2. the same model with more of the audio signal;
3. a convolutional Deep Softmax Autoencoder on raw audio, our main contribution;
4. a mel-spectrogram Deep Softmax Autoencoder, as an extra.

Every model runs with 5 seeds on two splits:
- the **clip split** is the paper's random split, used to compare with the paper;
- the **song split** puts no song in more than one partition, to test on unseen songs.

All settings are chosen on the dev set. Part 2 ends with one comparison table covering every model.

### Headline results (test accuracy, mean ± std over 5 seeds)

| Model | Clip split | Song split |
|---|---|---|
| Paper, Deep Softmax AE (quoted from the PDF) | 65.3% | not reported |
| Our reproduction of the paper's Deep Softmax AE | 61.0% ± 2.2% | 58.4% ± 3.1% |
| Convolutional Deep Softmax AE, raw audio (Step 3) | 87.2% ± 1.7% | 83.2% ± 2.9% |
| Mel-spectrogram Deep Softmax AE (Step 4, extra) | 97.7% ± 0.8% | 94.5% ± 2.9% |

## Setup

Requires Python 3.12 or newer. Tested on Python 3.14.6, macOS arm64, CPU only.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
```

Open the notebook in VS Code (or Jupyter) and select `.venv/bin/python` as the kernel. On Windows use
`.venv\Scripts\python.exe`.

## Dataset

The notebook uses the GTZAN Genre Collection: 1,000 thirty-second clips in 10 genres, of which only 4 are used.
If the audio is not already at `data/raw/gtzan_qiuqiangkong/genres/<genre>/`, the first data cell does the
following:
1. downloads the archive (about 1.2 GB) from a pinned Hugging Face mirror;
2. checks its SHA-256;
3. extracts the four genres;
4. converts the files to `.wav` in `data/interim/gtzan_wav/`, as the paper did.

Provenance, checksums and the audit of the dataset are in [docs/dataset.md](docs/dataset.md). The audio is not
committed to the repository.

## Running

Use **Run All**. How long it takes depends on whether the Part 2 cache exists:

| Situation | Time on a laptop CPU |
|---|---|
| Part 2 cache present (`artifacts/improvements/`) | about 4 minutes |
| No cache, or `RETRAIN = True` in the first Part 2 cell | about 2–3 hours |

Part 1 always retrains, which takes about 1 minute, plus about 3 minutes for the appendix experiments. Part 2
caches every trained model and result under a key built from its full configuration, so changing a setting
retrains only what changed. The `artifacts/` folder is Git-ignored, so a fresh clone trains Part 2 once.

The last cell of the notebook is a demo. It plays a held-out test clip and shows the best model's predicted genre
probabilities. Change `DEMO_INDEX` to try another clip.

## Outputs

| Folder | Contents |
|---|---|
| `artifacts/paper_reproduction/` | Part 1 models, training histories, Figures 1/3/4/5/6, `results.json` |
| `artifacts/improvements/` | Part 2 cached models, histories and results; comparison chart; best-model confusion matrix and PCA |

## Repository layout

| Path | What it is |
|---|---|
| `musical_genre_paper_reproduction.ipynb` | the whole project: reproduction, improvements, comparison, demo |
| `summary.md` | plain-language summary of the work and the results |
| `plan.md` | project plan, decisions and the improvement plan |
| `docs/reference_audit.md` | where the paper, poster and authors' code disagree |
| `docs/dataset.md` | dataset sources, checksums and audit results |
| `references/` | the paper, poster and assignment guidelines (read-only) |
| `scripts/`, `tests/`, `requirements-inspect.txt` | earlier standalone dataset-inspection tools (optional; the notebook does not use them) |
| `data/metadata/` | lightweight dataset inventories produced by those tools |

## Attribution

GTZAN was created by George Tzanetakis and collaborators (Tzanetakis & Cook, *Musical Genre Classification of
Audio Signals*, IEEE TSAP 2002). The paper and its code ([arjunsawknee/Genre-Extraction](https://github.com/arjunsawknee/Genre-Extraction))
belong to their authors; this repository reimplements the method in PyTorch and does not copy their code.
