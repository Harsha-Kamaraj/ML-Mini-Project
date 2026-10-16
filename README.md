# ML-Mini-Project — Latent Feature Extraction for Musical Genres from Raw Audio

Team: Harsha and Sunil (UE24CS352A Machine Learning mini-project).

This project implements the models from *Latent Feature Extraction for Musical Genres from Raw Audio*
(Sawhney, Vasavada & Wang, 2018; [paper](references/musical_genre.pdf)), then develops improved raw-audio
and mel-spectrogram models. The final experiment uses **all 1,000 GTZAN songs: 100 songs in each of 10 genres**.
The extension retains each existing model's hidden layers, losses and training schedules, with classifier
outputs expanded from 4 to 10.

**Final notebook: [musical_genre_classification.ipynb](musical_genre_classification.ipynb).** It contains the entire
workflow, with all 33 code cells completed and their outputs saved. It uses a CUDA GPU when available and
otherwise runs on CPU, including on a Mac. Completed CPU experiments were reused; the remaining 18
experiments ran on a Tesla T4. Results and checkpoints are stored separately from the earlier four-genre run.

Read **[Project_Writeup.pdf](Project_Writeup.pdf)** for the updated two-page project writeup,
or edit **[Project_Writeup.docx](Project_Writeup.docx)** in Word.
The **[editable PowerPoint](Musical_Genre_Classification_Final_10_Genres.pptx)** presents the same ten-genre experiment.

## View or rerun the final results

1. Open `musical_genre_classification.ipynb` in VS Code or Jupyter. Saved outputs can be viewed without execution.
2. To rerun locally, select the project's `.venv/bin/python` kernel after following **Setup** below.
3. Keep `REUSE_COMPLETED = True` in the first code cell and `RETRAIN = False` at the start of Part 2.
4. Click **Run All**. With the existing data and complete ten-genre artifact folders, the notebook reuses
   completed training and regenerates the tables, plots and predictions. Feature preparation and inference
   still take time.
5. Scroll to **Final comparison**, **Best model**, and **Demo** near the bottom. Change `DEMO_INDEX` in the
   last cell to play and classify another held-out clip.

Keep both `artifacts/paper_reproduction/gtzan_10_genres/` and
`artifacts/improvements/gtzan_10_genres/` to reuse the completed run. A missing cache entry starts training
for that experiment. The `artifacts/` and audio folders are Git-ignored and must be copied separately when
moving the project to another computer or Colab.

## What is in the notebook

**Part 1: paper models on all ten genres.** The notebook uses blues, classical, country, disco, hiphop, jazz,
metal, pop, reggae and rock. Each song supplies 20 segments of 20,000 samples, average-pooled to 500 values.
The resulting 20,000 examples use a random 15,000/2,500/2,500 split, preserving the original proportions,
and the paper's three models:

- a two-layer neural network;
- a vanilla autoencoder;
- the Deep Softmax Autoencoder.

It then recreates the paper's Tables 1–2 and Figures 3, 4 and 5. Where the paper gives no value, the
authors' linked code is used, and the first cell lists the source of every setting. An appendix retains the
paper's four-genre numbers as historical reference and runs the same robustness experiments on ten genres.
The quoted paper scores are not directly comparable with this expanded task.

**Part 2: improvements.** Four successive stages build on the reproduction:

1. the paper's model with its bugs fixed;
2. the same model with more of the audio signal;
3. a convolutional Deep Softmax Autoencoder on raw audio, our main contribution;
4. a mel-spectrogram Deep Softmax Autoencoder, as an extra.

The six classifiers in the final comparison each run with 5 seeds on two splits (60 evaluations):

- the **clip split** retains the paper's random-split proportions;
- the **song split** targets 80/10/10 songs per genre, keeping exact duplicate audio together across all
  genres, including the identical recording labelled both metal and rock, to test on unseen songs.

Hyperparameter sweeps select settings using the clip-split dev set at seed 0; the selected settings are then
evaluated across both splits and all five seeds. The displayed best model is selected by mean clip-split dev
accuracy. Test results do not determine these choices. The vanilla autoencoder has no classifier and is
evaluated through reconstruction rather than included in the classification comparison.

### Full ten-genre results (test clip accuracy, mean ± std over 5 seeds)

| Model | Clip split | Song split |
|---|---|---|
| Paper two-layer NN | 17.1% ± 0.8% | 17.2% ± 2.1% |
| Paper Deep Softmax AE | 28.0% ± 1.0% | 27.3% ± 1.9% |
| Step 1: fixed Deep Softmax AE | 27.6% ± 0.3% | 26.6% ± 1.6% |
| Step 2: more signal | 30.2% ± 0.7% | 29.2% ± 1.9% |
| Step 3: convolutional raw audio | 70.6% ± 1.8% | 61.2% ± 2.0% |
| Step 4: mel spectrograms | 87.3% ± 0.8% | 73.4% ± 5.7% |

The song split holds out entire songs and keeps exact duplicate recordings together. The clip split can
contain segments from the same song in training and test. These columns both measure **individual test-clip
accuracy**; the song split is the stronger test of performance on unseen songs.

The mel-spectrogram model is best overall. On the song split, its test macro-F1 is **73.1% ± 5.6%**.
Averaging clip probabilities into one prediction per held-out song gives **82.6% ± 6.3% song-level accuracy**,
compared with **74.8% ± 5.1%** for the convolutional raw-audio model. These song-level scores are a separate
metric from the table above.

Complete metrics are in
[comparison.csv](artifacts/improvements/gtzan_10_genres/comparison.csv) and
[all_results.csv](artifacts/improvements/gtzan_10_genres/all_results.csv).
The [comparison chart](artifacts/improvements/gtzan_10_genres/comparison_test_accuracy.png) shows the same
six classifiers. These links work when the local artifact folders are present.

### Historical four-genre results (test accuracy, mean ± std over 5 seeds)

These describe a different, four-class task and are retained only for historical context. They are not
directly comparable with the ten-class results above.

| Model | Clip split | Song split |
|---|---|---|
| Paper, Deep Softmax AE (quoted from the PDF) | 65.3% | not reported |
| Our reproduction of the paper's Deep Softmax AE | 61.0% ± 2.2% | 58.4% ± 3.1% |
| Convolutional Deep Softmax AE, raw audio (Step 3) | 87.2% ± 1.7% | 83.2% ± 2.9% |
| Mel-spectrogram Deep Softmax AE (Step 4, extra) | 97.7% ± 0.8% | 94.5% ± 2.9% |

## Setup

The local pinned environment was tested on **Python 3.14.6 / macOS arm64** with PyTorch 2.14.1 and
LibROSA 1.0.0. Use Python 3.14 for the pinned versions in `requirements-notebook.txt`.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
```

Open the notebook in VS Code (or Jupyter) and select `.venv/bin/python` as the kernel. On Windows use
`.venv\Scripts\python.exe`.

The GPU continuation used **Python 3.13.15, PyTorch 2.11.0+cu130, CUDA 13.0, LibROSA 0.11.0 and a Tesla T4**.
Its full package versions are recorded in
`artifacts/improvements/gtzan_10_genres/colab_environment.json`. The final comparison combines retained
CPU results with the completed GPU experiments. Cached scores retain their original values; freshly
computed predictions or training results can differ across hardware and package versions. Use the recorded
environment when reproducing GPU results. Cache keys do not include package versions.

## Dataset

The notebook uses the full GTZAN Genre Collection: **1,000 recordings, 100 per genre**, approximately
30 seconds each at 22,050 Hz. The genres are blues, classical, country, disco, hiphop, jazz, metal, pop,
reggae and rock. All 1,000 recordings were decoded successfully. Twenty-eight recordings share identical
audio with another recording; the song split groups these globally, including duplicates with different
genre labels.

The first 20 segments per song provide 20,000 paper-protocol examples. The clip split uses
15,000/2,500/2,500 train/dev/test clips. The song split targets 800/100/100 songs overall, with 80/10/10 per
genre; seed 0 meets those counts exactly. When selected by the dev sweep, 13 additional segments per song
are added to training. Under the song split, these extra clips come only from training songs.

If the audio is not already at `data/raw/gtzan_qiuqiangkong/genres/<genre>/`, the first data cell does the
following:

1. downloads the archive (about 1.2 GB) from a pinned Hugging Face mirror;
2. checks its SHA-256;
3. extracts all ten genres;
4. converts the files to `.wav` in `data/interim/gtzan_wav/`, as the paper did.

Archive provenance, checksums and earlier dataset-audit notes are in [docs/dataset.md](docs/dataset.md).
The final notebook checks the full ten-genre dataset and split isolation. The audio is not committed to the
repository.

## Training and cache behaviour

In the final notebook, `REUSE_COMPLETED = True` reloads completed Part 1 weights/histories and appendix
results. `RETRAIN = False` reloads completed Part 2 experiments. To deliberately retrain the entire workflow,
set **`REUSE_COMPLETED = False` and `RETRAIN = True`**, then run from the top. This includes five seeds,
both split protocols and the parameter sweeps, and can take substantial time on CPU. Four-genre weights
cannot be reused for the ten-class task.

Part 2 cache keys include the ordered genre labels, audio digest, preprocessing, split protocol and experiment
configuration. Improved-model entries need weights, metrics and a training-history file before reuse.
Part 1 uses simpler filename/history checks, and appendix reuse checks the genre list: disable reuse if you
change Part 1's data or settings. An interrupted experiment restarts from its seed; completed experiments
are retained.

### Using Colab

1. Open `musical_genre_classification.ipynb` in Colab and select a GPU runtime for training.
2. Make the project directory available to the runtime, including both ten-genre artifact folders for reuse.
   Supply the audio too, or let the acquisition cell download it.
3. Use the runtime's CUDA-enabled PyTorch installation. The local pinned requirements describe the Mac
   environment; the GPU run's versions are in `colab_environment.json`.
4. Optionally set `GTZAN_PROJECT_ROOT` to the project directory. Set `GTZAN_REQUIRE_CUDA=1` before executing
   the notebook to fail immediately if a training runtime has no CUDA GPU. Leave it unset for local CPU use.
5. Run all cells and save the executed notebook and artifact folders back to persistent storage.

The recorded GPU continuation completed the remaining 18 experiments and all 33 cells in about
43 minutes while reusing earlier CPU work. This is a continuation time, not a full-training benchmark.
GPU support changes execution and memory transfers while retaining the existing model architectures and
experiment settings.

## Outputs

| Folder | Contents |
|---|---|
| `artifacts/paper_reproduction/gtzan_10_genres/` | Part 1 models, training histories, Figures 3/4/5, `results.json` |
| `artifacts/improvements/gtzan_10_genres/` | Part 2 cached models, dataset configuration, histories, `all_results.csv`, `comparison.csv`, comparison chart, best-model confusion matrix and PCA |

The earlier four-genre artifacts remain in the parent folders.

## Repository layout

| Path | What it is |
|---|---|
| `musical_genre_classification.ipynb` | **Final notebook:** all models, ten-genre results, plots and demo; CPU/GPU support and completed-result reuse |
| `Project_Writeup.pdf` | updated two-page writeup of the full ten-genre experiment |
| `Project_Writeup.docx` | editable Word version of the two-page writeup, with an editable results table and chart |
| `Musical_Genre_Classification_Final_10_Genres.pptx` | editable presentation using the reference deck's colours, fonts and format |
| `musical_genre_paper_reproduction.ipynb` | earlier CPU implementation; retained as a reference for architecture checks |
| `docs/reference_audit.md` | where the paper, poster and authors' code disagree |
| `docs/dataset.md` | dataset sources, checksums and audit results |
| `references/` | the paper, poster and assignment guidelines (read-only) |
| `scripts/`, `requirements-inspect.txt` | standalone dataset-inspection tools |
| `tests/` | dataset, ten-class model, split-isolation and CPU/GPU checkpoint checks |
| `data/metadata/` | lightweight dataset inventories produced by those tools |

The final notebook runs independently of `musical_genre_paper_reproduction.ipynb`, but the current test suite
still reads that reference notebook. Retain it unless those tests and documentation references are updated.

## Interpretation and limitations

The convolutional raw-audio model reaches 61.2% clip accuracy on unseen songs, and mel spectrograms raise
this to 73.4%. Their clip-split scores are 9.4 and 14.0 percentage points higher, respectively, so the easier
clip split should not be treated as evidence of the same accuracy on new songs. Song-level probability
aggregation further improves the final mel model to 82.6% on held-out songs.

Exact duplicate grouping reduces recording leakage in the song split. Artist overlap, other dataset faults
and performance beyond GTZAN are not resolved by this experiment. Five-seed standard deviations describe
run/split variability, not confidence intervals. The reconstruction-weight sweeps use one dev split; they
do not establish a general causal benefit for the autoencoder objective.

## Attribution

GTZAN was created by George Tzanetakis and collaborators (Tzanetakis & Cook, *Musical Genre Classification of
Audio Signals*, IEEE TSAP 2002). The paper and its code ([arjunsawknee/Genre-Extraction](https://github.com/arjunsawknee/Genre-Extraction))
belong to their authors; this repository reimplements the method in PyTorch and does not copy their code.
