# Project summary: what we did and what we found

Team: Harsha and Sunil · UE24CS352A Machine Learning mini-project

All numbers here come from running
[musical_genre_paper_reproduction.ipynb](musical_genre_paper_reproduction.ipynb). The paper's own numbers are
always marked "paper".

---

## 1. The problem in one paragraph

Can a computer learn what makes a piece of music *jazz* rather than *metal* just by listening to the raw sound,
without any hand-made audio features? The paper we were given, *Latent Feature Extraction for Musical Genres from
Raw Audio* (Sawhney, Vasavada & Wang), tries to do this. It trains neural networks on short clips of raw audio from
four genres (classical, jazz, metal, pop). The networks have to:
1. squeeze each clip into a short list of 64 numbers, called an **encoding** or **embedding**;
2. predict the genre from that encoding.

The paper reports **65.3% test accuracy**. Our job had two parts:
1. **Reproduce** the paper as faithfully as possible.
2. **Improve** on it.

---

## 2. The data

- **Dataset:** GTZAN, the standard music-genre dataset of 1,000 thirty-second songs in 10 genres. Like the paper,
  we use only 4 genres × 100 songs = **400 songs**.
- **Source:** the original download link has been offline since 2022, so we used a mirror. We checked that the file
  is intact (checksum) and that all 400 songs decode correctly. An earlier mirror had a broken jazz file, so we
  switched sources (details in [docs/dataset.md](docs/dataset.md)).
- **Turning songs into examples, as the paper did:**
  - Load each song with LibROSA at 22,050 samples per second.
  - Cut each song into **20 pieces of 20,000 samples** each.
  - Shrink each piece by averaging every 40 neighbouring samples ("average pooling"). Each piece becomes a list of
    **500 numbers**.
  - Result: 400 songs × 20 pieces = **8,000 examples**, 2,000 per genre.
- **Split:** shuffle the 8,000 examples randomly into 6,000 for training, 1,000 for development ("dev", used for
  tuning) and 1,000 for testing.

One small contradiction in the paper: it says the pieces are "one second" long, but 20,000 samples at 22,050 per
second is 0.907 seconds. We followed the numbers (20,000 → 500), because those are what produce the paper's 8,000
examples.

---

## 3. Part 1: rebuilding the paper

### 3.1 The three models from the paper

| Model | What it does |
|---|---|
| **Two-layer neural network** (baseline) | 500 numbers → 128 hidden units → 4 genre scores. A plain classifier. |
| **Vanilla autoencoder** | Compresses 500 numbers down to 64 (the encoder), then tries to rebuild the original 500 (the decoder). It never sees the genre labels. |
| **Deep Softmax Autoencoder** (the paper's main model) | The same autoencoder, plus a small classifier (64 → 32 → 16 → 4) on the 64-number encoding. It is trained to rebuild the audio *and* predict the genre: loss = 0.9 × reconstruction error + 0.1 × classification error. |

Training settings from the paper: the Adam optimiser, learning rate 0.0001, batches of 512, and dropout with a
"keep probability" of 0.9.

### 3.2 Following the paper "exactly"

The paper leaves out several details: which activation functions the hidden layers use, where the dropout goes,
how the weights start, and how many epochs to train. Our rule:

> **The paper text comes first. Where the paper says nothing, use the authors' own code (linked in the paper).
> Where the two disagree, the paper wins.**

The first cell of the notebook has a table of every setting and where it came from. The paper and the code
disagreed in three places, and we used the paper's value each time:
- **Loss weights:** 0.9/0.1 in the paper; a different weighting in the code.
- **Keep probability:** 0.9 in the paper; 0.8 in the code.
- **Epochs for the main model:** about 500 in the paper's Figure 3; 200 in the code.

### 3.3 What we got (seed 0)

| Classification accuracy | Train | Dev | Test |
|---|---|---|---|
| Two-layer NN, paper | 52.0% | 38.1% | 36.4% |
| Two-layer NN, ours | 53.1% | 40.0% | 38.9% |
| Deep Softmax AE, paper | 94.9% | 64.1% | 65.3% |
| Deep Softmax AE, ours | 96.2% | 58.7% | 57.7% |

- **Baseline:** matches the paper well.
- **Main model, train accuracy:** matches.
- **Main model, dev/test accuracy:** came out lower.
- **Same patterns as the paper:** classical is the easiest genre, and the most common mistakes are metal↔pop and
  classical↔jazz. Our PCA pictures (Figures 5 and 6) look much like the paper's.

### 3.4 Why our numbers differ from the paper's

We didn't guess at reasons; we tested them in the notebook's appendix.

1. **Luck of the split.** The paper never says which clips went into the test set, and the authors' code shuffles
   without a fixed seed. We re-ran everything with 5 different seeds, each giving a different random split and
   different starting weights. Test accuracy for the main model ranged from **57.7% to 63.9%** (mean 61.0%). So
   about 6 points of difference comes from luck alone. Even the best seed stayed below the paper's 65.3%.
2. **The authors' code does not match their paper.** We also ran the main model with the code's exact settings. It
   reached only **85.4% train accuracy** on average, nowhere near the paper's 94.9%. So the published table was
   probably not produced by the code as it is on GitHub.
3. **The paper's autoencoder is broken.** The decoder's last layer uses a ReLU, which can only output numbers ≥ 0,
   but sound waves go negative half the time. During training this layer "died": it outputs **0 for every input**,
   which we measured. So:
   - the vanilla autoencoder reconstructs nothing;
   - in the main model the reconstruction part of the loss never changes and does nothing. The "Deep Softmax
     Autoencoder" quietly becomes just a deep classifier.

   The authors' own code comment gives a reconstruction error that matches an all-zero output, so their model most
   likely had the same problem. We kept the architecture exactly as published and reported the problem rather than
   silently fixing it.
4. **Things we cannot test:** unreported tuning (the paper says "we ran consistent experiments to tune"), the
   authors possibly reporting their best run, and small numerical differences between TensorFlow (theirs) and
   PyTorch (ours).

---

## 4. Part 2: building better models

### 4.1 Two ways of testing

The paper's random split has a hidden weakness. Every song is chopped into 20 clips and the clips are shuffled
individually. So the test set contains clips from songs whose *other* clips were used for training: in our split,
362 songs have clips on both sides. The model can partly recognise *the song* rather than *the genre*.

We therefore report every model on **two splits**:
- **Clip split:** the paper's split. Use this to compare with the paper.
- **Song split:** whole songs go to train (320 songs), dev (40) or test (40). The test songs are completely new to
  the model. GTZAN also contains 18 songs that are exact copies of other songs, and we keep each copy in the same
  partition as its original. This is the honest measure of how well a model works on new music.

### 4.2 Fair-testing rules

- **5 seeds for every model on both splits.** We report the average ± spread, never one lucky run.
- **Every choice is made on the dev set.** The test set is only used to report results, never to choose anything.
  Each choice (for example how much weight to give reconstruction, or how much pooling to use) is made by a small
  experiment ("sweep") on dev, and the code picks the winner automatically.
- **One change per step,** so we can see how much each change helps.
- **A song-level score.** We also report song accuracy: average the model's predictions over all clips of a test
  song and check whether the song gets the right genre.

### 4.3 The four improvement steps

**Step 1: fix the paper's model.** We keep the same input and layer sizes and change only the following:
- the decoder's last layer can now output negative numbers, so it can no longer die;
- the input is rescaled to a standard range;
- the training examples are reshuffled every epoch;
- the best epoch on dev is kept, and training stops when dev stops improving;
- the reconstruction weight is re-tuned.

*Result:* almost no change (61.0% → 61.5% test). The bugs were real, but they weren't what held the model back.

**Step 2: give the model more of the sound.** Averaging every 40 samples throws away everything above about
275 Hz, which is most of what makes a cymbal, guitar or voice sound the way it does. We also had the model use the
whole 30 seconds of each song instead of only the first 18. The dev sweep chose to keep the paper's pooling of 40,
because the plain network didn't benefit from wider input, but to use the extra audio.

*Result:* a small gain (64.3% test, roughly equal to the paper).

**Step 3: a convolutional Deep Softmax Autoencoder on raw audio (our main contribution).** The paper's layers are
"fully connected". They treat position 1 and position 2 of the clip as unrelated, so a drum hit early in the clip
and the same drum hit later look like completely different things. **Convolutional** layers slide the same small
pattern detectors along the whole clip, so they recognise a sound wherever it appears.

We kept the paper's idea: raw audio in, a 64-number encoding, a decoder that rebuilds the audio, and the same
32 → 16 → 4 genre classifier. Only the encoder and decoder became convolutional. The dev sweeps chose:
- light pooling of 4, keeping sounds up to about 2,750 Hz;
- a reconstruction weight of 0.

*Result:* **87.2% test accuracy, about 22 points better than the paper.** On completely unseen songs it gets
83.2%, and 91% of test songs get the right genre.

**Step 4: a mel-spectrogram Deep Softmax Autoencoder (extra).** A spectrogram is a picture of a sound: time on one
axis, pitch on the other, brightness for loudness. The "mel" version spaces the pitches the way human hearing does.
The paper deliberately avoided this kind of "feature engineering", but lists it as future work. We built the same
Deep Softmax Autoencoder design with 2-D convolutions on these pictures. The dev sweep chose a reconstruction weight
of 0.1.

*Result:* **97.7% test accuracy, and 94.5% on unseen songs.** This is the best model, but it breaks the paper's
"raw audio only" rule, so we present it as an extra.

### 4.4 The full comparison (test accuracy, mean ± std over 5 seeds)

| Model | Clip split (comparable with the paper) | Song split (unseen songs) |
|---|---|---|
| Paper, two-layer NN (quoted) | 36.4% | — |
| Paper, Deep Softmax AE (quoted) | 65.3% | — |
| Paper's two-layer NN, our run | 39.0% ± 1.4% | 38.1% ± 2.9% |
| Paper's Deep Softmax AE, our run | 61.0% ± 2.2% | 58.4% ± 3.1% |
| Step 1: fixed Deep Softmax AE | 61.5% ± 1.2% | 60.3% ± 3.2% |
| Step 2: + more of the signal | 64.3% ± 0.6% | 62.9% ± 2.3% |
| **Step 3: Conv Deep Softmax AE (raw audio)** | **87.2% ± 1.7%** | **83.2% ± 2.9%** |
| Step 4: Mel Deep Softmax AE (extra) | 97.7% ± 0.8% | 94.5% ± 2.9% |

The notebook's final table also gives train, dev, macro-F1 and song-level accuracy for every row.

### 4.5 The best model in the paper's format

The best model (picked by dev accuracy) is the mel-spectrogram model. On the test set (clip split, seed 0) its
per-genre F1 scores are:

| Genre | Ours | Paper |
|---|---|---|
| Classical | 0.97 | 0.78 |
| Jazz | 0.96 | 0.62 |
| Metal | 0.99 | 0.53 |
| Pop | 0.99 | 0.64 |

In its PCA pictures the four genres form four clearly separate groups. In the paper's pictures they overlap
heavily.

---

## 5. What we learned

1. **A reproduction can be faithful and still not match.** Following the paper gave the baseline almost exactly,
   but the main model came out about 4 points lower on average. The gap comes from missing details (the split, the
   tuning), from the authors' code not matching their paper, and from luck. That is why we report 5 seeds and not
   one run.
2. **The paper's autoencoder idea was never really tested.** Its decoder died, so the "reconstruction" half did
   nothing. When we fixed it and let the dev set decide:
   - in the convolutional models, all reconstruction weights scored within 0.2 points of each other on dev;
   - in Step 1 a small weight (0.1) won the sweep, but Step 1's final accuracy barely moved.

   The accuracy gains came from better architecture and better input, not from the autoencoder.
3. **Architecture matters most.** Changing fully connected layers to convolutional ones, on the same raw audio, gave
   the biggest jump (64% → 87%).
4. **Input representation matters next.** Spectrograms beat raw audio by about 10 more points. Hand-made features
   still help when data is small (only 400 songs).
5. **The paper's split overstates performance.** Every model scores lower on unseen songs (song split) than on the
   paper's clip split, by about 1–4 points. The gap is largest for the convolutional models.

### Limitations to mention honestly

- GTZAN is small (400 songs used) and has known problems: exact duplicates, which we handled, and possible
  near-duplicates and mislabelled songs, which we can't detect.
- The song split's test set has only 40 songs, so its numbers vary more between seeds.
- All training was on a laptop CPU, which limited how many settings we could try.

---

## 6. Files

| File | What's in it |
|---|---|
| `musical_genre_paper_reproduction.ipynb` | everything: Part 1 (reproduction + why numbers differ), Part 2 (improvements + final comparison + demo) |
| `README.md` | setup and how to run |
| `plan.md` | project plan and decisions |
| `docs/reference_audit.md`, `docs/dataset.md` | paper-vs-code differences; dataset sources and checks |
| `artifacts/` (not in Git) | trained models, figures, cached results |

A full **Run All** takes about 4 minutes with the cached Part 2 results, or 2–3 hours to retrain everything.

## 7. What is still left for the assignment

- the two-page write-up (PDF);
- the slide deck;
- rehearsing the live demo and Q&A;
- confirming the GitHub repository is private and shared with the faculty and TAs.
