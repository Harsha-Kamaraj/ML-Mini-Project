# ML-Mini-Project
Latent Feature Extraction for Musical Genres from Raw Audio

Team: Harsha and Sunil. This project reproduces the supplied paper with raw GTZAN
waveforms. See [plan.md](plan.md) for source-specific requirements, balanced task
ownership, approved choices, evidence, and phase checkpoints; see the
[reference audit](docs/reference_audit.md) for paper/poster/author-code conflicts.

The complete implementation is in
[musical_genre_replication.ipynb](musical_genre_replication.ipynb). All project logic
is inside the notebook: acquisition, verification, preprocessing, shared splits,
three PyTorch models, training, checkpoint loading, evaluation, PCA, and a demo.
It does not import the earlier standalone inspection scripts, which are preserved.
References and any synced `sources/` directory remain read-only.

[musical_genre_paper_reproduction.ipynb](musical_genre_paper_reproduction.ipynb) contains both parts of the project:
- **Part 1 — Paper reproduction.** It follows the paper text first, uses the authors' linked code only where the
  paper gives no value, and documents the source of every setting. Outputs go to `artifacts/paper_reproduction/`.
- **Part 2 — Improvements.** Fixed, wider-input, convolutional (raw audio) and mel-spectrogram Deep Softmax
  Autoencoders, each evaluated over 5 seeds on the paper's clip split and on a song-disjoint split. It ends with one
  comparison table for all models. Trained models and results are cached in `artifacts/improvements/`.

Run All takes about 4 minutes when the Part 2 cache exists. With an empty cache, or with `RETRAIN = True`, training
Part 2 from scratch takes about 2–3 hours on a laptop CPU.

## Start the notebook in VS Code

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
```

Open the notebook and select `.venv/bin/python` as the kernel (Windows:
`.venv\Scripts\python.exe`). Run cells in order. Dependencies can also be installed
from the package list inside the notebook; the requirements file is optional.

The default `CFG["mode"] = "smoke"` runs one epoch per model on 256 training and
128 validation examples, checks the complete workflow, and leaves test unevaluated.
These are real executed diagnostics, **not full reproduction results**. To train
the full experiments use `mode="full"` (200 baseline, 400 AE, 200 joint epochs).
Keep `evaluate_test=False` until choices are fixed; then set it true and rerun
with checkpoint reuse enabled. Full runs and smoke runs use separate directories.

The default dataset is the separately preserved AU archive in
`data/raw/gtzan_qiuqiangkong/`: **1,000/1,000 recordings decode, including all
400 selected recordings**. Download/source links, checksums and the comparison
with the earlier failed WAV mirror are in the notebook and
[dataset provenance](docs/dataset.md). It creates 8,000 × 500 inputs and preserves
recording/segment metadata and split indices. Caches, checkpoints and figures go
under Git-ignored `artifacts/notebook/`.

## Earlier standalone inspection setup (optional)

Run from the repository root. Python 3.12+ is required by the extraction script;
the executed inspection environment is Python 3.14.6 on macOS arm64. The minimal
dependencies are pinned; this is not yet a validated training environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-inspect.txt
.venv/bin/python -m pip check
```

Expected: imports for NumPy/SoundFile work and pip reports no broken requirements.
Audio remains in its native format, sample rate, and channel count during inspection.

## Earlier WAV mirror acquisition (optional; known unreadable file)

Source: [Marsyas GTZAN audio mirror on Hugging Face](https://huggingface.co/datasets/marsyas/gtzan).
This is the Genre Collection audio archive. No Hugging Face login or dataset loader
is needed. The mirror does not specify an audio license; see
[dataset provenance](docs/dataset.md) for attribution, checksum and verification.

For a new or interrupted download (about 1.23 GB compressed):

```bash
mkdir -p data/downloads data/raw/gtzan
curl -L --fail --continue-at - --retry 5 \
  --output data/downloads/genres.tar.gz.part \
  'https://huggingface.co/datasets/marsyas/gtzan/resolve/d2146561ecc7df707d9e6b8318885fe6a39668a2/data/genres.tar.gz?download=true'
.venv/bin/python scripts/extract_gtzan.py
```

The extractor verifies size and SHA-256, rejects unsafe paths and links, refuses
to overwrite differing raw files, and renames the verified `.part` archive to
`genres.tar.gz`. Expected checksum:
`b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531`.
Extracted audio is under `data/raw/gtzan/genres/<genre>/`.

If the archive already exists, skip downloading and use:

```bash
.venv/bin/python scripts/extract_gtzan.py --archive data/downloads/genres.tar.gz
```

Existing identical files are retained. If downloading manually, save the pinned
URL above as `data/downloads/genres.tar.gz` and run that command. If using an
existing dataset elsewhere, pass its directory to `--root` below; do not copy
feature CSVs or spectrogram images as substitutes for audio.

## Verify the earlier WAV mirror

```bash
.venv/bin/python scripts/inventory_gtzan.py \
  --root data/raw/gtzan --output-dir data/metadata
.venv/bin/python -m unittest discover -s tests -v
```

The inventory reads every recording to the end. It produces:

- `data/metadata/gtzan_inventory.csv`: paths, stable recording IDs, genres,
  headers, full-decode status, durations, file hashes and amplitude statistics.
- `data/metadata/gtzan_summary.json`: actual counts, missing/duplicate IDs,
  failures, advisory anomalies, decoder versions and archive metadata files.
- `data/metadata/gtzan_waveforms.svg`: native raw waveform previews for one
  recording in each selected genre; this is inspection, not training segmentation.

An exit code of 2 means verification found a hard issue, while preserving the
full manifest. **This downloaded mirror is expected to exit 2**: 1,000 recording
files exist, but `jazz.00054.wav` fails decoding. The selected four genres contain
400 recording files and 399 currently decodable recordings. No file has been
excluded, repaired, or replaced. Review [dataset status](docs/dataset.md) before
preprocessing; do not treat the failure as a successful 400-recording audit.

Raw audio, archives, `.venv`, caches and large generated artifacts are Git-ignored.
Keep code, requirements, documentation and lightweight manifests in Git. Do not
publish the dataset with the source repository. Repository privacy and faculty
access remain to be verified as required by the assignment.

## Reproduction policy and remaining work

The user approved twenty consecutive 20,000-sample chunks per recording, pooled
in groups of 40. At 22,050 Hz these are approximately 0.907-second clips, an
explicit interpretation based on author code. The complete notebook now implements
this approved policy and uses the fully readable AU source. Main split sizes and
loss/dropout settings follow the paper; source-code conflicts are documented.
Full experimental training/test evaluation, human review, the final PDF/slides,
faculty repository access, and demonstration rehearsal remain outstanding. The
assignment's one-page/two-page contradiction still needs faculty clarification.
