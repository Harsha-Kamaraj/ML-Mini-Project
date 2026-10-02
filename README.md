# ML-Mini-Project
Latent Feature Extraction for Musical Genres from Raw Audio

Team: Harsha and Sunil. This project reproduces the supplied paper with raw GTZAN
waveforms. See [plan.md](plan.md) for source-specific requirements, balanced task
ownership, approved choices, evidence, and phase checkpoints; see the
[reference audit](docs/reference_audit.md) for paper/poster/author-code conflicts.

Current scope: reference review and dataset inspection only. No preprocessing,
models, training, or reproduced metrics have been implemented. The intended
implementation uses Python and PyTorch; training packages will be added in their
phase. References and any synced `sources/` directory are read-only.

## Inspection setup

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

## Obtain the audio

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

## Verify raw recordings

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

## Next phase

The user approved twenty consecutive 20,000-sample chunks per recording, pooled
in groups of 40. At 22,050 Hz these are approximately 0.907-second clips, an
explicit interpretation based on author code. Phase 2 still requires the user's
continuation and resolution of the unreadable recording. Main split sizes and
loss/dropout settings will follow the paper; source-code conflicts are documented.
