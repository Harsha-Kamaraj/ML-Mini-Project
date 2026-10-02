# Dataset provenance and verified status

Acquired and audited on **2026-10-02**, local timezone Asia/Kolkata. This is the GTZAN **Genre Collection audio**, not music/speech data or extracted features. Both downloaded sources and their raw files are preserved; no recording was fabricated, padded, repaired or silently dropped.

## Complete source used by the notebook

- [Qiuqiang Kong GTZAN archive](https://huggingface.co/datasets/qiuqiangkong/gtzan/resolve/9df3f542ce11be457c4d4eeb81d9755349f72012/genres.tar.gz?download=true), documented by the maintainer's [audio project downloader](https://github.com/qiuqiangkong/audio_understanding/blob/main/scripts/download_gtzan.sh).
- Source revision: `9df3f542ce11be457c4d4eeb81d9755349f72012`.
- Local archive: `data/downloads/genres-qiuqiangkong.tar.gz`; **1,225,573,614 bytes**, verified SHA-256 `74a8901dfdd256ac13cc6cd529f36aaa156a4d304447f970c4dc87d7cb3fd957` matches source headers.
- Raw location: `data/raw/gtzan_qiuqiangkong/genres/<genre>/<genre>.NNNNN.au`.
- Full audit: **1,000 files, 1,000 successful decodes**, ten genre folders of 100 each. All are native **22,050 Hz, one channel**. Durations range **29.931973–30.648889 seconds**; none outside the audit's 29–31-second advisory range. No missing expected IDs or nonfinite-sample failures.

| Main genre | Found | Fully decoded | Duration range (seconds) |
|---|---:|---:|---|
| classical | 100 | 100 | 29.992925–30.488980 |
| jazz | 100 | 100 | 30.007982–30.480726 |
| metal | 100 | 100 | 30.000181–30.013333 |
| pop | 100 | 100 | 30.000181–30.000181 |
| **Selected total** | **400** | **400** | |

The verified AU archive is used consistently for all recordings; this is not a mixed-source single-file replacement. `jazz.00054.au` decodes to **662,112 frames**, resolving the unreadable-file blocker in the first mirror. A direct comparison of decoded float32 waveforms across the selected subset found **399 exact matches**, zero differences among comparable files, and one uncomparable unreadable WAV (`jazz.00054`). See [source comparison](../data/metadata/source_comparison.json).

The AU audit detects seven exact-file duplicate pairs (14 files); all remain included. The notebook additionally hashes decoded waveforms and verifies **28 duplicate-content files**, catching audio with different file headers. The earlier WAV audit identified fourteen exact-file pairs, including nine pairs within the selected genres and one metal/rock pair. Recording IDs are not evidence of unique musical content. Report recording/content overlap and do not call a filename-disjoint split fully leakage-free.

Lightweight evidence: [AU inventory](../data/metadata/gtzan_qiuqiangkong/gtzan_inventory.csv), [AU summary](../data/metadata/gtzan_qiuqiangkong/gtzan_summary.json). The notebook independently re-audits its configurable source and saves a run-local manifest.

## First mirror: retained with an explicit failure

[Marsyas dataset card](https://huggingface.co/datasets/marsyas/gtzan) and [pinned archive](https://huggingface.co/datasets/marsyas/gtzan/resolve/d2146561ecc7df707d9e6b8318885fe6a39668a2/data/genres.tar.gz?download=true).

- Revision: `d2146561ecc7df707d9e6b8318885fe6a39668a2`.
- Archive: `data/downloads/genres.tar.gz`, **1,226,192,050 bytes**, SHA-256 `b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531`, verified against source headers.
- Extracted to `data/raw/gtzan/`; 2,013 files comprise 1,000 recording files, 1,012 confirmed AppleDouble metadata files, and one `.DS_Store`.
- **999/1,000 audio files decode**, **399/400** in the selected subset. `genres/jazz/jazz.00054.wav` is 1,207,592 bytes, has no recognizable WAV header, and fails SoundFile decoding. Its rate/channels/duration are unknown rather than inferred from its name.
- Confirmed AppleDouble files require both a `._` name and signature `0x00051607`; they are preserved and enumerated, not counted as audio failures. No problematic real audio is hidden by this rule.
- Repeated safe extraction verified all 2,013 existing files identical and overwrote none. [First inventory](../data/metadata/gtzan_inventory.csv) and [summary](../data/metadata/gtzan_summary.json) retain the failure.

## Attribution and terms

GTZAN was created by George Tzanetakis and collaborators. The Marsyas card cites Tzanetakis, Essl and Cook, *Automatic Musical Genre Classification of Audio Signals*, ISMIR 2001. The supplied paper cites Tzanetakis et al., *Musical Genre Classification of Audio Signals*, 2002.

The Marsyas mirror does not provide an audio license (licensing information is marked missing); the alternate mirror supplies no dataset card or stated audio license. No broader usage permission is inferred from a software-loader license. Raw audio/archives remain local and Git-ignored; attribution and lightweight manifests are tracked.

Original historical location: `http://opihi.cs.uvic.ca/sound/genres.tar.gz`; [official torchaudio documentation](https://docs.pytorch.org/audio/2.1.2/generated/torchaudio.datasets.GTZAN.html) reports that URL unavailable since October 2022. These mirror archive hashes are not claimed to be the original archive's hash.

## Re-run

The self-contained `musical_genre_replication.ipynb` has download, safe extraction, inspection and processing code. Select the `.venv` kernel and execute Phase 1. It reuses existing files, records failures, and refuses to proceed to the 8,000-example experiment unless all 400 selected recordings are usable. The earlier standalone inspection scripts are retained as existing work; the notebook does not import them.
