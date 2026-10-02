# Reference and author-code audit

Reviewed 2026-10-02 by Codex as automated assistance; no human review or experimental reproduction is claimed. Local PDFs were read in full and visually inspected. The paper remains the primary mathematical specification.

## Immutable local references

| File | Pages | SHA-256 |
|---|---:|---|
| `references/musical_genre.pdf` | 6 | `723b2bb5334279cf4eb7af4f3a13f43901a8c38ab92e5b9541730d26f7e97e64` |
| `references/musical_genre_poster.pdf` | 1 | `5688248708c3fbc0bce105b959574bc1fe1507b2609eefe90bdc9d00100bcd5d` |
| `references/Guidelines and Instructions_Mini Project Assignment.pdf` | 2 | `1d6690cce8cbd240e253b4831171d6af625a927d0948f7cd980415d65fb445eb` |

See [plan.md](../plan.md) for page-specific requirements, assignment dates and task ownership. No separate assignment master list/team list/review timetable or `AGENTS.md` was found.

## Author repository evidence

The paper p.6 links [arjunsawknee/Genre-Extraction](https://github.com/arjunsawknee/Genre-Extraction). The inspected revision is [00b1802e24d1ab6fb3135b24a44976a51033ed24](https://github.com/arjunsawknee/Genre-Extraction/commit/00b1802e24d1ab6fb3135b24a44976a51033ed24), dated 2018-12-09. Attribution: paper authors Arjun Sawhney, Vrinda Vasavada and Woody Wang. Code was read for evidence, not executed or copied into this project. An archived checkout/source downloads remain in temporary storage only. This revision does not prove how the published results were generated.

### Segmentation and split

[Joint loader, lines 28–69](https://github.com/arjunsawknee/Genre-Extraction/blob/00b1802e24d1ab6fb3135b24a44976a51033ed24/softmaxautoencoder.py#L28-L69) uses WAV files in classical/jazz/metal/pop order and LibROSA without an explicit sampling-rate argument or version pin. It truncates to 600,000 samples, reshapes into 15,000 groups of 40, averages, then takes the first twenty 500-value chunks. Thus only the first 400,000 raw samples contribute examples. Assuming the paper's 22,050 Hz, each chunk is 0.907029 seconds and coverage is 18.140590 seconds per recording. The preliminary reshape still requires 600,000 samples. This accounts for 400 × 20 = 8,000 examples by inference from source; it does not make them literal one-second clips.

The [joint main function](https://github.com/arjunsawknee/Genre-Extraction/blob/00b1802e24d1ab6fb3135b24a44976a51033ed24/softmaxautoencoder.py#L229-L252) shuffles individual examples, unseeded, and splits 6,000 train / 2,000 dev. Baseline code does the same. The downloaded `labels_dev.csv` has 2,000 rows: classical 468, jazz 549, metal 481, pop 502. This supports the poster's development-size label, but conflicts with the paper's 6,000/1,000/1,000 protocol.

Recording IDs are discarded. Actual split membership and overlap cannot be reconstructed reliably. A clip-level random split allows clips of the same recording in different partitions; this is not evidence of leakage-free evaluation. Our main protocol will retain paper sizes, document a new seed/membership, and measure recording overlap. A recording-disjoint comparison is an additional experiment.

The [vanilla AE main](https://github.com/arjunsawknee/Genre-Extraction/blob/00b1802e24d1ab6fb3135b24a44976a51033ed24/encoder.py#L95-L102) trains on all 8,000 loaded examples; inspected test scripts also currently load the full dataset, with saved-dev loading commented out. We will train every model on training data only and reserve test evaluation. The older `process_data.py` contains ten-genre/MFCC debugging and exits before segmentation; it is not a specification for this raw-waveform reproduction.

### Architecture and defaults

[Joint forward pass/parameters](https://github.com/arjunsawknee/Genre-Extraction/blob/00b1802e24d1ab6fb3135b24a44976a51033ed24/softmaxautoencoder.py#L79-L135) matches the paper's widths. Source-only settings:

| Component | Evidence from author code, not fully specified in paper |
|---|---|
| Encoder activations | sigmoid → tanh → ReLU → ReLU |
| Decoder activations | sigmoid → sigmoid → ReLU → ReLU |
| Joint head | tanh → tanh → linear logits |
| Initialization | Xavier weights and zero biases |
| Dropout placement | After 192-unit encoder and decoder activations; none in head |
| Actual AE/joint keep probability | 0.8 in training, conflicting with paper's 0.9 |
| Baseline | tanh hidden, linear output, no dropout call |
| Duration | Baseline/joint 200 epochs; vanilla AE 400; paper plot reaches about 500 |
| Shared settings | Adam, lr 0.0001, batch 512; short final batch retained; no explicit seed |

Decoder ReLU prevents negative reconstruction of signed waveforms. Using it preserves source behavior; replacing it with a linear output is a consequential new choice. Document this before Phase 4. No model choice has been implemented. Prefer paper keep probability 0.9, source-informed placement/activation/initialization where appropriate, and an explicitly agreed training budget; do not infer a universal 500-epoch rule from a plot.

### Loss conflict

[Joint loss, line 153](https://github.com/arjunsawknee/Genre-Extraction/blob/00b1802e24d1ab6fb3135b24a44976a51033ed24/softmaxautoencoder.py#L153) is elementwise mean squared error plus 0.1 times batch-mean cross-entropy. The paper Eq.3 instead gives 0.9 times squared L2 error plus 0.1 times cross-entropy per example. With batch averaging, the paper-oriented implementation is:

```text
reconstruction = mean_over_examples(sum_over_500_values(squared_error))
classification = mean_over_examples(cross_entropy)
total = 0.9 * reconstruction + 0.1 * classification
```

The source divides reconstruction by 500 and omits the 0.9 coefficient. Relative classification/reconstruction weighting therefore differs by a factor of 450. Vanilla AE source also uses elementwise mean MSE. Preserve paper mathematics for the primary experiment; never describe source mean MSE at fixed weights as equivalent. A source-loss variant would be a separately approved additional experiment.

## Approved waveform decision and remaining choices

On 2026-10-02, Harsha explicitly selected the code-backed policy: twenty consecutive 20,000-sample chunks per recording with group-of-40 averaging, labeled approximately 0.907 seconds at 22,050 Hz. Alternative literal seconds resampled to 20 kHz, or 22,050-to-500 variable-width pooling, would introduce preprocessing absent from inspected source and were not selected. Approval resolves the waveform/count interpretation, but does not authorize advancing past the requested Phase 1 pause. No preprocessing has been implemented.

Paper loss/split/dropout precedence is already supported by the user's instructions. Missing seed, initialization/defaults and exact durations will be stated explicitly when their phase is authorized. No published metrics are acceptance targets.
