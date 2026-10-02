# Musical genre latent-feature reproduction plan

Updated: 2026-10-02 (Asia/Kolkata). Team: Harsha and Sunil.

## Problem and intended outputs

Learn a compact representation of musical genre directly from raw audio amplitudes. Compare a simple genre classifier, a reconstruction-only autoencoder, and an autoencoder trained jointly to reconstruct audio and classify genre. An embedding is the short vector the encoder learns; useful embeddings should preserve information and may group similar genres together.

Deliver working Python/PyTorch code, reproducible data/split manifests and configurations, measured classification results, reconstruction/training curves, PCA visualizations, a private GitHub repository with setup instructions, a PDF write-up, slides, and a live demonstration. Use only classical, jazz, metal, and pop for the main experiment. Do not substitute engineered features or pretrained embeddings. Music generation, style transfer, extra genres, and VAE extensions are outside the initial scope.

## Source hierarchy and workspace baseline

1. [Assignment guidelines](references/Guidelines%20and%20Instructions_Mini%20Project%20Assignment.pdf): mandatory deliverables and dates (2 pages).
2. [Paper](references/musical_genre.pdf): primary technical specification (6 pages), Arjun Sawhney, Vrinda Vasavada, Woody Wang, “Latent Feature Extraction for Musical Genres from Raw Audio.”
3. [Poster](references/musical_genre_poster.pdf): architecture/results cross-check (1 page).
4. [Paper-linked author code](https://github.com/arjunsawknee/Genre-Extraction): supplemental evidence; conflicts must be logged, not silently preferred over the paper.

Initial inspection found only the three PDFs and the existing two-line README, a clean Git working tree, and no code, dataset, configured dataset environment variable, or `sources/` directory. No `AGENTS.md` was found in the workspace or its ancestor directories. No separate team list, problem master list, or review schedule was present. Preserve all references and any future synced `sources/` directory as read-only. Preserve the existing README title and project identity.

The complete text of all PDFs was extracted and every page rendered and visually inspected, including diagram labels absent from extracted text. No model or preprocessing implementation exists yet.

## Mandatory assignment checklist

These are deliverable checks; accountable task owners are assigned in Phase 7 below.

- [ ] Private GitHub repository containing all project code; verify privacy and faculty/assigned TA access (guidelines p.1, §3A; do not claim access before verification).
- [ ] README with actual setup and run instructions (p.1, §3A).
- [ ] PDF write-up covering problem, dataset, approach, implementation overview, and conclusions (pp.1–2, §3B).
- [ ] Faculty clarification of write-up length: heading says “One-Page Write-up,” body says “two-page summary” (p.1). Prepare adaptable content; do not claim a resolved limit.
- [ ] Presentation slide deck and live demonstration of working code at mandatory review (p.2, §4).
- [ ] Both members can explain methodology/code and answer panel questions; maintain truthful individual contribution records (p.2, §§4–5).
- [ ] Confirm assigned problem serial number and review slot from missing faculty lists (p.1, §2; p.2, §4).
- [ ] Submit deliverables by **Saturday, October 10, 2026, 11:59 PM**; guideline does not specify timezone, use local Asia/Kolkata provisionally and confirm if necessary.

Start: Tuesday, September 29, 2026; stated duration: one week. Reviews: **Monday, October 5 through Friday, October 9, 2026** (p.1, §1). Assessment: 10 marks, including functionality, maintenance, write-up, presentation, demo, Q&A, and individual contribution (p.2, §5). No language/framework restriction is stated.

## Technical specification with evidence

| Area | Explicit reference requirement | Location / limits |
|---|---|---|
| Data | GTZAN Genre Collection: 1,000 approximately 30-second recordings, 10 genres; select classical/jazz/metal/pop | Paper §§2–3, pp.1–2; poster dataset panel explicitly gives 400 source songs |
| Input | Raw amplitudes, conversion to WAV and LibROSA loading in reference workflow; one-second segments; stated 22.05 kHz | Paper §3, p.2; poster dataset panel; sample-count contradiction unresolved |
| Pooling | 20,000 waveform values → average pooling with pool size 40 → 500 inputs | Paper §3, p.2 and Fig.2, p.3; no MFCC, mel, feature CSV, or pretrained embedding replacement |
| Counts/split | 8,000 total examples balanced across genres; random 6,000/1,000/1,000 train/dev/test split | Paper §3, p.2 and Table 1, p.3; no explicit seed, recording grouping, stratification, or segment selection |
| Classifier | 500 → 128 (tanh) → 4; cross-entropy | Paper §4.1, p.2, Eq.1; implement four raw logits, stable cross-entropy handles softmax |
| Vanilla AE | Encoder 500 → 256 → 192 → 128 → 64; decoder 64 → 128 → 192 → 256 → 500 | Paper §4.2, p.2 says use extended deeper baseline, top half of Fig.2, p.3; not the abandoned single-hidden-layer AE |
| AE objective | Squared L2 reconstruction error per example | Paper §4.2, p.2, Eq.2 |
| Joint model | Same AE with embedding classifier 64 → 32 → 16 → 4; extract 64-dimensional bottleneck and four pre-softmax logits | Paper §4.3/Fig.2, pp.2–3 and §5.2, pp.4–5 |
| Joint loss | `gamma * reconstruction + (1 - gamma) * cross_entropy`; gamma = 0.9 | Paper Eq.3/§4.3, p.3; equation label says reconstruction although expression is joint |
| Optimizer | Adam; learning rate 0.0001; batch size 512 | Paper Fig.2, p.3; poster architecture. Figure annotates final model; applying identically to both baselines is a documented controlled-comparison choice |
| Dropout | Between layers, final keep probability 0.9 (PyTorch dropout probability 0.1) | Paper §5.1, p.4; exact layers/model coverage not specified |
| Unspecified settings | AE/classifier-branch activations, decoder output activation, dropout placement, initialization, Adam ancillary parameters, duration/stopping, seed, preprocessing amplitude scaling | Not fully specified by paper; inspect author code and record decisions separately |
| Training duration evidence | Accuracy plots extend to about 500 epochs | Paper Fig.3, p.3; plot is evidence of displayed run, not a complete training/stopping specification |
| Evaluation | Accuracy, class precision/recall/F1, confusion matrix; PCA of 500-d inputs, AE and joint 64-d embeddings, both classifiers' 4-d logits | Paper Figs.1,3–6/Tables 1–2, pp.2–5; add train/dev loss curves and error analysis |

### Loss scaling contract

For a batch of B examples and 500 input values, implement reconstruction as `((x_hat - x) ** 2).sum(dim=1).mean()`: sum over 500 input dimensions, then mean over examples. The paper specifies the per-example squared norm; batch averaging is our explicit aggregation convention. Cross-entropy is likewise averaged over examples. Plain elementwise mean MSE is 500 times smaller and is **not equivalent** at fixed gamma. Log raw reconstruction, raw classification, and weighted total separately. Before training verify a hand-calculated case (all residuals one gives reconstruction 500), batch-size invariance, and correct gradient flow to both branches. Author-code deviations will be recorded rather than copied silently.

### Published results, never reproduced results

| Model | Paper train (6,000) | Paper dev (1,000) | Paper test (1,000) | Our results |
|---|---:|---:|---:|---|
| Two-layer classifier | 52.0% | 38.1% | 36.4% | Not run |
| Deep Softmax Autoencoder | 94.9% | 64.1% | 65.3% | Not run |

Source: paper Table 1, p.3. Table 2, p.4 gives class precision/recall/F1: classical .783/.775/.779; jazz .606/.627/.616; metal .515/.554/.5337; pop .670/.608/.638. These are comparison values, not acceptance targets. The vanilla AE has no genre-classification head; do not invent its classifier accuracy. Compare its reconstruction and embeddings directly.

## Ambiguities, assumptions, and deviations

| ID / status | Exact issue | Options and consequences / recommendation |
|---|---|---|
| A01 — Resolved by user, 2026-10-02 | A second at 22,050 Hz contains 22,050 samples, not 20,000; pool 40 cannot yield 500 from a full second | Harsha explicitly chose the code-backed policy: twenty consecutive 20,000-sample chunks starting at sample 0, average groups of 40. Label each as ~0.907029 seconds at 22,050 Hz, not an exact second. Resampling to 20 kHz or variable-width pooling were alternatives, not chosen. Phase 2 implementation still awaits continuation. |
| A02 — Resolved by user, 2026-10-02 | 400 × 30 seconds could yield roughly 12,000 non-overlapping one-second clips, yet paper reports 8,000 balanced examples | Inspected code uses 20 chunks/recording: 400 × 20 = 8,000; only first 400,000 samples (~18.14 s) contribute, although it first truncates/reshapes 600,000. User approved this source-backed selection; interpretation is source evidence/inference, not stated paper text. |
| A03 — Paper precedence | Poster accuracy table labels dev as 2,000; paper §3 and Table 1 give dev/test 1,000 each | Inspected source uses 6,000/2,000 and no test; its saved labels have 2,000 rows. Retain paper's 6,000/1,000/1,000 for main experiment; actual original partition membership is unrecoverable. |
| A04 — Blocked for Phase 2 | Paper says random example split, without seed or explicit recording-level separation | Record IDs and segment intervals; measure recording-ID intersections and temporal overlap across splits. Random clips from a recording may occur in multiple splits. Do not call reference protocol leakage-free. Recommend paper-oriented clip split plus a separately labeled recording-disjoint additional experiment; grouping can change counts, so define it explicitly. Grouping recordings also cannot guarantee artist/duplicate independence. |
| A05 — Open before models | Activations/dropout placement/initialization/training duration incomplete | Source encoder activations sigmoid/tanh/ReLU/ReLU, decoder sigmoid/sigmoid/ReLU/ReLU, head tanh/tanh/linear, Xavier/zero initialization, dropout after encoder/decoder 192-unit layers. Source trains keep=.8 vs paper .9, 200 classifier/joint epochs and 400 AE vs paper plot near 500. Decoder ReLU cannot reconstruct negative values; choose/document before Phase 4. See detailed audit. |
| A06 — Decision documented | PyTorch instead of reference TensorFlow (paper reference [8], p.6; code to verify) | User permits framework change; preserve architecture/math, report nonidentical defaults and numerical behavior. Python and PyTorch planned; install only inspection dependencies now. |
| A07 — Open faculty clarification | One-page heading versus two-page summary requirement | Ask faculty for binding length; retain content that can fit either. No outgoing message sent. |
| A08 — Open cross-check | Paper Fig.4 caption says held-out test but image title says validation; poster title also says validation | Label our evaluation partitions from manifests, not copied figure titles; preserve this reference uncertainty. |
| A09 — Open dataset audit | Decode failures, short tracks, duplicates or other suspect files may occur | Inventory every discovered recording and retain failures. No dropping, replacing, padding, or synthetic data without an explicit documented decision. Full decode does not certify labels or absence of duplicate performances. |
| A10 — Planned implementation convention | Native audio decoding may avoid reference AU→WAV conversion; no amplitude standardization specified | Direct decoding into float amplitudes is sufficient for inspection; preserve originals. Before preprocessing confirm dtype/scaling and any normalization/resampling choices. Fit any learned transform only on training data. |
| A11 — Paper precedence | Author joint loss is elementwise mean MSE + .1 mean CE, not paper .9 squared-L2 + .1 CE | Preserve paper loss contract above. Source classification/reconstruction relative weighting is 450 times the paper's; any source-loss experiment must be separately labeled. AE source also divides squared-L2 by 500. |
| A12 — Documented deviation from current source | Author AE trains all 8,000 examples, and inspected test scripts load the full dataset | Train all our models only on training partition, fit PCA on training, and reserve test for final evaluation. Prevent evaluation contamination; do not claim our pipeline duplicates current source exactly. |

## Ownership and working rules

Task ownership means accountability, **not evidence that a human performed the work**. Automated work is recorded as Codex-assisted; human implementation/review/contribution claims need actual evidence. Each task has one owner; the other teammate reviews it. Estimates are effort points, not hours or marks. Harsha: **30 points**; Sunil: **30 points**, including substantive model/evaluation work for both.

Use `🔵 Sunil` and `🟣 Harsha` while incomplete, with explicit status. Only after all completion criteria are verified use `[x] 🟢 Owner` and evidence. Review recommendations do not count as completed reviews. Do not mark tasks green just because files exist.

Work one phase at a time. At each checkpoint update this plan, state changes and executed checks, give reproducible commands/expected outputs, identify decisions, name next phase, and pause for the user's continuation. **First-turn exception expressly requested by the user:** complete Phase 0 and minimal Phase 1 together, then stop before preprocessing/models. Use dev for choices, reserve test for final evaluation, and keep raw audio/archives/venvs/caches/large artifacts out of Git.

## Phase 0 — Reference review and planning

- [x] 🟢 Harsha | P0-T01 | Assignment audit, technical specification, balanced roadmap (2 points; reviewer: Sunil)
  - Completion criteria: read all three PDFs, inspect existing workspace/instructions, link requirements to pages, record dates/deliverables/contradictions, and create this plan before model code.
  - Evidence: Codex extracted 2+6+1 pages and visually reviewed all rendered pages; `rg --files --hidden`, ancestor `AGENTS.md` checks, and `git status --short` inspected; this file created before project implementation. No human review claimed.
- [x] 🟢 Sunil | P0-T02 | Author-code evidence and ambiguity audit (2 points; reviewer: Harsha)
  - Depends on: P0-T01.
  - Completion criteria: inspect linked repository if accessible; record commit/file evidence, preprocessing/split/loss conflicts, unresolved decisions, and preservation/attribution policy.
  - Evidence: Codex read pinned author revision `00b1802e24d1ab6fb3135b24a44976a51033ed24`, loader/model/test scripts and 2,000-row saved dev labels; [reference audit](docs/reference_audit.md) records permalinks/conflicts. Harsha's explicit waveform choice recorded under A01/A02; no human code review claimed.

## Phase 1 — Environment and dataset

- [ ] 🟣 Harsha | P1-T01 | Minimal inspection environment and Git hygiene (2 points; reviewer: Sunil)
  - Status: In progress. Depends on: Phase 0 plan created.
  - Completion criteria: isolated environment imports audio reader/array packages, versions pinned; documented repeatable setup command; Git ignores raw/derived bulk, archives, envs/caches; original materials unchanged.
- [ ] 🔵 Sunil | P1-T02 | Acquire original GTZAN audio (1 point; reviewer: Harsha)
  - Status: In progress. Depends on: Phase 0 plan created.
  - Completion criteria: check local/configured locations, obtain reputable audio archive (resumable where practical), safely extract, record source/date/attribution/terms and checksum; verify actual audio exists. If blocked, record exact source/manual steps without claiming success.
- [ ] 🔵 Sunil | P1-T03 | Inventory and full decode verification (2 points; reviewer: Harsha)
  - Status: Not started. Depends on: P1-T01, P1-T02.
  - Completion criteria: reproducible inventory with path, recording ID, genre, sample rate, channels, duration, decode status; actual counts by genre; full decode of files, suspect/missing files listed; no implicit exclusions.
- [ ] 🟣 Harsha | P1-T04 | Initial waveform inspection and dataset checkpoint (1 point; reviewer: Sunil)
  - Status: Not started. Depends on: P1-T03.
  - Completion criteria: inspect real waveform amplitude/finite values and representative plots, report rates/channels/duration ranges and anomalies; record data readiness and next decisions. No segmentation or pooling implementation.

## Phase 2 — Preprocessing and split protocol

- [ ] 🟣 Harsha | P2-T01 | Resolve waveform/count policy and implement segmentation/pooling (4 points; reviewer: Sunil)
  - Status: Not started — A01/A02 approved; awaits Phase 1 checkpoint and user's continuation. Depends on: Phase 1, P0-T02.
  - Completion criteria: decision log approved; deterministic selected-recording list and segment boundaries in original sample/time units; pooled outputs exactly 500; label mapping checked against genre folders; real-data count/boundary/pool checks pass; no unsupported paper-equivalence claim.
- [ ] 🔵 Sunil | P2-T02 | Reproducible splits and leakage audit (4 points; reviewer: Harsha)
  - Status: Blocked — A04 and P2-T01. Depends on: P2-T01.
  - Completion criteria: approved main protocol yields specified train/dev/test sizes and recorded class counts; fixed seed/config and stable segment IDs; assert segment uniqueness and measure shared recording IDs/intervals; report any overlap; define optional recording-disjoint experiment separately, with zero recording intersection verified if run. Test partition locked.

## Phase 3 — Baseline classifier

- [ ] 🟣 Harsha | P3-T01 | Implement and train 500→128 tanh→4 baseline (5 points; reviewer: Sunil)
  - Status: Not started. Depends on: Phase 2, A05 configuration decisions.
  - Completion criteria: tensor/label/logit and CE checks pass; finite real-data training run completes; reproducible config, checkpoint, seed, train/dev curves and metrics saved; checkpoint selection uses dev only.
- [ ] 🔵 Sunil | P3-T02 | Independently validate baseline outputs and evaluation path (2 points; reviewer: Harsha)
  - Status: Not started. Depends on: P3-T01.
  - Completion criteria: rerun checkpoint evaluation on train/dev; verify confusion/metric arithmetic, prediction-ID alignment, four-dimensional pre-softmax export, eval mode/dropout disabled; test remains unopened until Phase 6 final evaluation.

## Phase 4 — Vanilla autoencoder

- [ ] 🔵 Sunil | P4-T01 | Implement/train deeper reconstruction-only autoencoder (5 points; reviewer: Harsha)
  - Status: Not started. Depends on: Phase 3 checkpoint; shares Phase 2 data/config contracts.
  - Completion criteria: exact encoder/decoder widths verified; squared-L2 scaling check passes; finite train/dev reconstruction curves and checkpoint saved; no label loss; actual 64-dimensional embeddings with IDs exported.
- [ ] 🟣 Harsha | P4-T02 | Validate reconstructions and embedding extraction (2 points; reviewer: Sunil)
  - Status: Not started. Depends on: P4-T01.
  - Completion criteria: independent loss recomputation matches; shape/ID/order/finite checks pass; inspect actual reconstruction examples and embedding variance; document findings and limitations without invented classification results.

## Phase 5 — Deep Softmax Autoencoder

- [ ] 🔵 Sunil | P5-T01 | Joint architecture and mathematical objective (5 points; reviewer: Harsha)
  - Status: Not started. Depends on: Phase 4.
  - Completion criteria: shared 64-d bottleneck and 64→32→16→4 head verified; gamma .9; loss reduction and gradient-flow checks pass; raw component and weighted total logging verified on real batches; document activations/dropout/defaults.
- [ ] 🟣 Harsha | P5-T02 | Joint training, dev selection, and exports (5 points; reviewer: Sunil)
  - Status: Not started. Depends on: P5-T01.
  - Completion criteria: reproducible finite full run completed with train/dev reconstruction/CE/total/accuracy curves; checkpoint selected on predefined dev rule; actual 64-d embeddings and 4-d logits exported with IDs; component totals independently reconcile. Test held for Phase 6.

## Phase 6 — Experiments and analysis

- [ ] 🟣 Harsha | P6-T01 | Final quantitative evaluation and comparison (5 points; reviewer: Sunil)
  - Status: Not started. Depends on: Phase 5; freeze configs/checkpoints before test.
  - Completion criteria: one final held-out test evaluation per frozen classifier; accuracy, per-class precision/recall/F1, confusion matrices and counts verified; train/dev/test results separately labeled; compare paper numbers with methodological differences; review representative errors and known dataset limitations.
- [ ] 🔵 Sunil | P6-T02 | PCA, leakage comparison, and qualitative analysis (5 points; reviewer: Harsha)
  - Status: Not started. Depends on: Phase 5 and locked Phase 6 protocol.
  - Completion criteria: PCA plots for inputs, AE/joint 64-d embeddings, baseline/joint logits with labels/sample IDs; PCA fitted on training only and displayed partitions stated; variance/limitations explained; execute approved additional recording-disjoint run if feasible or explicitly document deferral (never replace main result); jointly review quantitative findings.

## Phase 7 — Submission and demonstration

- [ ] 🟣 Harsha | P7-T01 | Reproducible repository, demo and own contribution/Q&A record (4 points; reviewer: Sunil)
  - Status: Not started. Depends on: Phase 6.
  - Completion criteria: README commands run from clean environment; private repo/faculty access verified with authorized user action; code/config/lightweight metadata included and bulk data excluded; live demo rehearsed and timed; Harsha explains baseline, loss, training and limitations; review report/slides and record actual contributions/AI assistance.
- [ ] 🔵 Sunil | P7-T02 | PDF, slides, demo and own contribution/Q&A record (4 points; reviewer: Harsha)
  - Status: Not started. Depends on: Phase 6; faculty length/review-slot clarification.
  - Completion criteria: PDF length clarified, required content covered, rendered pages/slides inspected; results source-labeled; Sunil runs demo and explains AE/joint model/data/splits; review README/evaluation; truthful contributions and both members' rehearsal recorded; final submission confirmed by deadline.

## Dependencies and indicative timing

P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7. Within a phase only independent tasks may run in parallel; use phase checkpoints before continuing. Preliminary reference/data work: October 2; approved preprocessing/baselines: October 3; joint model/evaluation and minimum demo readiness: October 4; review-ready from October 5; refinements/report submission by October 10. These are planning estimates, not promises of training completion or a confirmed review slot. Reassess compute/runtime after small real-data checks, without tuning on test data.

## Teammate progress and automated assistance

| Owner | Planned effort | Verified tasks | In progress / blockers | Human contribution evidence |
|---|---:|---|---|---|
| Harsha | 30 points | P0-T01, Codex-assisted artifact | Phase 1 environment/inspection pending | None recorded yet; ownership is not authorship |
| Sunil | 30 points | None yet | P0-T02 code audit in progress; Phase 1 acquisition/inventory pending | None recorded yet; ownership is not authorship |

Automated assistance on 2026-10-02: Codex inspected the existing repository and all local reference PDFs, drafted this plan and effort allocation, and began external source/code research. Temporary PDF reading/rendering tools were installed outside the project in `/private/tmp/ml-reference-tools`; no reference files were modified. Record further executed commands and actual verification evidence below as Phase 1 progresses.

## Current checkpoint

Phase 0 plan created. Author-code audit and Phase 1 are underway under the first-turn authorization. No preprocessing, model, split, training, or reproduced result has been created. Dataset has not yet been downloaded or verified; expected counts are not observed counts.
