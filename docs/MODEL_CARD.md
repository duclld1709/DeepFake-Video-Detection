# Model card: `80_frame` (ResNeXt-50 + LSTM)

## Overview

| | |
|---|---|
| Task | Binary classification of a face video: manipulated (deepfake) vs. authentic |
| Architecture | ImageNet-pretrained ResNeXt-50 (32x4d) frame encoder, 2048-d single-layer LSTM (no bias), linear head |
| Input | 80 consecutive face crops, 112x112, **BGR** order, ImageNet mean/std |
| Output | Softmax over `[FAKE, REAL]` per clip. In `legacy` mode, also per frame (exact decomposition) |
| Temporal mode | `legacy`: each frame passes through one LSTM step from a zero state, and the outputs are averaged. This is how the checkpoint was trained (see README, audit item 1) |
| Face detector at training time | `face_recognition` (dlib HOG), tight box without margin |
| Weights | Google Drive (`scripts/download_weights.py`) |

## Intended use

- Give a **second opinion** on whether a video showing one person's face has been manipulated.
- Support triage and education about deepfakes, as one signal among several (source checks,
  calling the person back, metadata).

## Out of scope

- Legal, forensic or journalistic proof of authenticity or manipulation.
- Videos without a clearly visible frontal face, with several people talking, or with heavy
  occlusion, and non-face manipulations (audio-only, lip-sync-only, background edits).
- Automated moderation decisions without human review.

## Training data and evaluation

- Trained on a curated, class-balanced set of real and manipulated videos (course project; see the
  v1 report in `docs/report/`).
- The reported v1 metric is **90% validation accuracy on a random video-level split**. That split does
  not separate fakes from their source videos or identities, so the number is likely optimistic, and it
  says nothing about other datasets or compression levels.
- `training/evaluate.py` implements the recommended protocol: a cross-dataset test with H.264 CRF
  23/28/35, 0.5x downscale and 15 FPS perturbations, reporting AUC, EER, F1, ECE and coverage.

## Known limitations

- **Generalisation.** Detectors trained on one generation method or dataset often degrade sharply on
  unseen ones.
- **Compression.** Messaging and social-media re-encoding removes artefacts the model may rely on.
- **Calibration.** The probabilities are not calibrated. The `inconclusive` band (default ±0.1
  around 0.5) is a heuristic until temperature scaling is fitted on held-out data.
- **Detector mismatch.** With the OpenCV fallback detector, crops differ from training and the
  service adds a warning to the result.
- **Fairness.** Performance across skin tones, ages, genders and lighting has not been measured.
  Do not assume it is uniform.

## Ethical considerations

A false "real" verdict can give a scam unwarranted credibility. A false "fake" verdict can discredit
genuine footage. The UI and API therefore show uncertainty, face coverage and warnings, and the
documentation says that the output is not proof.
