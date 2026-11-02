# English and Telugu FastSpeech 2

Independent FastSpeech 2 acoustic models for the IIIT Hyderabad Voice AI assignment. This repository adapts [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) at commit `d4e79eb52e8b01d24703b2dfc0385544092958f3`.

**Both training runs and all ten test syntheses are complete. Listening review and final report approval are pending. The Google Form has not been submitted.** The [submission audit](Report/submission_audit.json) records outstanding checks.

- [Concise report in Word](Report/Technical_Report_Concise.docx)
- [Full technical report in PDF](Report/Technical_Report.pdf)
- [English audio and plots](Results/English/README.md)
- [Telugu audio and plots](Results/Telugu/README.md)
- Trained checkpoints: [English](English/checkpoints/best.pt) and [Telugu](Telugu/checkpoints/best.pt)

| Model | Training hours | Validation hours | Run ended | Selected checkpoint |
|---|---:|---:|---:|---:|
| English | 5.0014 | 1.0020 | 19,000 | 7,000 |
| Telugu | 5.0001 | 1.0009 | 26,000 | 14,000 |

## Setup

Python 3.10 is the tested interpreter. Use a virtual environment. For an NVIDIA GPU:

```bash
python -m venv .venv
# Activate .venv (Windows: .venv\Scripts\Activate.ps1; Linux: source .venv/bin/activate)
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-dev.txt
python scripts/check_environment.py
```

Accept the [Svarah dataset conditions](https://huggingface.co/datasets/ai4bharat/Svarah) using your own account, then run `python scripts/sign_in_huggingface.py` interactively. Tokens belong in the Hugging Face credential store, never in this repository or a command argument. Telugu is downloaded from the specified [SPRINGLab dataset](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu). Dataset revisions are pinned in the language configurations.

Allow roughly 30 GB of free storage for downloads, exported audio, features and resume states. Training time depends on the GPU and accepted utterance lengths; no completion-time guarantee is assumed. Keep the computer awake and powered during long runs.

## Run both experiments

Use an external work directory, outside this repository:

```bash
python scripts/download_aligner.py --work-dir /absolute/path/to/work
python scripts/run_project.py --work-dir /absolute/path/to/work --wait-for-access
```

The runner processes Telugu first, then English, sequentially on one GPU. It exports data, selects disjoint subsets, aligns targets, trains, synthesizes all ten prompts, saves all 30 variance plots and rebuilds the report. It reuses saved features and resumes the optimizer. A lock prevents duplicate runners; remove a stale lock only after checking that its recorded PID is inactive. Failures are recorded in `STATUS.json`.

## Run individual stages

```bash
python English/preprocess.py --work-dir /absolute/path/to/work
python English/train.py --work-dir /absolute/path/to/work --resume
python English/inference.py --work-dir /absolute/path/to/work
python Telugu/preprocess.py --work-dir /absolute/path/to/work
python Telugu/train.py --work-dir /absolute/path/to/work --resume
python Telugu/inference.py --work-dir /absolute/path/to/work
```

Preprocessing supports `--stage download|export|prepare|features|all` and `--local-parquet-dir PATH` for an authorized local copy of the specified dataset. Telugu explicitly uses the corpus's two voice labels. Svarah's HF release omits speaker IDs, so English uses hashed demographic-profile groups as conditioning proxies; these are not verified speaker identities. See [Svarah adaptation](docs/SVARAH_ADAPTATION.md). Inference defaults to the conditioning group with the most accepted training audio; `--speaker ID` selects another trained group.

For isolated diagnostics, use `--smoke` with a work directory containing `smoke`. Train diagnostics with `--smoke --steps 200`; pass their checkpoint explicitly to inference and set `--output` to a path containing `smoke`. Such artifacts stay outside assignment Results. Normal training refuses data that fails the five-hour/one-hour requirement.

## Method

The reference encoder, decoder, variance predictors, length regulator and postnet are retained. `docs/upstream.patch` shows modifications, and `docs/upstream_manifest.json` records original and adapted file hashes. The compact configuration has four encoder and four decoder blocks, hidden width 192, two heads, and a 256-channel postnet. Each language has its own initialization, speaker embeddings, normalization, optimizer and checkpoint.

Text is normalized consistently for training, alignment and synthesis. Telugu uses NFC, explicit number/ordinal/clock rules and uroman. Linguistic units are **romanized graphemes**, although the upstream API calls their feature level `phoneme_level`. This is a documented alternative to a phoneme frontend, with possible pronunciation ambiguity. The original prompt text remains unchanged in the input manifests and output metadata; the malformed spacing in E3 and T3 is retained.

Audio is resampled to mono 22.05 kHz. HiFi-GAN-compatible log mels use FFT/window 1024, hop 256, 80 mel channels and a 0–8000 Hz band. The official multilingual MMS forced aligner supplies character timing; silence/word-gap tokens represent observed CTC gaps. Durations sum exactly to the mel-frame count. WORLD DIO/StoneMask supplies F0; unvoiced gaps are interpolated in log F0. Token energy is mean log STFT magnitude L2. All statistics are fitted on training data only.

Selection removes duplicate decoded audio, groups identical normalized transcripts, reserves voice coverage for training, and uses a seeded group ordering to select approximately five training hours and one validation hour. Speaker overlap is intentional; utterances, audio hashes and normalized prompts do not overlap. Actual accepted durations, exclusions, hashes and source rows are exported as manifests.

The model predicts duration, pitch and energy independently of external TTS weights. The frozen [SpeechBrain multispeaker HiFi-GAN](https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz) is a separate published vocoder, not a model trained for this assignment. Its weights are pinned and strictly matched after translating convolution-wrapper names.

## Results and interpretation

The local live monitor shows the current stage, loss curves, step count, measured-speed ETA and generated artifacts. Start it with `python monitor/server.py --work-dir /absolute/path/to/work`, then open [the progress page](http://127.0.0.1:8765). It binds only to this computer and refreshes every two seconds. Closing the page does not stop training. Its time estimate covers training to the configured maximum; it excludes listening review, publication and submission.

Expected folders match the assignment: `English/`, `Telugu/`, `Results/{English,Telugu}/`, and `Report/Technical_Report.pdf`. Training saves `checkpoints/best.pt` for evaluation and keeps optimizer/RNG state in the external work directory. Model files embed their configuration, symbols, speaker mapping and statistics.

Synthesis writes five WAV files per language (E2 and T2 have separate a/b outputs), duration/pitch/energy plots for every sentence, and CSV/NPZ/JSON prediction traces. Duration plots use milliseconds, continuous F0 uses Hz, and energy uses relative STFT units. Continuous pitch is not measured waveform voicing. Normalized-space pitch/energy control factors are not physical Hz/amplitude multipliers.

Training loss uses minibatch windows. Validation covers the entire held-out split, weights each loss by its valid element count and uses ground-truth variance inputs. Free-running test synthesis uses predictions. Neither lower reconstruction loss nor a regular-looking plot establishes intelligibility.

Listen to each generated sample, then enter `pronunciation`, `overall_quality` and `specific_observations` in `Results/LANGUAGE/listening_review.json`. The report includes only recorded observations. Telugu T2 changes the meaning of “pattu”; this does not inherently require a different segmental pronunciation. English E2 requires a contextual stress check, which average pitch alone cannot establish.

```bash
python scripts/run_checks.py
python scripts/build_report.py
python scripts/audit_submission.py --strict
```

The audit fails on missing real checkpoints, insufficient hours, absent plots/audio, changed prompts, mismatched hashes, diagnostic results or missing listening observations. Review the rendered PDF after rebuilding it.

## Publication

This repository contains the experiment files for review. Complete the listening notes and obtain approval of the final report before submitting the Google Form. The strict audit remains incomplete while those checks are pending. Both compact inference checkpoints fit GitHub's per-file limit. Raw recordings, caches, tokens and optimizer states remain outside Git.

See `docs/THIRD_PARTY.md` for sources and licensing, and `docs/EXPLAIN_THE_PROJECT.md` for an architecture walkthrough.
