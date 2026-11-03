# Automatic assessment protocol

The author does not know Telugu. This revision therefore uses automatic speech recognition, predicted quality and acoustic measurements for the Telugu assessment. Both languages are assessed automatically. No native-speaker listening, human MOS study or certified phoneme-level pronunciation rating is claimed.

## Measurements

- **Whisper large-v3:** faster-whisper, int8/float16 on CUDA, beam size 5, temperature 0, explicit English/Telugu language, transcription task, no VAD and no reference text supplied as a prompt. Word timestamps are recognizer estimates.
- **MMS-1b-all:** independent greedy CTC decoding with English and Telugu adapters (`eng`, `tel`), no external language model and no reference hint. The second recognizer was added because Whisper made substantial errors on the real Telugu controls. Both recognizers' results are retained; a favorable recognizer is not selected separately for each synthesized sample.
- **UTMOSv2:** official `fusion_stage3`, fold 0, seed 42, default `sarulab` dataset conditioning. Three crop predictions per recording are averaged. Silence removal is disabled. These are raw, unbounded regression predictions. Values outside the nominal 1-5 listener scale are retained and flagged as extrapolations, not converted into human ratings. Crop variation is not a listener confidence interval. Calibration on these Telugu voices is unverified.
- **Acoustics:** original token-duration, continuous-F0 and energy predictions are retained. WORLD estimates waveform F0; frame-level energy uses the same STFT convention as training. Low-energy intervals use 20 ms RMS windows, 10 ms hop, a -30 dB relative threshold and minimum duration of 80 ms. A -40 dB sensitivity check is also retained. These thresholds do not establish linguistic silence or perceptual quality.

## Controls and scoring

Each model processes all five test utterances in each language, plus the first ten validation utterances ordered by ID in each language. This is the same fixed control selection as the earlier MMS alignment diagnostic. The controls were not used for TTS training or checkpoint selection; they are not matched to test sentences by duration, speaker or recording condition. They provide descriptive checks of evaluation-tool behavior, not a matched listening study.

Reference and hypothesis receive the same deterministic number/time expansion. Clock notation such as `10.45 a.m.` is interpreted as `10:45 AM` only when there is an explicit clock marker; ordinary decimal numbers remain decimals. Case and punctuation are normalized with Unicode NFC. The original prompts and raw hypotheses are retained unchanged.

Word error rate retains word boundaries. Character error rate counts Unicode codepoints after whitespace removal, allowing the supplied internal spacing in E3/T3 to be examined separately from character content. Telugu spelling and segmentation differences still affect these scores. Corpus rates sum edit counts and divide by total reference length, rather than averaging sentence percentages. Error rates can exceed 100% when there are insertions.

The `con`/`duct` regions and other word spans follow the TTS model's grapheme durations. They are not independent phonetic or syllable alignments. Higher pitch, duration or energy in a region alone cannot certify stress placement. The two uses of Telugu `pattu` need not have different segmental pronunciations.

## Evidence and limits

Each result records the WAV SHA256, model revision, decoding settings and available checkpoint hashes. `automatic_review.json` ties the observations to hashes of all four measurement files. `AUTOMATIC_ASSESSMENT.md` in each language's results folder provides the per-sentence observations, both native-script hypotheses and acoustic-overlay links. The original `listening_review.json` files remain unfilled: their contents are not presented as completed human listening.

ASR disagreements may arise from the recognizer, accent, orthography, segmentation or synthesis. UTMOS predictions are not human judgments. Neither tool certifies native Telugu pronunciation or natural prosody. No paired reference recordings exist for the supplied test prompts, so reference-based waveform quality metrics are not reported.

## Reproduction

Install `requirements-evaluation.txt` after the documented PyTorch setup. Use an external work directory and keep model caches outside Git:

The complete evaluation is `python scripts/run_assessment.py --work-dir PATH`. Its individual stages are:

```bash
python scripts/prepare_evaluation_models.py --work-dir PATH
python scripts/automatic_assessment.py asr --work-dir PATH
python scripts/automatic_assessment.py quality --work-dir PATH
python scripts/automatic_assessment.py acoustics --work-dir PATH
python scripts/mms_crosscheck.py --work-dir PATH
python scripts/plot_automatic_assessment.py
python scripts/build_automatic_review.py
python scripts/audit_submission.py
```

Pinned model revisions are in `docs/automatic_evaluation_models.json`. The quality backbones are prepared in advance and loaded offline, which also avoids a background Transformers model-conversion job. Fixed seeds aid repeatability; exact floating-point predictions may vary across hardware/software versions.

The report builder `scripts/build_assessed_report.py --output PATH.docx` creates a staged editable report from completed evidence. Render and inspect that document before replacing the selected PDF. The older report generators preserve the selected report's hash rather than overwriting it.
