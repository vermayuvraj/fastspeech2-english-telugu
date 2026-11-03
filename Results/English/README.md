# English experiment

Trained on 5.0014 hours and validated on 1.0020 hours. Training ended at 19,000 steps; synthesis uses step 7,000.

I do not know Telugu, so I used automatic transcription, predicted-quality scores and acoustic measurements to assess the Telugu outputs. Assessment of both languages is automatic. No native-speaker listening or human MOS study was performed.

## Audio and plots

| Sample | Audio | Duration | Pitch | Energy | Waveform comparison |
|---|---|---|---|---|---|
| E1 | [WAV](test_generated_audios/E1.wav) | [Plot](variance_adaptor_plots/duration/E1.png) | [Plot](variance_adaptor_plots/pitch/E1.png) | [Plot](variance_adaptor_plots/energy/E1.png) | [Overlay](automatic_assessment_plots/E1.png) |
| E2a | [WAV](test_generated_audios/E2a.wav) | [Plot](variance_adaptor_plots/duration/E2a.png) | [Plot](variance_adaptor_plots/pitch/E2a.png) | [Plot](variance_adaptor_plots/energy/E2a.png) | [Overlay](automatic_assessment_plots/E2a.png) |
| E2b | [WAV](test_generated_audios/E2b.wav) | [Plot](variance_adaptor_plots/duration/E2b.png) | [Plot](variance_adaptor_plots/pitch/E2b.png) | [Plot](variance_adaptor_plots/energy/E2b.png) | [Overlay](automatic_assessment_plots/E2b.png) |
| E3 | [WAV](test_generated_audios/E3.wav) | [Plot](variance_adaptor_plots/duration/E3.png) | [Plot](variance_adaptor_plots/pitch/E3.png) | [Plot](variance_adaptor_plots/energy/E3.png) | [Overlay](automatic_assessment_plots/E3.png) |
| E4 | [WAV](test_generated_audios/E4.wav) | [Plot](variance_adaptor_plots/duration/E4.png) | [Plot](variance_adaptor_plots/pitch/E4.png) | [Plot](variance_adaptor_plots/energy/E4.png) | [Overlay](automatic_assessment_plots/E4.png) |

## Automatic measurements

| Sample | Whisper CER % | MMS CER % | Raw UTMOS prediction |
|---|---:|---:|---:|
| E1 | 36.7 | 57.1 | 0.84 |
| E2a | 0.0 | 35.0 | 0.80 |
| E2b | 39.1 | 46.9 | 0.64 |
| E3 | 35.7 | 47.6 | 0.66 |
| E4 | 46.2 | 53.8 | 0.69 |

CER measures transcript disagreement, not pronunciation accuracy. UTMOS is an unbounded quality prediction, not human MOS; predictions below 1 are not listener ratings. See the same-language controls and full limitations before interpreting the numbers.

- [Per-sentence findings and both ASR transcripts](AUTOMATIC_ASSESSMENT.md)
- [Whisper measurements](automatic_asr.json)
- [Independent MMS measurements](automatic_asr_mms.json)
- [Quality predictions and crop variation](automatic_quality.json)
- [Acoustic measurements](automatic_acoustic_analysis.json)
- [Assessment evidence and control summaries](automatic_review.json)
- [Protocol and reproduction](../../docs/AUTOMATIC_ASSESSMENT.md)

Human listening notes in `listening_review.json` remain unfilled and are not claimed as completed. The earlier forced-alignment diagnostic remains available separately in `objective_audio_evaluation.json`.
