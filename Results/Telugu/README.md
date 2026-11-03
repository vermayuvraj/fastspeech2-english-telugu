# Telugu experiment

Trained on 5.0001 hours and validated on 1.0009 hours. Training ended at 26,000 steps; synthesis uses step 14,000.

I do not know Telugu, so I used automatic transcription, predicted-quality scores and acoustic measurements to assess the Telugu outputs. Assessment of both languages is automatic. No native-speaker listening or human MOS study was performed.

## Audio and plots

| Sample | Audio | Duration | Pitch | Energy | Waveform comparison |
|---|---|---|---|---|---|
| T1 | [WAV](test_generated_audios/T1.wav) | [Plot](variance_adaptor_plots/duration/T1.png) | [Plot](variance_adaptor_plots/pitch/T1.png) | [Plot](variance_adaptor_plots/energy/T1.png) | [Overlay](automatic_assessment_plots/T1.png) |
| T2a | [WAV](test_generated_audios/T2a.wav) | [Plot](variance_adaptor_plots/duration/T2a.png) | [Plot](variance_adaptor_plots/pitch/T2a.png) | [Plot](variance_adaptor_plots/energy/T2a.png) | [Overlay](automatic_assessment_plots/T2a.png) |
| T2b | [WAV](test_generated_audios/T2b.wav) | [Plot](variance_adaptor_plots/duration/T2b.png) | [Plot](variance_adaptor_plots/pitch/T2b.png) | [Plot](variance_adaptor_plots/energy/T2b.png) | [Overlay](automatic_assessment_plots/T2b.png) |
| T3 | [WAV](test_generated_audios/T3.wav) | [Plot](variance_adaptor_plots/duration/T3.png) | [Plot](variance_adaptor_plots/pitch/T3.png) | [Plot](variance_adaptor_plots/energy/T3.png) | [Overlay](automatic_assessment_plots/T3.png) |
| T4 | [WAV](test_generated_audios/T4.wav) | [Plot](variance_adaptor_plots/duration/T4.png) | [Plot](variance_adaptor_plots/pitch/T4.png) | [Plot](variance_adaptor_plots/energy/T4.png) | [Overlay](automatic_assessment_plots/T4.png) |

## Automatic measurements

| Sample | Whisper CER % | MMS CER % | Raw UTMOS prediction |
|---|---:|---:|---:|
| T1 | 24.2 | 14.5 | 1.72 |
| T2a | 3.6 | 0.0 | 1.99 |
| T2b | 20.6 | 11.8 | 1.57 |
| T3 | 6.9 | 0.0 | 2.05 |
| T4 | 15.2 | 9.5 | 1.74 |

CER measures transcript disagreement, not pronunciation accuracy. UTMOS is an unbounded quality prediction, not human MOS; predictions below 1 are not listener ratings. See the same-language controls and full limitations before interpreting the numbers.

- [Per-sentence findings and both ASR transcripts](AUTOMATIC_ASSESSMENT.md)
- [Whisper measurements](automatic_asr.json)
- [Independent MMS measurements](automatic_asr_mms.json)
- [Quality predictions and crop variation](automatic_quality.json)
- [Acoustic measurements](automatic_acoustic_analysis.json)
- [Assessment evidence and control summaries](automatic_review.json)
- [Protocol and reproduction](../../docs/AUTOMATIC_ASSESSMENT.md)

Human listening notes in `listening_review.json` remain unfilled and are not claimed as completed. The earlier forced-alignment diagnostic remains available separately in `objective_audio_evaluation.json`.
