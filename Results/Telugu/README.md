# Telugu experiment

The model was trained on 5.0001 hours and validated on 1.0009 hours of accepted IndicTTS Telugu recordings. Training stopped at 26,000 optimizer steps. The inference checkpoint was selected at step 14,000 using the configured validation-improvement threshold.

| Sample | Audio | Duration plot | Pitch plot | Energy plot |
|---|---|---|---|---|
| T1 | [WAV](test_generated_audios/T1.wav) | [Plot](variance_adaptor_plots/duration/T1.png) | [Plot](variance_adaptor_plots/pitch/T1.png) | [Plot](variance_adaptor_plots/energy/T1.png) |
| T2a | [WAV](test_generated_audios/T2a.wav) | [Plot](variance_adaptor_plots/duration/T2a.png) | [Plot](variance_adaptor_plots/pitch/T2a.png) | [Plot](variance_adaptor_plots/energy/T2a.png) |
| T2b | [WAV](test_generated_audios/T2b.wav) | [Plot](variance_adaptor_plots/duration/T2b.png) | [Plot](variance_adaptor_plots/pitch/T2b.png) | [Plot](variance_adaptor_plots/energy/T2b.png) |
| T3 | [WAV](test_generated_audios/T3.wav) | [Plot](variance_adaptor_plots/duration/T3.png) | [Plot](variance_adaptor_plots/pitch/T3.png) | [Plot](variance_adaptor_plots/energy/T3.png) |
| T4 | [WAV](test_generated_audios/T4.wav) | [Plot](variance_adaptor_plots/duration/T4.png) | [Plot](variance_adaptor_plots/pitch/T4.png) | [Plot](variance_adaptor_plots/energy/T4.png) |

Numerical predictions are saved in `prediction_traces/`. `objective_audio_evaluation.json` compares generated-waveform diagnostics with ten held-out recordings. CTC alignment confidence is an uncalibrated transcript-compatibility measure; it is not a human pronunciation score.

Listening observations remain unfilled in `listening_review.json`. No naturalness rating or native-speaker pronunciation judgment is claimed.
