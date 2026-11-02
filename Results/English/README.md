# English experiment

The model was trained on 5.0014 hours and validated on 1.0020 hours of accepted Svarah recordings. Training stopped at 19,000 optimizer steps. Test synthesis uses the checkpoint selected at step 7,000, with validation loss 2.7096.

| Sample | Audio | Duration plot | Pitch plot | Energy plot |
|---|---|---|---|---|
| E1 | [WAV](test_generated_audios/E1.wav) | [Plot](variance_adaptor_plots/duration/E1.png) | [Plot](variance_adaptor_plots/pitch/E1.png) | [Plot](variance_adaptor_plots/energy/E1.png) |
| E2a | [WAV](test_generated_audios/E2a.wav) | [Plot](variance_adaptor_plots/duration/E2a.png) | [Plot](variance_adaptor_plots/pitch/E2a.png) | [Plot](variance_adaptor_plots/energy/E2a.png) |
| E2b | [WAV](test_generated_audios/E2b.wav) | [Plot](variance_adaptor_plots/duration/E2b.png) | [Plot](variance_adaptor_plots/pitch/E2b.png) | [Plot](variance_adaptor_plots/energy/E2b.png) |
| E3 | [WAV](test_generated_audios/E3.wav) | [Plot](variance_adaptor_plots/duration/E3.png) | [Plot](variance_adaptor_plots/pitch/E3.png) | [Plot](variance_adaptor_plots/energy/E3.png) |
| E4 | [WAV](test_generated_audios/E4.wav) | [Plot](variance_adaptor_plots/duration/E4.png) | [Plot](variance_adaptor_plots/pitch/E4.png) | [Plot](variance_adaptor_plots/energy/E4.png) |

Numerical predictions are saved in `prediction_traces/`. Generated samples scored 0.204-0.346 on MMS transcript-alignment confidence, compared with a mean of 0.746 on ten held-out recordings. These scores are uncalibrated diagnostics, not pronunciation accuracy or listener ratings. The English outputs need close listening review.

Listening notes remain pending in `listening_review.json`. No naturalness rating or confirmed pronunciation judgment is claimed.
