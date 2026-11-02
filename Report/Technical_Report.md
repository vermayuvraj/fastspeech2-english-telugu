# Implementation of FastSpeech 2 for English and Telugu Speech Synthesis

Yuvraj Verma

## Abstract

Two FastSpeech 2 models were trained separately: one on Svarah English and one on IndicTTS Telugu. Each used about five hours of training audio and one hour of validation audio. The work adapts the reference implementation to these datasets, prepares duration, pitch and energy targets, and generates speech with a pretrained HiFi-GAN vocoder. Both training runs finished, and the models generated all ten test utterances. The English outputs scored below the held-out baseline on the transcript-alignment check; Telugu scores were closer to their baseline. These measurements do not replace listening, which is still needed to assess pronunciation and prosody.

## 1. Introduction

The assignment is to train English and Telugu speech-synthesis models and examine how they handle the supplied test sentences. FastSpeech 2 [1] is useful for this task because it predicts duration, pitch and energy separately. Those predictions can be plotted alongside the generated audio to examine timing and prosody.

The starting code is ming024/FastSpeech2 [2]. The two models use the same code but are initialized and trained independently. Each has its own speaker embeddings, feature statistics and optimizer state. The pretrained vocoder is shared and kept fixed.

## 2. Model implementation

The text encoder maps input tokens to hidden features. A speaker embedding is added to these features before they enter the variance adaptor. The main settings are listed in Table 1.

Table 1. Acoustic-model configuration.

| Component | Setting |
| Encoder and decoder | Four feed-forward Transformer blocks each; hidden size 192; two attention heads |
| Feed-forward convolution | 768 channels; kernel sizes 9 and 1 |
| Variance predictors | 192 channels; kernel size 3; dropout 0.3 |
| Pitch and energy embeddings | 256 bins for each feature |
| Mel projection and postnet | 80 mel channels; five convolutional layers; 256 hidden channels |

The variance adaptor has three predictors: duration, pitch and energy. Training uses targets extracted from the recordings; synthesis uses the model's predictions. The length regulator repeats each token's hidden features for the predicted number of frames. The decoder then produces a mel spectrogram, which the postnet refines.

This implementation uses the reference code's token-level pitch option: it predicts mean log F0 for each token rather than the continuous wavelet representation in the original paper. The tokens are romanized characters, not phonemes. The code still calls this feature level phoneme_level, so that name should not be read as a phoneme-based frontend. Romanization lets the model and aligner use the same text, but loses some pronunciation distinctions.

The code changes also fix the energy-control path, keep length-regulator tensors on the correct device and prevent long decoder outputs from being truncated. Attention uses PyTorch scaled-dot-product attention. The patch and source-file hashes are saved in docs/upstream.patch and docs/upstream_manifest.json.

SpeechBrain's pretrained multispeaker HiFi-GAN [7] converts the mel spectrograms to 22.05 kHz audio. Its weights are not updated. The loader translates convolution-wrapper parameter names and then checks that every weight matches.

## 3. Dataset preparation

The English recordings come from Svarah [3], which contains Indian-accented English speech. The Telugu recordings come from SPRINGLab's IndicTTS Telugu release [4]. Dataset revisions are recorded in the configuration files. The Telugu release contains two voice labels, supplied through its gender field.

Svarah does not provide speaker IDs in this release. For English speaker conditioning, a hash is made from eight profile fields: gender, age group, primary language, native state, native district, qualification, job category and occupation domain. This gives 115 groups in the accepted data. A group is not a verified speaker: people with the same profile share an embedding. Telugu uses the two supplied voice labels.

Preprocessing removes silent or invalid recordings, clips shorter than 0.7 seconds or longer than 20 seconds, and transcripts longer than 450 tokens. Audio hashes identify duplicate recordings. Normalized transcripts are grouped before splitting so the same prompt cannot appear in both training and validation. The split also reserves training examples for each conditioning group. Voices or profile groups may appear in both splits; recordings and normalized prompts do not overlap.

The candidate pool includes 15% extra audio to allow for failed alignments. The five-hour training and one-hour validation targets are checked again using the accepted recordings. Split seeds are 20261101 for English and 20261102 for Telugu.

Text is normalized to Unicode NFC, and rules expand numbers, dates, ordinals and times. Telugu is romanized with uroman [6]. Alignment, training and inference use this same frontend. Original text is saved alongside the normalized form. The unusual internal spacing in E3 and T3 is retained because it is part of the test.

Audio is converted to mono at 22,050 Hz. MMS forced alignment [5] estimates character boundaries. Gaps between words and at the start or end of a recording are assigned to boundary tokens. These timings become integer frame counts whose sum must equal the mel-spectrogram length.

## 4. Acoustic features and training setup

Mel spectrograms use a 1,024-sample FFT and window, a 256-sample hop and 80 Slaney mel filters from 0 to 8,000 Hz. Reflect padding and uncentered STFT frames match the vocoder's feature settings. WORLD DIO and StoneMask estimate F0. Unvoiced intervals are filled by interpolation in log F0, then averaged over each token. Energy is calculated from the L2 norm of the STFT magnitude and averaged in log space over the same intervals.

Only training tokens are used to calculate pitch and energy normalization statistics and quantization bounds. Validation and synthesis reuse these values. The total loss adds mel and postnet-mel L1 losses to MSE losses for log(1 + duration), normalized log F0 and normalized log energy. Padding is masked throughout; tokens with zero duration are also excluded from pitch and energy loss.

Table 2. Optimization settings for each language.

| Parameter | Value |
| Optimizer | AdamW; betas (0.9, 0.98); epsilon 1e-9 |
| Peak learning rate and schedule | 0.0005; 1,000-step warmup, then inverse-square-root decay |
| Batch size | 6 training; 4 validation |
| Weight decay and gradient clipping | 1e-6 and 1.0, respectively |
| Validation interval and maximum steps | 1,000 and 40,000, respectively |
| Early stopping | After 15,000 steps; 12 checks without a 0.002 improvement |

At each validation check, the model processes the full validation split using ground-truth duration, pitch and energy. Losses are averaged over valid elements, and their sum is used to select the checkpoint. Test synthesis uses predicted duration, pitch and energy, so validation loss alone does not measure final speech quality. Training uses CUDA mixed precision, with BF16 where supported and FP16 otherwise.

## 5. Verification and experimental results

The saved verification run passed all 12 tests, static checks and Python compilation. Tests cover text processing, data splits, duplicate removal, duration targets and model operations. A full forward and backward pass also completed on the local RTX 4050 GPU. These checks verify execution, not speech quality.

English: validation loss fell from 3.5498 at the first check to 2.7359 at the final check. Its lowest value was 2.7096 at step 7,000.

The accepted English subset contains 5.0014 training hours and 1.0020 validation hours. Training ended at step 19,000 after 12 validation checks without enough improvement. Test audio uses the checkpoint from step 7,000. A saved checkpoint is replaced only when validation loss improves by at least 0.002.

{{CURVES:English}}

E1: generated duration 6.34 s; predicted continuous F0 range 114-173 Hz; 0 zero-duration tokens.

E2a: generated duration 5.19 s; predicted continuous F0 range 97-182 Hz; 0 zero-duration tokens.

E2b: generated duration 4.27 s; predicted continuous F0 range 84-165 Hz; 0 zero-duration tokens.

E3: generated duration 2.96 s; predicted continuous F0 range 117-173 Hz; 0 zero-duration tokens.

E4: generated duration 6.78 s; predicted continuous F0 range 105-171 Hz; 2 zero-duration tokens.

The mean MMS alignment confidence for ten held-out English recordings was 0.746. The generated samples were checked with the same aligner. This uncalibrated score measures transcript alignment; it is not pronunciation accuracy or a listener rating.

| Sample | Waveform measurements |
| E1 | Duration 6.34 s; voiced F0 median 120.9 Hz; alignment confidence 0.346; clipped samples 0.0% |
| E2a | Duration 5.19 s; voiced F0 median 109.2 Hz; alignment confidence 0.329; clipped samples 0.0% |
| E2b | Duration 4.27 s; voiced F0 median 119.0 Hz; alignment confidence 0.313; clipped samples 0.0% |
| E3 | Duration 2.96 s; voiced F0 median 120.8 Hz; alignment confidence 0.204; clipped samples 0.0% |
| E4 | Duration 6.78 s; voiced F0 median 117.5 Hz; alignment confidence 0.326; clipped samples 0.0% |

E3 had the lowest alignment confidence among the generated English samples. It needs a closer listening check; the score alone does not identify a pronunciation error.

Telugu: validation loss fell from 2.5752 at the first check to 2.0433 at the final check. Its lowest value was 2.0231 at step 23,000.

The accepted Telugu subset contains 5.0001 training hours and 1.0009 validation hours. Training ended at step 26,000 after 12 validation checks without enough improvement. Test audio uses the checkpoint from step 14,000. A saved checkpoint is replaced only when validation loss improves by at least 0.002.

{{CURVES:Telugu}}

T1: generated duration 6.05 s; predicted continuous F0 range 100-294 Hz; 0 zero-duration tokens.

T2a: generated duration 3.33 s; predicted continuous F0 range 113-247 Hz; 0 zero-duration tokens.

T2b: generated duration 3.54 s; predicted continuous F0 range 89-252 Hz; 0 zero-duration tokens.

T3: generated duration 3.23 s; predicted continuous F0 range 106-270 Hz; 0 zero-duration tokens.

T4: generated duration 9.83 s; predicted continuous F0 range 100-258 Hz; 1 zero-duration tokens.

The mean MMS alignment confidence for ten held-out Telugu recordings was 0.946. The generated samples were checked with the same aligner. This uncalibrated score measures transcript alignment; it is not pronunciation accuracy or a listener rating.

| Sample | Waveform measurements |
| T1 | Duration 6.05 s; voiced F0 median 199.7 Hz; alignment confidence 0.876; clipped samples 0.0% |
| T2a | Duration 3.33 s; voiced F0 median 185.1 Hz; alignment confidence 0.980; clipped samples 0.0% |
| T2b | Duration 3.54 s; voiced F0 median 178.8 Hz; alignment confidence 0.918; clipped samples 0.0% |
| T3 | Duration 3.23 s; voiced F0 median 173.2 Hz; alignment confidence 0.863; clipped samples 0.0% |
| T4 | Duration 9.83 s; voiced F0 median 185.2 Hz; alignment confidence 0.812; clipped samples 0.0% |

T4 had the lowest alignment confidence among the generated Telugu samples. It needs a closer listening check; the score alone does not identify a pronunciation error.

## 6. Evaluation of the test sentences

Each model generated five utterances, counting the alternatives in E2 and T2 separately. Table 3 lists what needs to be checked in each pair. Listening notes have not yet been completed, so these are evaluation criteria rather than claims that the models passed the tests.

Table 3. Sentence-level evaluation criteria.

| Samples | Aspect examined |
| E1 and T1 | Pronunciation and completeness of the rare English word and Telugu technical loanword |
| E2a and E2b | Context-dependent stress in the noun and verb uses of conduct |
| T2a and T2b | Phrase-level prosody for the two uses of pattu; the meaning difference need not alter its segmental pronunciation |
| E3 and T3 | Effect of the supplied internal word spacing on segmentation and timing |
| E4 and T4 | Date and time verbalization; E4 also contains the romanized Telugu word andaroo |

Each utterance has three plots: token duration in milliseconds, continuous F0 in Hz and energy in relative STFT-magnitude units. The numerical traces are saved as CSV and NPZ files. Since F0 is interpolated through unvoiced regions, a pitch value over a consonant or silence does not mean that region is voiced. The plots show the predicted contours; listening is needed to judge stress and pronunciation.

## 7. Limitations

The frontend uses romanized characters rather than language-specific phonemes. It can therefore merge sounds that should remain distinct. Errors in forced alignment can also give the duration predictor incorrect targets. Continuous F0 does not explicitly model voiced and unvoiced regions, and the fixed vocoder may not suit every voice or recording condition.

The English and Telugu losses should not be used to rank the two models: the datasets, voices and normalization statistics differ. There are no reference recordings of the test sentences for a direct audio comparison. A useful next check is to pass held-out ground-truth mel spectrograms through the vocoder. Comparing those outputs with model-generated speech would help identify which errors come from the acoustic model and which come from the vocoder.

## 8. Reproducibility

The language configuration files record dataset revisions, feature settings, model dimensions, seeds and optimizer settings. Each checkpoint includes the configuration, token inventory, speaker mapping and training statistics. Separate files save optimizer and random-number states for resuming training. Dependencies are listed in requirements.txt, and the README gives the setup and commands for each stage. The full pipeline can be run with:

`python scripts/run_project.py --work-dir PATH --wait-for-access`

Fixed seeds make the split repeatable, though training results can still vary across CUDA hardware and software versions. Raw recordings and access credentials are kept outside the repository.

## References

[1] Y. Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech," ICLR, 2021. https://arxiv.org/abs/2006.04558

[2] ming024, FastSpeech2 reference implementation, commit d4e79eb52e8b01d24703b2dfc0385544092958f3. https://github.com/ming024/FastSpeech2

[3] T. Javed et al., "Svarah: Evaluating English ASR Systems on Indian Accents," Interspeech, 2023, pp. 5087-5091. Dataset: https://huggingface.co/datasets/ai4bharat/Svarah

[4] SPRINGLab, IndicTTS Telugu dataset release, derived from the IIT Madras Indic TTS database. https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu

[5] TorchAudio 2.6, "Forced alignment for multilingual data," MMS_FA documentation. https://docs.pytorch.org/audio/2.6.0/tutorials/forced_alignment_for_multilingual_data_tutorial.html

[6] U. Hermjakob, J. May and K. Knight, "Out-of-the-box Universal Romanization Tool uroman," ACL Demonstrations, 2018. https://github.com/isi-nlp/uroman

[7] SpeechBrain, HiFi-GAN trained on LibriTTS at 22,050 Hz, model revision 4188503131602dc234f48d7f22eebea93d788736. https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz
