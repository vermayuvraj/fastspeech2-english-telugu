# English automatic assessment

I do not know Telugu, so I used automatic transcription, predicted-quality scores and acoustic measurements to assess the Telugu outputs. Assessment of both languages is automatic. No native-speaker listening or human MOS study was performed.

These are automatic observations. Human listening fields remain unfilled and are not represented as completed.

## E1

- Whisper WER is 110.0%; whitespace-free CER is 36.7%. Independent MMS gives WER 120.0%, CER 57.1%. Whisper has 6 word substitutions, 0 deletions and 5 insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error.
- UTMOSv2 predicts 0.84, compared with a mean of 2.77 for ten same-language held-out recordings. No waveform clipping was found.
- Utterance duration is 6.34 s. The target word occupies 1393 ms (22% of the utterance).
- Predicted continuous F0 spans 114-173 Hz. Its duration-weighted F0 is 144.9 Hz.
- Predicted relative STFT energy spans 1.7-96.1. Its duration-weighted mean is 48.7.

Whisper transcript: "then there is established materialism of the considerable momentum leading up to the notary proceedings."

MMS transcript: "balande yesus tablis mataalis anbablanggi dainik parsidelaban warandan liwinla parlinan ta posadins"

[Waveform and acoustic overlay](automatic_assessment_plots/E1.png)

## E2a

- Whisper WER is 0.0%; whitespace-free CER is 0.0%. Independent MMS gives WER 72.7%, CER 35.0%. Whisper has 0 word substitutions, 0 deletions and 0 insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error.
- UTMOSv2 predicts 0.80, compared with a mean of 2.77 for ten same-language held-out recordings. No waveform clipping was found.
- Utterance duration is 5.19 s. Conduct takes 360 ms; con/duct grapheme regions receive 139/221 ms.
- Predicted continuous F0 spans 97-182 Hz. Con/duct means are 135.7/125.2 Hz.
- Predicted relative STFT energy spans 2.1-109.7. Con/duct means are 49.1/29.4. Higher first-region energy occurs in both contexts; this cue does not demonstrate the requested stress reversal.

Whisper transcript: "The orchestra's conduct during the ceremony was praised by the audience."

MMS transcript: "the orcestra standep during the seremony wast pras ni e argans"

[Waveform and acoustic overlay](automatic_assessment_plots/E2a.png)

## E2b

- Whisper WER is 54.5%; whitespace-free CER is 39.1%. Independent MMS gives WER 81.8%, CER 46.9%. Whisper has 6 word substitutions, 0 deletions and 0 insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error.
- UTMOSv2 predicts 0.64, compared with a mean of 2.77 for ten same-language held-out recordings. No waveform clipping was found.
- Utterance duration is 4.27 s. Conduct takes 372 ms; con/duct grapheme regions receive 163/209 ms.
- Predicted continuous F0 spans 84-165 Hz. Con/duct means are 122.4/119.2 Hz.
- Predicted relative STFT energy spans 1.0-88.0. Con/duct means are 51.7/28.3. Higher first-region energy occurs in both contexts; this cue does not demonstrate the requested stress reversal.

Whisper transcript: "researchers need to conduct an experiment to manipulate the progress report."

MMS transcript: "thi serches wi pandetan expamen to winini the propos report"

[Waveform and acoustic overlay](automatic_assessment_plots/E2b.png)

## E3

- Whisper WER is 100.0%; whitespace-free CER is 35.7%. Independent MMS gives WER 112.5%, CER 47.6%. Whisper has 5 word substitutions, 0 deletions and 3 insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error.
- UTMOSv2 predicts 0.66, compared with a mean of 2.77 for ten same-language held-out recordings. No waveform clipping was found.
- Utterance duration is 2.96 s. The retained rhyth/m split receives a 35 ms boundary; the sentence's median interword boundary is 58 ms.
- Predicted continuous F0 spans 117-173 Hz. The two fragments' mean F0 values are 132.8/146.2 Hz.
- Predicted relative STFT energy spans 1.3-90.5. Their mean energy values are 64.6/83.8. Character-error scoring ignores whitespace, while word-error scoring retains this split.

Whisper transcript: "then I do what the orchestra changed in respect to me."

MMS transcript: "ni ldin os te or cestel change in espectili"

[Waveform and acoustic overlay](automatic_assessment_plots/E3.png)

## E4

- Whisper WER is 56.0%; whitespace-free CER is 46.2%. Independent MMS gives WER 80.0%, CER 53.8%. Whisper has 13 word substitutions, 0 deletions and 1 insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error.
- UTMOSv2 predicts 0.69, compared with a mean of 2.77 for ten same-language held-out recordings. No waveform clipping was found.
- Utterance duration is 6.78 s. The a/m tokens each receive 23 ms. The boundary after the year ends is 406 ms.
- Predicted continuous F0 spans 105-171 Hz. Andaroo's mean is 138.5 Hz.
- Predicted relative STFT energy spans 1.6-97.9. Two zero-duration items are word-boundary tokens, not omitted lexical characters. ASR checks the expanded time/date and the mixed-language word.

Whisper transcript: "The meeting will start at 10.45 a.m. and will end around 11.06 a.m. So, there was a very good time."

MMS transcript: "the meeting ley stard and den fortifin the on got of t mmen otrini six so an le a sommary won dim"

[Waveform and acoustic overlay](automatic_assessment_plots/E4.png)
