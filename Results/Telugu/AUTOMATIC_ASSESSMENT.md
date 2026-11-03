# Telugu automatic assessment

I do not know Telugu, so I used automatic transcription, predicted-quality scores and acoustic measurements to assess the Telugu outputs. Assessment of both languages is automatic. No native-speaker listening or human MOS study was performed.

These are automatic observations. Human listening fields remain unfilled and are not represented as completed.

## T1

- Whisper WER is 100.0%; whitespace-free CER is 24.2%. Independent MMS gives WER 83.3%, CER 14.5%. Whisper has 5 word substitutions, 0 deletions and 1 insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error.
- UTMOSv2 predicts 1.72, compared with a mean of 3.14 for ten same-language held-out recordings. No waveform clipping was found.
- Utterance duration is 6.05 s. The target word occupies 1289 ms (21% of the utterance).
- Predicted continuous F0 spans 100-294 Hz. Its duration-weighted F0 is 249.1 Hz.
- Predicted relative STFT energy spans 0.1-108.9. Its duration-weighted mean is 53.8.

Whisper transcript: "నూరో పార్మకోలాజికల్ పరిసోధనలో నూతన అవస్తదాల ప్రభావాని విషేషించారు"

MMS transcript: "న్యూరోపారమకోలాజికల్ పరిశోదనలో మూతన అవశధాల ప్రభావాన్ని విశేషించారు"

[Waveform and acoustic overlay](automatic_assessment_plots/T1.png)

## T2a

- Whisper WER is 33.3%; whitespace-free CER is 3.6%. Independent MMS gives WER 0.0%, CER 0.0%. Whisper has 1 word substitutions, 0 deletions and 1 insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error.
- UTMOSv2 predicts 1.99, compared with a mean of 3.14 for ten same-language held-out recordings. No waveform clipping was found.
- Utterance duration is 3.33 s. Pattu takes 279 ms.
- Predicted continuous F0 spans 113-247 Hz. Pattu's mean is 201.9 Hz.
- Predicted relative STFT energy spans 0.3-78.2. Pattu's mean is 19.4. The meaning difference does not itself require a different segmental pronunciation.

Whisper transcript: "టాడు పై అతని పట్టు చాలా బలంగా ఉంది"

MMS transcript: "తాడుపై అతని పట్టు చాలా బలంగా ఉంది"

[Waveform and acoustic overlay](automatic_assessment_plots/T2a.png)

## T2b

- Whisper WER is 50.0%; whitespace-free CER is 20.6%. Independent MMS gives WER 16.7%, CER 11.8%. Whisper has 3 word substitutions, 0 deletions and 0 insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error.
- UTMOSv2 predicts 1.57, compared with a mean of 3.14 for ten same-language held-out recordings. No waveform clipping was found.
- Utterance duration is 3.54 s. Pattu takes 267 ms.
- Predicted continuous F0 spans 89-252 Hz. Pattu's mean is 200.2 Hz.
- Predicted relative STFT energy spans 0.3-78.0. Pattu's mean is 16.3. The meaning difference does not itself require a different segmental pronunciation.

Whisper transcript: "ఆను పట్టు చిరనోక్ దరించి వేడుకకు వచ్చింది"

MMS transcript: "ఆమె పట్టు చీరను ఉద్దరించి వేడుకకు వచ్చింది"

[Waveform and acoustic overlay](automatic_assessment_plots/T2b.png)

## T3

- Whisper WER is 60.0%; whitespace-free CER is 6.9%. Independent MMS gives WER 40.0%, CER 0.0%. Whisper has 2 word substitutions, 1 deletions and 0 insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error.
- UTMOSv2 predicts 2.05, compared with a mean of 3.14 for ten same-language held-out recordings. No waveform clipping was found.
- Utterance duration is 3.23 s. The retained adda/viloo split receives a 81 ms boundary; the sentence's median interword boundary is 104 ms.
- Predicted continuous F0 spans 106-270 Hz. The two fragments' mean F0 values are 173.8/209.9 Hz.
- Predicted relative STFT energy spans 0.1-86.1. Their mean energy values are 38.8/53.8. Character-error scoring ignores whitespace, while word-error scoring retains this split.

Whisper transcript: "పులి అడవిలో నిస్యబ్దంగా నడిచింది"

MMS transcript: "పులి అడవిలో నిశ్శబ్దంగా నడిచింది"

[Waveform and acoustic overlay](automatic_assessment_plots/T3.png)

## T4

- Whisper WER is 52.6%; whitespace-free CER is 15.2%. Independent MMS gives WER 31.6%, CER 9.5%. Whisper has 8 word substitutions, 1 deletions and 1 insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error.
- UTMOSv2 predicts 1.74, compared with a mean of 3.14 for ten same-language held-out recordings. No waveform clipping was found.
- Utterance duration is 9.83 s. The boundary between nalabhai/aidu (forty/five) is 348 ms; the later clause boundary is 453 ms.
- Predicted continuous F0 spans 100-258 Hz. Mean word F0 falls from 200.3 Hz at the start to 121.1 Hz at the end.
- Predicted relative STFT energy spans 0.1-91.4. One zero-duration word-boundary token occurs in the year; no lexical token has zero duration. This is not evidence of an omitted spoken word.

Whisper transcript: "ఈ సమావేసం వెండు వేలా ఇరవై ఆరు నవంబర్ పంన్నండవతేది ఉదేయం పది గంటల నలబై ఐదు నిమిషాలకు ప్రారంభం హోతుంది సమయానికి సిద్ధంగా ఉ"

MMS transcript: "ఈ సమావేశం 2,0020 6 నవంబర్ 1ెండవ తేది ఉదయం పదగంటల నలభై ఐదు నిమిషాలకు ప్రారంభమౌుతుంది సమయానికి సిద్ధంగా ఉంచండి"

[Waveform and acoustic overlay](automatic_assessment_plots/T4.png)
