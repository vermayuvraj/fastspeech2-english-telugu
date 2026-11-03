# Sources and attribution

* FastSpeech 2 paper: https://arxiv.org/abs/2006.04558
* Reference code: https://github.com/ming024/FastSpeech2, commit `d4e79eb52e8b01d24703b2dfc0385544092958f3`. Its MIT license is retained in `vendor/ming024/LICENSE`. Modified files are supplied, with a reviewable patch and hash manifest.
* English corpus: https://huggingface.co/datasets/ai4bharat/Svarah. The dataset card identifies CC BY 4.0 and requires account acceptance before file access. Cite Javed et al., “Svarah: Evaluating English ASR Systems on Indian Accents,” Interspeech 2023, pp. 5087–5091.
* Telugu corpus: https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu. The card refers to the original Indic TTS terms at https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf. That original link timed out during this setup; its full terms have not been independently verified. The raw corpus is not included in this repository. Acknowledge the Speech Technology Consortium, IIT Madras, Hema A. Murthy and S. Umesh.
* Alignment: official TorchAudio MMS_FA asset and tutorial: https://docs.pytorch.org/audio/2.6.0/tutorials/forced_alignment_for_multilingual_data_tutorial.html. MMS is pretrained external alignment supervision, not the submitted acoustic model. MMS weights are not redistributed here; consult the MMS noncommercial model terms for use beyond this research evaluation.
* Romanization: https://github.com/isi-nlp/uroman, Ulf Hermjakob, Jonathan May and Kevin Knight, ACL 2018 demonstration paper.
* Vocoder: https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz, pinned revision `4188503131602dc234f48d7f22eebea93d788736`, published SpeechBrain model. The model card identifies Apache 2.0. HiFi-GAN architecture: https://github.com/jik876/hifi-gan. The generator code supplied by the reference repository is covered by its retained license.

New experiment code was developed with AI assistance. Experimental results must come from saved runs; this note does not imply that any pending experiment has been performed. No new license is asserted over third-party datasets or pretrained weights.

## Automatic assessment tools

* Whisper large-v3 is decoded with [SYSTRAN faster-whisper](https://github.com/SYSTRAN/faster-whisper), using the pinned `Systran/faster-whisper-large-v3` conversion. It is an evaluation recognizer, not a source of TTS acoustic-model weights.
* The independent recognizer is [Meta MMS-1b-all](https://huggingface.co/facebook/mms-1b-all), with `eng` and `tel` adapters. Its model card identifies CC BY-NC 4.0. These pretrained weights are downloaded to the external research cache and are not redistributed in this repository.
* Quality predictions use the [official SaruLab UTMOSv2 implementation](https://github.com/sarulab-speech/UTMOSv2), commit `cc2700db57bb83ee13dc31ebe1b868c254e15d09`, and its published `fusion_stage3` fold-0 checkpoint. Its pretrained backbones are Facebook wav2vec2-base and timm EfficientNetV2-S. These evaluation weights are not submitted as trained assignment models and are not redistributed here.
* Exact evaluation-model revisions are in `docs/automatic_evaluation_models.json`; the method and its limitations are documented in `docs/AUTOMATIC_ASSESSMENT.md`.
