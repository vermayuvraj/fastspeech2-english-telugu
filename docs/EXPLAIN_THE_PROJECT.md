# Explaining this implementation

FastSpeech 2 separates the linguistic sequence from its timing and prosody. An encoder maps text units to hidden vectors. Three convolutional predictors estimate log duration, continuous pitch and energy. During training, measured targets condition the hidden vectors; during inference, the predictions do. Integer repetition in the length regulator converts the short text sequence into a mel-frame sequence. The decoder and postnet predict a mel spectrogram; a separate frozen neural vocoder reconstructs a waveform.

The acoustic model is non-autoregressive: it predicts frames in parallel after durations are available. It still needs accurate timing supervision. This implementation obtains timing from a multilingual CTC forced aligner rather than using equal-duration guesses. Rounding boundaries is done once, and all durations must sum to the exact mel length.

Why two models? The assignment asks for independent language experiments. Sharing Python functions does not share learned parameters. Different seeds, data, statistics, optimizers and files preserve that distinction. Speaker conditioning is also distinct from language conditioning: Svarah has many voices, and the Telugu dataset has two. Without a voice embedding, an acoustic model can average incompatible voice characteristics.

Why romanization? The supplied English symbol inventory cannot represent Telugu directly. A shared deterministic romanizer gives the aligner and acoustic model the same units. These are graphemes, not phonemes. The tradeoff is loss of some pronunciation distinctions; a phoneme frontend would need matching alignment supervision and language-specific validation.

Why train-only statistics? Using validation audio to fit pitch/energy normalization or quantization ranges leaks held-out information. The checkpoint stores the training statistics so inference uses exactly the same transformation. Plotting reverses these transformations to show Hz and STFT energy units.

What do the losses mean? Mel and postnet L1 losses measure spectral reconstruction. Duration MSE compares log(1+frame count). Pitch and energy MSE operate in normalized log space. Padding is excluded; zero-duration units contribute no pitch/energy target. The sum is an optimization objective, not a perceptual score. Full validation is teacher-forced, while the assignment samples test the accumulated errors of predicted durations and prosody.

What should you inspect? Compare training and validation trajectories, listen for skipped/repeated units and intelligibility, check time/date verbalization, and inspect the specific stress cases in every prompt. For English “conduct,” stress placement needs a phonetic or listening check. Telugu “pattu” can have two meanings with the same pronunciation. Meaning alone is not a reason to force a pitch change.

What is not demonstrated by a smoke test? A small diagnostic establishes code execution, gradient flow, serialization and synthesis plumbing. It cannot establish five-hour training, generalization, convergence, Telugu naturalness or assignment completion. Those require the full saved runs and review.
