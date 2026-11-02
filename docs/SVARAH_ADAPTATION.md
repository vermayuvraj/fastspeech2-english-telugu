# Svarah schema and conditioning

The pinned Hugging Face release contains an `audio_filepath` Audio struct, `duration`, `text`, and demographic fields. It does not expose a `speaker_id` column. Recording filenames contain recording/file identifiers, not a verified mapping to the 117 speakers described in the dataset card. They are not treated as speaker IDs.

The English model conditions on deterministic profile groups formed from gender, age group, primary language, native state, native district, highest qualification, job category and occupation domain. The ordered field values are SHA256-hashed to a local group identifier. Raw demographic values are not added to the public training manifests. These groups are proxies: different people with the same full profile will share an embedding. The implementation does not claim to recover ground-truth speaker identities.

Telugu uses the dataset's explicit two-voice gender labels. This schema-specific choice is configured explicitly; the general loader never silently substitutes gender for missing speaker identifiers.

The English split remains disjoint by utterance ID, decoded-audio hash and normalized transcript. Profile groups intentionally overlap between training and validation. An authenticated download is required for the exact dataset revision, and interrupted downloads are resumed with bounded retries for connection failures.
