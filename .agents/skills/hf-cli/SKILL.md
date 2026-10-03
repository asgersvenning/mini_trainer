---
name: hf-cli
description: Hugging Face authentication, downloads and release publication for Nemo.
---

# Hugging Face release work

Follow the [publication guide](../../../dev/releases/mambo_v3/publication.md)
for model and Space changes. Use the existing release workflows and their trusted
publishing credentials; this skill does not authorize publication.

Use an existing environment's `hf --help` and subcommand help for its installed
version. The [official CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)
owns command syntax. Do not synchronize the training environment to obtain Hub tools.
Never print tokens or commit credentials. Preserve immutable release revisions.

Adapted from [huggingface/skills](https://github.com/huggingface/skills/tree/80f9fa530e46f4ae642fcb9e1725bad0e1979395/skills/hf-cli)
under the included [Apache-2.0 license](LICENSE); the generic command catalogue
has been removed in favor of the repository workflow and upstream documentation.
