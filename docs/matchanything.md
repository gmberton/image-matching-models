# MatchAnything (ELoFTR / RoMa)

The `matchanything-eloftr` and `matchanything-roma` wrappers use the upstream MatchAnything repo (HF Space: https://huggingface.co/spaces/LittleFrog/MatchAnything), included here as a git submodule at `matching/third_party/MatchAnything`.

## Submodule setup

If you cloned without submodules:

```bash
git submodule update --init --recursive matching/third_party/MatchAnything
```

## Weights cache location

Checkpoints are cached under `matching/model_weights/matchanything`:

- `matchanything_eloftr.ckpt`
- `matchanything_roma.ckpt`

The wrapper will also reuse checkpoints previously downloaded to the legacy location under the MatchAnything submodule.

## Optional: manual prefetch (PowerShell)

```powershell
New-Item -ItemType Directory -Force matching/model_weights/matchanything | Out-Null
python -m gdown 12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d --fuzzy -O matching/model_weights/matchanything/weights.zip
tar -xf matching/model_weights/matchanything/weights.zip -C matching/model_weights/matchanything
Remove-Item matching/model_weights/matchanything/weights.zip
```

If `tar` is unavailable:

```powershell
Expand-Archive -Path matching/model_weights/matchanything/weights.zip -DestinationPath matching/model_weights/matchanything -Force
Remove-Item matching/model_weights/matchanything/weights.zip
```

