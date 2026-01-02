# MatchAnything (ELoFTR / RoMa)

The `matchanything-eloftr` and `matchanything-roma` wrappers use the upstream MatchAnything repo (HF Space: https://huggingface.co/spaces/LittleFrog/MatchAnything), included here as a git submodule at `matching/third_party/MatchAnything`.

## Submodule setup

If you cloned without submodules:

```bash
git submodule update --init --recursive matching/third_party/MatchAnything
```

## Use

Run either variant via:
```bash
# ELoFTR backbone (defaults to 832px NPE size)
python main_matcher.py --matcher matchanything-eloftr --device cuda --im_size 832 --out_dir outputs_matchanything-eloftr

# RoMa backbone (AMP disabled on CPU automatically)
python main_matcher.py --matcher matchanything-roma --device cuda --im_size 832 --out_dir outputs_matchanything-roma
```
Weights download automatically on first MatchAnything use and are cached under `matching/model_weights/matchanything`.
For submodule setup and troubleshooting, see [docs/matchanything.md](docs/matchanything.md).

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

