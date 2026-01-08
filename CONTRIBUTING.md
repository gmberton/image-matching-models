Thanks for your interest in contributing to IMM!

## To add a new method:

1. Create a new file in the `matching/im_models` folder called `[method].py`
2. If the method requires external modules, add them to `./matching/third_party` with `git submodule add`: for example, I've used this command to add the LightGlue module which is automatically downloaded when using `--recursive`

```bash
git submodule add https://github.com/cvg/LightGlue matching/third_party/LightGlue
```
This command automatically modifies `.gitmodules` (and modifying it manually doesn't work).

3. Add the method by subclassing `BaseMatcher` and implementing `_forward`, which takes two image tensors as input and returns a dict with keys `['num_inliers','H', 'mkpts0', 'mkpts1', 'inliers0', 'inliers1', 'kpts0', 'kpts1', 'desc0', desc1']`. The value of any key may be 0, if that model does not produce that output, but they key must exist. See `TEMPLATE.py` for an example.
<br></br>You may also want to implement `preprocess`, `download_weights`, and anything else necessary to make the model easy to run. 

4. Open `__init__.py` and add the model name (all lowercase) to the `available_models` list.
<br></br>Add an `elif` case to `get_matcher()` with this model name, following the template from the other matchers. 

5. If it requires additional dependencies, add them to `requirements.txt` or to the `[project.optional-dependencies]` of `pyproject.toml`.

6. Format the code with [ruff](https://github.com/astral-sh/ruff)
```
ruff format .
ruff check --fix .
```

7. Test your model and submit a PR!

Note: as authors update their model repos, consider updating the submodule reference here using the below:
To update a submodule to the head of the remote, run 
```bash
git submodule update --remote matching/third_party/[submodule_name]
```

## Adding models to the Hugging Face Hub:

Although not mandatory, we encourage authors to upload their models to the Hugging Face Hub, under the [image matching organization](https://huggingface.co/image-matching-models). This will increase model visibility and help track usage of each model.

Here are the steps that we took to add the ELoFTR model:

1. Dowloaded the model from Google Drive (any other storage)
```py
!pip install -q pytorch_lightning  # Needed for the ELoFTR download
from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import upload_file
import gdown
import torch

weights_src = "https://drive.google.com/file/d/1jFy2JbMKlIp82541TakhQPaoyB5qDeic/view"
model_path = "eloftr_outdoor.ckpt"
gdown.download(weights_src, output=model_path, fuzzy=True)
```

2. Save the state dict as a [safetensor](https://huggingface.co/docs/safetensors/en/index)
```py
state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)["state_dict"]
save_file(state_dict, "eloftr_outdoors.safetensors")
```

3. Upload the safetensor file to the Hub (you can upload it to your personal account and later transfer to the organization)
```
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="eloftr_outdoors.safetensors",
    path_in_repo="eloftr_outdoors.safetensors ",
    repo_id="ariG23498/eloftr", # personal repository
)
```

You can access the weights here: https://huggingface.co/ariG23498/eloftr

You can now see in this [PR](https://github.com/alexstoken/image-matching-models/pull/46) how we can move holistically to the Hub.
