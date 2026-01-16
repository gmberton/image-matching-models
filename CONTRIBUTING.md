# To add a new method:
Let's for example add a matcher called new_matcher.
1. Copy the template to create a new file: `cp imm/TEMPLATE.py imm/im_models/new_matcher.py`
2. If the method requires external modules (for example the offical repository of new_matcher), use `git submodule add`: for example, I used this command to add the LightGlue module
   ```bash
   git submodule add https://github.com/cvg/LightGlue imm/third_party/LightGlue
   ```
   This command automatically modifies `.gitmodules` (you should not modify `.gitmodules` manually!), and when cloning the repository it will automatically clone also the LightGlue repo in `imm/third_party`.

3. In `imm/im_models/new_matcher.py` you only need to implement the method `_forward`, which takes two image tensors as input and returns 6 objects: `[mkpts0, mkpts1, kpts0, kpts1, desc0, desc1]`. The template has more details on how to implement the class.

4. Open `imm/__init__.py` and add the model name (all lowercase) to the `available_models` list. Add an `elif` case to instantiate the class, as for the other matchers.

5. If it requires additional dependencies, add them to `requirements.txt` or to the `[project.optional-dependencies]` of `pyproject.toml`.

6. Format the code with [ruff](https://github.com/astral-sh/ruff)
   ```
   ruff format .
   ruff check --fix .
   ```

7. Test your model. Make sure the model weights are downloaded automatically in the code, either with huggingface_hub (if on HF), gdown (if on GDrive), or py3_wget (any other platform or HTTP URL).
   ```
   # Run this and have a look at the generated images in outputs_new_matcher
   python imm_match.py --matcher new_matcher --out_dir outputs_new_matcher
   # Run this and make sure it passes the test
   python imm_benchmark.py --matcher new_matcher
   ```
   Now submit a PR!

Note: as authors update their model repos, consider updating the submodule reference here using the below:
To update a submodule to the head of the remote, run 
```bash
git submodule update --remote imm/third_party/[submodule_name]
```

## Optional: add docs
You can create the file `docs/new_matcher.md` to explain how the model is used. This is especially useful if the model has multiple hyperparameters.

## Optional: adding model weights to the Hugging Face Hub

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

2. Although weights can be uploaded as PyTorch file, safetensor is preferred. Save the state dict as a [safetensor](https://huggingface.co/docs/safetensors/en/index)
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
