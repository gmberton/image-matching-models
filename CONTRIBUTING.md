Thanks for your interest in contributing to IMM!

To add a new method:
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

6. Format the code with [Black](https://github.com/psf/black), like this
```
pip install black
cd image-matching-models && black --line-length 120 ./
```

7. Test your model and submit a PR!

Note: as authors update their model repos, consider updating the submodule reference here using the below:
To update a submodule to the head of the remote, run 
```bash
git submodule update --remote matching/third_party/[submodule_name]
```
