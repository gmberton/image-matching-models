# install script for IMM

# need to install editable dependencies seperately
# since editable deps are not supported by pyproject.toml
pip install -e ./matching/third_party/UFM/UniCeption
pip install -e ./matching/third_party/UFM

# now, install the rest of IMM 
# deps will be installed via pyproject.toml
pip install -e .