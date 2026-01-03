@echo off

REM install script for IMM
REM need to install editable dependencies separately
REM since editable deps are not supported by pyproject.toml

pip install -e .\matching\third_party\UFM\UniCeption
pip install -e .\matching\third_party\UFM

REM now, install the rest of IMM
REM deps will be installed via pyproject.toml
pip install -e .