# 02465 Introduction to reinforcement learning and control theory

This repository contains code for 02465, introduction to machine learning and control theory. For installation instructions and course material please see:

 - https://www2.compute.dtu.dk/courses/02465/information/installation.html


## License
Some of the code in this repository is not written by me and the licensing terms are as indicated in the relevant files.

Files authored by me are only intended for educational purposes. Please do not re-distribute or publish this code without written permision from Tue Herlau (tuhe@dtu.dk)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

1. Install `uv` (if not already installed):
   - Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`
   - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

2. Create the virtual environment and install all dependencies:
   ```
   uv sync
   ```

3. Activate the environment:
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Linux/macOS: `source .venv/bin/activate`

## Updating packages

In case a package has been updated, please run:

```
uv sync --upgrade
```
