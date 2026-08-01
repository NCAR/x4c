---
title: Installation
---

## Install the Conda environment

You may skip this step if your Conda environment has been installed already.

### Step 1: Download the installation script for miniconda3

#### macOS (Intel)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
```

#### macOS (Apple Silicon)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
```

#### Linux

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### Step 2: Install Miniconda3

```bash
chmod +x Miniconda3-latest-*.sh && ./Miniconda3-latest-*.sh
```

During the installation, a path `<base-path>` needs to be specified as the base location of the python environment.
After the installation is done, we need to add the two lines into your shell environment (e.g., `~/.bashrc` or `~/.zshrc`) as below to enable the `conda` package manager (remember to change `<base-path>` with your real location):

```bash
export PATH="<base-path>/bin:$PATH"
. <base-path>/etc/profile.d/conda.sh
```

### Step 3: Test your Installation

```bash
source ~/.bashrc  # assume you are using Bash shell
which python  # should return a path under <base-path>
which conda  # should return a path under <base-path>
```

## Install `x4c`

Taking a clean installation as example, first let's create a new environment named `x4c-env` via `conda`

```bash
conda create -n x4c-env python=3.13   # supports Python 3.12 and 3.13
conda activate x4c-env
```

Then install some dependencies via `conda`:

```bash
conda install -c conda-forge jupyter notebook xesmf cartopy geocat-comp
```

Once the above dependencies have been installed, simply

```bash
pip install x4c
```

and you are ready to

```python
import x4c
```

in Python.
