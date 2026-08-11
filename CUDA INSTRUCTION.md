# CUDA Installation Instructions


TensorFlow with CUDA available only:

1. Python 3.7–3.10 + TensorFlow 2.10 without WSL
2. WSL + Python >=3.11.x

## Option 1. Python 3.7–3.10 + TensorFlow 2.10 without WSL
  Install exactly CUDA 11.2 + cuDNN 8.1 (not 13.3/9.x),




## Option 2. WSL

> PyCharm work with WSL only in PRO version ( subscription or license required). If you want to use PyCharm with WSL, you need to have the PRO version.

1. Install WSL

```shell
wsl --install
```

2. Reboot your computer.
3. Open PowerShell and check the available WSL distributions:

```shell
wsl --list --online
``` 

4. Install latest ubuntu distribution (e.g., Ubuntu 22.04 LTS) for WSL2:

```shell
wsl --install -d Ubuntu-26.04
```

5. Create env on WSL
- Add new Iterpreter -> Select WSL -> Select your WSL distribution (e.g., Ubuntu-22.04) -> Create new environment (e.g., conda or venv) -> Install required packages (e.g., TensorFlow, PyTorch, etc.)

6. Update repo in WSL ( For TensorFlow required Python version 3.13.x)

```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13-full
cd ~
sudo python3.13 -m venv .venv
sudo chown -R $USER:$USER ~/.venv
source .venv/bin/activate
pip install -r ../path/to/project/requirements.txt
```

7. Check if CUDA is available in WSL:

```shell
# add WSL GPU shim + pip CUDA libs to runtime linker path
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$(python - <<'PY'
import site,glob,os
dirs=[]
for p in site.getsitepackages()+[site.getusersitepackages()]:
    dirs += glob.glob(os.path.join(p, "nvidia", "*", "lib"))
print(":".join(d for d in dirs if os.path.isdir(d)))
PY
):$LD_LIBRARY_PATH
```

```shell
# check if TensorFlow is available
python -c "import tensorflow as tf; print(tf.__version__)"
```

```shell
# check if GPU is available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Conclusion:

>  For Windows and Windows + WSL id doesn't make sense at all.
>  CUDA doesn't work correctly, Python version is old, many limitations and issues.