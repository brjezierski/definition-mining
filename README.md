# Environment setup for M1 Macs

1. Make sure to mark "Open using Rosetta" in Get info of Terminal in Applications.

2. Open the terminal and install Rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

3. Restart the terminal and verify the installation with ```rustc```.

4. Download Miniforge3-MacOSX-arm64 from [here](https://github.com/conda-forge/miniforge)
```
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```

5. ```cd``` into the definition-mining repo and run
```
conda env create -f environment.yml
conda activate ap_env
```

6. Verify PyTorch and transformers installation. Open a Python interpreter and type 
```
import torch
import transformers
```

7. TODO: verify if the M1 GPU is actually used.