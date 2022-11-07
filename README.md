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
torch.has_mps # should be True
import transformers
```

7. TODO: verify if the M1 GPU is actually used.

# Testing deft-eval-2020 implementation

1. Run the python script.
```
python run_defteval.py --model bert-large-uncased --data_dir data --eval_per_epoch 4 --max_seq_length 256 --do_train --do_validate --train_mode random_sorted --train_batch_size 32 --eval_batch_size 8 --num_train_epochs 7 --max_grad_norm 1.0 --warmup_proportion 0.1 --gradient_accumulation_steps 8 --filter_task_3 --subtokens_pooling_type first --sequence_mode not-all --lr_schedule linear_warmup --seed 42 --learning_rate 1e-5 --weight_decay 0.1 --dropout 0.1 --sent_type_clf_weight 0 --tags_sequence_clf_weight 1 --relations_sequence_clf_weight 0 --output_dir bert_stype_tags_first_slen_256_eval_perep_4_linear/lr-1e-5-wd-0.1-drp-0.1-w1-0-w2-1-w3-0
```

## Issues 


# DEFT data

## Tags
B- prefix for the first term and I- for all subsequent terms. 

Term - a primary term
Alias Term - a secondary, less common name for the primary term. Links to a term tag.
Ordered Term  - Multiple terms that have matching sets of definitions which cannot be separated from each other without creating an non-contiguous sequence of tokens. E.g. x and y represent positive and negative versions of the definition, respectively
Referential Term - An NP reference to a previously mentioned term tag. Typically this/that/these + NP
Definition - A primary definition of a term. May not exist without a matching term.
Secondary Definition - Supplemental information that may qualify as a definition sentence or phrase, but crosses a sentence boundary.
Ordered Definition - Multiple definitions that have matching sets of terms which cannot be separated from each other. See Ordered Term.
Referential Definition - NP reference to a previously mentioned definition tag. See Referential Term.
Qualifier - A specific date, location, or condition under which the definition holds
## Schema