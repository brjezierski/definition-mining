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
conda activate dm_env
```

6. Install PyTorch using pip
```
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

7. Verify PyTorch and transformers installation. Open a Python interpreter and type 
```
import torch
torch.has_mps # should be True
import transformers
```

8. To use spacy for German, first install the following package
```
spacy download de_dep_news_trf
```

## Input format
The input file should be in the JSON fromat with a sentence per row. Each sentence should not be longer than 256 characters.

# Commands for training

1. Training from checkpoint
```
python train.py --model deepset/gbert-large \
                --data_dir data/single/de \
                --num_train_epochs 2 \
                --eval_per_epoch 4 \
                --max_seq_length 256 \
                --do_train --do_validate \
                --train_mode random_sorted \
                --train_batch_size 32 \
                --eval_batch_size 8 \
                --max_grad_norm 1.0 \
                --warmup_proportion 0.1 \
                --gradient_accumulation_steps 8 \
                --subtokens_pooling_type first \
                --sequence_mode not-all \
                --lr_schedule linear_warmup \
                --seed 42 \
                --learning_rate 1e-5 \
                --weight_decay 0.1 \
                --dropout 0.1 \
                --tags_sequence_clf_weight 0 \
                --sent_type_clf_weight 1 \
                --output_dir training/de-gbert_6e-0-1 \
                --checkpoint_dir training/de-gbert_4e
```


## Results

|                 | 3 epochs | 10 epochs |
|-----------------|----------|-----------|
| type 0 F1       | 0.89     |           |
| type 1 F1       | 0.77     |           |
| loss            | 0.48     |           |
| macro avg F1    | 0.83     |           |
| weighted avg F1 | 0.85     |           |

# DEFT data

## Tags
B- prefix for the first term and I- for all subsequent terms.  

* Term - a primary term
* Alias Term - a secondary, less common name for the primary term. Links to a term tag.
* Ordered Term  - Multiple terms that have matching sets of definitions which cannot be separated from each other without creating an non-contiguous sequence of tokens. E.g. x and y represent positive and negative versions of the definition, respectively
* Referential Term - An NP reference to a previously mentioned term tag. Typically this/that/these + NP
* Definition - A primary definition of a term. May not exist without a matching term.
* Secondary Definition - Supplemental information that may qualify as a definition sentence or phrase, but crosses a sentence boundary.
* Ordered Definition - Multiple definitions that have matching sets of terms which cannot be separated from each other. See Ordered Term.
* Referential Definition - NP reference to a previously mentioned definition tag. See Referential Term.
* Qualifier - A specific date, location, or condition under which the definition holds


# How to use the tool
First remember to switch to the right environment. 

## Show definitions from the example German corpus
```
python display_tagged_sentences.py --input_file training/de-gbert_4e/labeled_phrases.tsv --show_only_def
```

## And English
```
python display_tagged_sentences.py --input_file ../data/glanos/aerospace_def_labels.tsv --show_only_def
```

## Show sentences containing a specific substring
```
python display_tagged_sentences.py --input_file training/de-gbert_4e/labeled_phrases.tsv --substr ist
```

## To label a dataset
```
python mine_definitions.py --input_file ../data/de/kurier_phrases.json --language de --text_column phrase
```