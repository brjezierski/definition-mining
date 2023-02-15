# Project overview

This repository contains scripts for training a definition extraction model as well as using the model to mine for definitions. It also contains additional scripts described in the sections below. The script for training was based on [the submission by Davletov et al.](https://aclanthology.org/2020.semeval-1.59.pdf) for the [Task 6 of SemEval-2020](https://aclanthology.org/2020.semeval-1.41.pdf). 

# DEFT data

Data used for the training came from the [Task 6 of SemEval-2020 conference](https://aclanthology.org/2020.semeval-1.41.pdf). The training set contains 24,184 sentences, the validation set 1,179 sentences, and testing set 1,189. Each sentence is labeled with a binary label (sent_type) indicating whether it contains a definition or not as well as a sequence of tags (one tag per word) indicating which words are part of a definition and in which capacity. The overview of the tags can be found below.

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


## Best scores

|                     | EN     | DE     |
|---------------------|--------|--------|
| type sent 1 F1      | 0.79   | 0.77   |
| tag weighted avg F1 | 0.70   | 0.66   |


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


# Scripts overview
Before running any script, remember to activate the environment dm_env. 

## Creating a dataset 

The following script is used to create a dataset that can be used for training from the format of DeftEval corpus provided for SemEval 2020. [The DeftEval repository](https://github.com/Elzawawy/DeftEval) need to be clones into the definition-mining directory before running the script. The following command creates an English dataset ready for training 

```
python create_dataset.py --target_dir data/single/en --lang en --sent_aggregation single
```

In order to create a German dataset a list of translations 
## Training 

### Input format
The input file should be in the JSON fromat with a sentence per row. Each sentence should not be longer than 256 characters.

### Commands
The description of input flags can be found using a help command. The hyperparameter values were chosen based on the results of experiments by Davletov et al. 2020. 

Training from scratch saved in training/de-gbert_4e while only optimizing for the tags sequence loss.
```
python train.py --language en \
                --data_dir data/single/en \
                --num_train_epochs 2 \
                --eval_per_epoch 400 \
                --max_seq_length 256 \
                --do_train --do_validate \
                --train_mode random_sorted \
                --train_batch_size 32 \
                --eval_batch_size 8 \
                --max_grad_norm 1.0 \
                --warmup_proportion 0.1 \
                --gradient_accumulation_steps 8 \
                --subtokens_pooling_type first \
                --lr_schedule linear_warmup \
                --seed 42 \
                --learning_rate 1e-5 \
                --weight_decay 0.1 \
                --dropout 0.1 \
                --tags_sequence_clf_weight 1 \
                --sent_type_clf_weight 1 \
                --output_dir training/de-gbert_6e-0-1 \
                --checkpoint_dir training/de-gbert_4e
```

Training from the checkpoint saved in training/de-gbert_4e while only optimizing for the tags sequence loss.
```
python train.py --model de \
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

## Show definitions from the example German corpus
```
python display_tagged_sentences.py --input_file output/labeled_phrases.tsv --show_only_def
```
To analyze the results as compared to searching for definitions by looking for a specific substring (e.g. ist), include a flag `-substr ist`.


## Label a dataset
```
python mine_definitions.py --input_file ../data/de/kurier_phrases.json --language de --text_column phrase
```