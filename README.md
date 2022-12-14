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
# Setup
- learning rate = 1e-5  
- weight decay = 0.1
- dropout = 0.1
- for subtask 1 we will use w1-2=1
- for subtask 2 we will use w1=0, w2=1


# Commands for training

1. Original command
```
python run_defteval.py --model bert-large-uncased --data_dir data --eval_per_epoch 4 --max_seq_length 256 --do_train --do_validate --train_mode random_sorted --train_batch_size 32 --eval_batch_size 8 --num_train_epochs 7 --max_grad_norm 1.0 --warmup_proportion 0.1 --gradient_accumulation_steps 8 --filter_task_3 --subtokens_pooling_type first --sequence_mode not-all --lr_schedule linear_warmup --seed 42 --learning_rate 1e-5 --weight_decay 0.1 --dropout 0.1 --sent_type_clf_weight 0 --tags_sequence_clf_weight 1 --relations_sequence_clf_weight 0 --output_dir bert_stype_tags_first_slen_256_eval_perep_4_linear/lr-1e-5-wd-0.1-drp-0.1-w1-0-w2-1-w3-0
```

2. Current training for 3 epochs:
```
python run_defteval.py --model bert-large-uncased --data_dir data --eval_per_epoch 4 --max_seq_length 256 --do_train --do_validate --train_mode random_sorted --train_batch_size 32 --eval_batch_size 8 --num_train_epochs 3 --max_grad_norm 1.0 --warmup_proportion 0.1 --gradient_accumulation_steps 8 --filter_task_3 --subtokens_pooling_type first --sequence_mode not-all --lr_schedule linear_warmup --seed 42 --learning_rate 1e-5 --weight_decay 0.1 --dropout 0.1 --sent_type_clf_weight 1 --tags_sequence_clf_weight 1 --relations_sequence_clf_weight 0 --output_dir main/13-12-22

python run_defteval.py --model bert-large-uncased --data_dir data --eval_per_epoch 4 --max_seq_length 256 --do_eval --train_mode random_sorted --train_batch_size 32 --eval_batch_size 8 --num_train_epochs 3 --max_grad_norm 1.0 --warmup_proportion 0.1 --gradient_accumulation_steps 8 --filter_task_3 --subtokens_pooling_type first --sequence_mode not-all --lr_schedule linear_warmup --seed 42 --learning_rate 1e-5 --weight_decay 0.1 --dropout 0.1 --sent_type_clf_weight 1 --tags_sequence_clf_weight 1 --relations_sequence_clf_weight 0 --output_dir main/13-12-22
```

3. Next training (same model but 10 epochs):
```
python run_defteval.py --model bert-large-uncased --data_dir data --eval_per_epoch 4 --max_seq_length 256 --do_train --do_validate --train_mode random_sorted --train_batch_size 32 --eval_batch_size 8 --num_train_epochs 10 --max_grad_norm 1.0 --warmup_proportion 0.1 --gradient_accumulation_steps 8 --filter_task_3 --subtokens_pooling_type first --sequence_mode not-all --lr_schedule linear_warmup --seed 42 --learning_rate 1e-5 --weight_decay 0.1 --dropout 0.1 --sent_type_clf_weight 1 --tags_sequence_clf_weight 1 --relations_sequence_clf_weight 0 --output_dir main/w1-1-w2-1-e-10
```

## Results

|                 | 3 epochs | 10 epochs |
|-----------------|----------|-----------|
| type 0 F1       | 0.89     |           |
| type 1 F1       | 0.77     |           |
| loss            | 0.48     |           |
| macro avg F1    | 0.83     |           |
| weighted avg F1 | 0.85     |           |

## Issues 


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

# Evaluation of labels on aerospace dataset 
Partial definitions explain one of the functions of the term (aas opposed to full definitions).
Features of full definitions:  
  - program is an ... program that
  - is the
  - are the
  - apposition: belly cargo business, using...; ..., an initiative...; 
  - recognises
  - outlined
  - project aims
  - Program (AIP) to help
  - Through their Cool Effect Program, American Airlines allows their passengers to reduce their carbon footprints

Partial:
  - is based on 
  - pursues
  - which are generated
  - in the form of
  - allows
  - permits
  - is to
  - including (includes)
  - enabled
  - provides
  - is a ... way
  - aims
  - who oversees
  - legislation allows
features:
  - for more abstract terms (legislation, project, programme) more verbs make sense


Examples of non-definitions labeled as definitions:
  - Sustainable Aviation Fuels (SAF) are the first viable alternative to fossil kerosene and a key lever to reduce CO2 emissions in the aviation industry
  - The jet fuel spike *is a reversal* of fortunes after the collapse in air travel during the onset of Covid
  - machine is a copying-milling machine ...
  - means that
  - is engaged
  - Provisions for customer claims comprise
  - a new rule that would allow ...
  - enhances
  - contribute significantly
  - plays a key role
  - always, flying and holidaying with Jet2 **means** that customers receive the best customer service but also the knowledge of ATOL holiday protected holidays
And features:
  - the term is a general descriptor, such as machine or the air show. Its referant must have been in a previous sentence. Using a window of 2-3 sentences could help. Also using named entities labeling could address this issue.
  - extracting suggested pairs could help for examples like "outreach programme,Reach for the Sky" where Reach for the Sky is marked as a term but outreach programme is not marked as a definition
  - sometimes there are no labels for a sentence, sometimes correctly e.g. carbon reduced sub-regional aviation that is cost-effective, safe, reliable; and sometimes incorrectly, e.g. Xiamen Airlines Co Ltd (Xiamen Airlines), a subsidiary of China Southern Airlines Company Ltd, is an air transportation service provider, based in China,

Labeled non-definitions with **is a** contruction
  - It is very clear that 2020 is a completely abnormal situation
  - is a significant factor
  - Hong Kong is a free port that can help many companies - too general
  - The growth ... is a major contributor
  - is a major factor
  - this is
  - is a clear sign

# Code modifications

relations_sequence_clf_weight - we do not need
sent_type_clf_weight - we need
tags_sequence_clf_weight - we need