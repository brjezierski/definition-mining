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

8. To use nltk open python and run the following
```
import nltk
nltk.download('punkt')
```

9. To use spacy for German, first install the following package
```
spacy download de_dep_news_trf
```

# Models

For English, a multitask large uncased BERT model was used to predict whether a phrase contains a definition and to predict the tag for each word within the definition. For German, a GBERT model published by deepset was used. Trained models can be found on huggingface for both [English](https://huggingface.co/brjezierski/def_mining_en) and [German](https://huggingface.co/brjezierski/def_mining_de).

## Best F1 scores

The models were trained for 6 epochs. For the last two epochs only the tag sequence loss was used to further improve the performance for the tag labeling task.

|                     | EN     | DE     |
|---------------------|--------|--------|
| type sent 1         | 0.79   | 0.77   |
| tag weighted avg    | 0.70   | 0.66   |


# Tags
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

The following script is used to create a dataset that can be used for training from the format of DeftEval corpus provided for SemEval 2020. [The DeftEval repository](https://github.com/Elzawawy/DeftEval) need to be cloned into the definition-mining directory before running the script. The following command creates an English dataset ready for training 

```
python create_dataset.py --target_dir data/single/en --lang en --sent_aggregation single --translation_dir ../data/de
```

In order to create a German dataset a list of translations is needed in the tsv format with columns named Text and Translation. Three files called train, test and dev are required to be in the same directory which can be passed using the `translation_dir` flag. The dataset for training of the definition mining model can include each sentence as a separate training example (`--sent_aggregation single`), group sentences into training examples of three sentences each in order to increase the accuracy of detecting definitions spanning over one sentence (`--sent_aggregation window`) or both (`--sent_aggregation both`). The script can be easily modified to any data augmentation task in which sentences with each word labeled with a different tag need to be translated together with the tags into another language. For the training of our models we used single sentence aggregation.

## Training the model

The description of input flags can be found using a help command. The hyperparameter values were chosen based on the results of experiments by Davletov et al. 2020. 

Below are the examples of commands we used for training. The following is for training an English model for 2 epochs from a pre-trained BERT model.
```
python train.py --language en \
                --data_dir data/single/en \
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
                --lr_schedule linear_warmup \
                --seed 42 \
                --learning_rate 1e-5 \
                --weight_decay 0.1 \
                --dropout 0.1 \
                --tags_sequence_clf_weight 1 \
                --sent_type_clf_weight 1 \
                --output_dir training/en-2e
```

And the following is used for training from the checkpoint saved in training/de-gbert_4e while only optimizing for the tags sequence loss.
```
python train.py --language de \
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

## Extract definitions
The input file should be in the JSON format with a phrase per row in a column whose title is passed to the `--text_column` flag. The most relevant flags are displayed in the example below. The two languages supported are `de` and `en` which use the models available in our huggingface repo. Instead, a local directory with the model called `pytorch_model.bin` can be passed as an argument to the flag `--model_dir`. More flags are avilable with the help command. They overlap with some of the flags of the training script. The output file can be found in the labeled_data directory.

```
python mine_definitions.py --input_file ../data/de/kurier_phrases.json --language de --text_column phrase
```

## Show definitions from the example German corpus
To display color coded reaults of using a definition extraction model use `display_tagged_sentences.py` script with a flag `--show_only_def`. To analyze the results as compared to searching for definitions by looking for a specific substring (e.g. ist), include a flag `-substr ist`.

```
python display_tagged_sentences.py --input_file labeled_phrases/kurier_phrases.tsv --show_only_def
```

## Push the model to huggingface

Before running the script, it may be necesssary to [configure the repository for huggingface](https://huggingface.co/docs/transformers/model_sharing#setup). An example command:

```
python upload_model_to_huggingface.py --language de --username brjezierski --model_name def_mining_de --commit_msg "Trained on 6 epochs" --model_dir training/de-gbert_6e
```
