
# Resources

[SemEval-2020 Task 6: Definition extraction from free text with the DEFT corpus](https://aclanthology.org/2020.semeval-1.41.pdf)
  - three-sentence window around a potential term specifically for this task 
  - Subtask 1: Sentence classification: a [script](https://github.com/adobe-research/deft_corpus/blob/master/task1_converter.py) to convert the existing dataset in its sequence-labeling format to individual sentences with a binary label indicating whether or not the sentence contained a definition 
  - Subtask 2: Sequence labeling: 
  - Subtask 3: Relation Extraction
[DEFT: A corpus for definition extraction in free- and semi-structured text](https://aclanthology.org/W19-4015.pdf)
  - 
[Automated detection and annotation of term definitions in German text corpora](http://www.lrec-conf.org/proceedings/lrec2006/pdf/128_pdf.pdf) - possibly has a German definition dataset
[Weakly Supervised Definition Extraction](https://aclanthology.org/R15-1025.pdf) - presents a W00 dataset

https://hal.archives-ouvertes.fr/hal-01798704/ (Huge dataset with german words definitions)

# Approaches
  - fine-tune a language model (at inference, first fine-tune the model on the large corpus available)
  - definition labeling

# Tasks
  - find a definition corpus
  - find "difficult" definitions: those whose term-definition pair span crosses a sentence boundary and those lacking explicit definition phrases.

# Keywords
  - definition extraction
  - information extraction
  - hypernym detection

# Data
  - WCL - a portion of the data was extracted by taking the first sentences of randomly sampled Wikipedia articles
  - ukWaC
  - a large crawled dataset of the .uk domain name
  - W00
  - [DEFT corpus](https://github.com/adobe-research/deft_corpus)

# Things to check

Geyken, A., Barbaresi, A., Didakowski, J., Jurish, B., Wiegand, F., and Lemnitzer, L. (2017a). Die Kor- pusplattform des ”Digitalen Wo ̈rterbuchs der deutschen Sprache” (DWDS). Zeitschrift fu ̈r germanistische Lin- guistik, 45(2):327–344.