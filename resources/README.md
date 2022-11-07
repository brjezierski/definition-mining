
# Resources

[SemEval-2020 Task 6: Definition extraction from free text with the DEFT corpus](https://aclanthology.org/2020.semeval-1.41.pdf)
  - three-sentence window around a potential term specifically for this task 
  - Subtask 1: Sentence classification: a [script](https://github.com/adobe-research/deft_corpus/blob/master/task1_converter.py) to convert the existing dataset in its sequence-labeling format to individual sentences with a binary label indicating whether or not the sentence contained a definition 
  - Subtask 2: Sequence labeling: 
  - Subtask 3: Relation Extraction  

[DEFT: A corpus for definition extraction in free- and semi-structured text](https://aclanthology.org/W19-4015.pdf)
  - annotated by humans?
  - [dataset](https://github.com/adobe-research/deft_corpus)  

[Automated detection and annotation of term definitions in German text corpora](http://www.lrec-conf.org/proceedings/lrec2006/pdf/128_pdf.pdf) - describes a method to extract a German definition dataset  

[Weakly Supervised Definition Extraction](https://aclanthology.org/R15-1025.pdf) - presents a W00 dataset  

[Syntactically Aware Neural Architectures for Definition Extraction](https://orca.cardiff.ac.uk/id/eprint/111116/1/syntactically-aware-neural-2.pdf) - neural approach  

[Definition Extraction with LSTM Recurrent Neural Networks](https://link.springer.com/chapter/10.1007/978-3-319-47674-2_16) - we generate sentence feature using LSTM which is fit for both short-term and long-term structure learning. Our method consists of the following three steps: (See Fig. 1 for graphical structure.)

  - Token transformation: each word in a sentence will be transformed into a token according to its frequency in training set. (Sect. 3.1)

  - Word feature generation: each word will be represented as a word vector by capturing features of the word‘s context. (Sect. 3.2)

  - Sentence feature generation: A LSTM encoder will be used to automatically transform a sentence into a vector representation by learning sentence hidden structure feature. A classifier will also be trained to predict a sentence label by its feature. (Sect. 3.3)  

The reason we do not choose words’ tf-idf is that some words (e.g. is, a, etc.) are important to form a definitional pattern in spite of their low idf. They use word2vec (50 dimensions)

[SemEval-2020 Task 6: Definition extraction from free text with the DEFT
corpus](https://arxiv.org/pdf/2008.13694.pdf) 

[DeftEval corpus description](https://aclanthology.org/W19-4015.pdf)

[Best performing model for subtask 1 of the DEFT Eval competition](https://www.researchgate.net/publication/355429534_Gorynych_Transformer_at_SemEval-2020_Task_6_Multi-task_Learning_for_Definition_Extraction)

[Huge dataset with german words definitions](https://hal.archives-ouvertes.fr/hal-01798704/) - a specialized web corpus, that is a collection of web documents targeting web pages which are defined in advance; TODO: request access

https://hal.archives-ouvertes.fr/hal-01575661/document (in german)

# Metrices 
- Precision
- Recall
- F1

# Approaches
  - fine-tune a language model (at inference, first fine-tune the model on the large corpus available)
  - definition labeling

# Tasks
  - find a definition corpus
  - find "difficult" definitions: those whose term-definition pair span crosses a sentence boundary and those lacking explicit definition phrases.
  - The vector representation will be fed to two classifiers to determine whether the noun is hyponym, hypernym or neither of two. If a sentence has both hyponym and hypernym, it will be labeled as definitional.

# Keywords
  - definition extraction
  - information extraction
  - hypernym detection
  - Word-Class Lattices - use a directed acyclic graph to represent definitional sentences

# Data
  - WCL - a portion of the data was extracted by taking the first sentences of randomly sampled Wikipedia articles
  - ukWaC
  - a large crawled dataset of the .uk domain name
  - W00 - a manually annotated subset of ACL ARC
  - [DEFT corpus](https://github.com/adobe-research/deft_corpus)
  - [Wikipedia benchmark corpus](https://aclanthology.org/P10-1134.pdf)

# Things to check

Geyken, A., Barbaresi, A., Didakowski, J., Jurish, B., Wiegand, F., and Lemnitzer, L. (2017a). Die Kor- pusplattform des ”Digitalen Wo ̈rterbuchs der deutschen Sprache” (DWDS). Zeitschrift fu ̈r germanistische Lin- guistik, 45(2):327–344.

