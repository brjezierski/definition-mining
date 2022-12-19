from collections import defaultdict, Counter
import os
from itertools import groupby
import string
# from flair.data import Sentence
# from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
import torch
import nltk
from nltk.tokenize import word_tokenize
# import tensorflow_hub as hub
import numpy as np
# import tensorflow_text
import pandas as pd
# import tensorflow as tf
# from flair.models import MultiTagger
# import flair
from tqdm import tqdm
from sentence_aligner import SentenceAligner
import spacy

class bcolors:
	Default      = "\033[39m"
	Black        = "\033[30m"
	Red          = "\033[31m"
	Green        = "\033[32m"
	Yellow       = "\033[33m"
	Blue         = "\033[34m"
	Magenta      = "\033[35m"
	Cyan         = "\033[36m"
	LightGray    = "\033[37m"
	DarkGray     = "\033[90m"
	LightRed     = "\033[91m"
	LightGreen   = "\033[92m"
	LightYellow  = "\033[93m"
	LightBlue    = "\033[94m"
	LightMagenta = "\033[95m"
	LightCyan    = "\033[96m"
	White        = "\033[97m"


tqdm.pandas()
nltk.download('punkt')
myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

spacy.require_gpu()
# os.system("spacy download de_dep_news_trf")

# nlp = load_spacy('de_dep_news_trf')  # Will download the model if it isn't installed yet
nlp = spacy.load('de_dep_news_trf')

EVAL_TAGS = [
    'B-Term', 'I-Term', 'B-Definition', 'I-Definition',
    'B-Alias-Term', 'I-Alias-Term', 'B-Referential-Definition', 'I-Referential-Definition',
    'B-Referential-Term', 'I-Referential-Term', 'B-Qualifier', 'I-Qualifier'
]


def cstr(s, color=bcolors.Default):
   return f"{color}{s}{color}"


def remove_order_tag(relation):
   return relation.replace('B-', '').replace('I-', '')


def display_tagged_sent(sent, tags):
  tagged_sent = []
  for word, tag in zip(sent, tags):
    tag = remove_order_tag(tag)
    if tag == "Definition":
      tagged_sent.append(cstr(word, color=bcolors.Blue))
    elif tag == "Secondary-Definition":
      tagged_sent.append(cstr(word, color=bcolors.LightBlue))
    elif tag == "Ordered-Definition":
      tagged_sent.append(cstr(word, color=bcolors.Cyan))
    elif tag == "Referential-Definition":
      tagged_sent.append(cstr(word, color=bcolors.LightCyan))
    elif tag == "Term":
      tagged_sent.append(cstr(word, color=bcolors.Red))
    elif tag == "Ordered-Term":
      tagged_sent.append(cstr(word, color=bcolors.LightRed))
    elif tag == "Referential-Term":
      tagged_sent.append(cstr(word, color=bcolors.LightMagenta))
    elif tag == "Qualifier":
      tagged_sent.append(cstr(word, color=bcolors.Green))
    elif tag == "O":
      tagged_sent.append(cstr(word, color=bcolors.Default))
    else:
      tagged_sent.append(cstr(word, color=bcolors.Default))
  return " ".join(tagged_sent)


def align_words(en_raw, de_raw, en_relations_sequence, label, ind, to_print=False):
#   print(en_relations_sequence)
  relation_tags = list(filter(lambda relation: relation.startswith("B-"), en_relations_sequence))
  relation_tags = [remove_order_tag(tag) for tag in relation_tags]
  en_relations_sequence = [remove_order_tag(seq) for seq in en_relations_sequence]

  de_sent = de_raw
  de_sent = word_tokenize(de_sent)
  de_raw = word_tokenize(de_raw)

  if label == 0:
    return ["O"] * len(de_raw)

  en_sent = en_raw
  en_sent = word_tokenize(en_sent)
  en_raw = word_tokenize(en_raw)

  alignments = myaligner.get_word_aligns(en_raw, de_raw, use_only_rev=True)
  
  docs = list(nlp.pipe([' '.join(de_raw)]))
  de_pos_tags = [t.pos_ for t in docs[0]]
  de_relations_sequence = ["O"] * len(de_raw)

  for en_ind, de_ind in alignments["inter"]:
    if en_ind < len(en_relations_sequence):
      tag = en_relations_sequence[en_ind]
      de_relations_sequence[de_ind] = tag
      if to_print:
        print(de_raw[de_ind],":", en_raw[en_ind], '-', tag)

  seen_relations = []
#   print('before',de_relations_sequence)
  for ind, (de_relation, de_pos_tag) in enumerate(zip(de_relations_sequence, de_pos_tags)):
    if (de_relation == "O") and (ind-1 >= 0) and (ind+1 < len(de_relations_sequence)) and de_relations_sequence[ind-1] == de_relations_sequence[ind+1]:
      de_relations_sequence[ind] = de_relations_sequence[ind-1]
    if (de_pos_tag == "DET") and (de_relation != "O") and (ind+1 < len(de_relations_sequence)) and de_relations_sequence[ind] != de_relations_sequence[ind+1]:
      de_relations_sequence[ind] = de_relations_sequence[ind+1]

  # Second iteration
  for ind, (de_relation, de_pos_tag) in enumerate(zip(de_relations_sequence, de_pos_tags)):
    # Label first non-O tag in the sentence with B-
    if (de_relation not in seen_relations and (de_relation != "O")) and (
            (ind == 0)
                or
            (ind-1 >= 0) and de_relations_sequence[ind] != remove_order_tag(de_relations_sequence[ind-1])
        ):
      if de_relation in relation_tags:
         relation_tags.remove(de_relation)
      seen_relations.append(de_relation)
      de_relations_sequence[ind] = f"B-{de_relations_sequence[ind]}"
    elif de_relation in seen_relations:
      de_relations_sequence[ind] = f"I-{de_relations_sequence[ind]}"
    elif de_relation != "O":
      print("ERROR ", de_relation)
      
  # Third iteration to add B- to double tags
  for ind, (de_relation, de_pos_tag) in enumerate(zip(de_relations_sequence, de_pos_tags)):
    if de_relation in [f"I-{tag}" for tag in relation_tags] and ind-1 >= 0 and de_relations_sequence[ind-1] == "O":
       de_relations_sequence[ind] = f"B-{remove_order_tag(de_relations_sequence[ind])}"
       relation_tags.remove(remove_order_tag(de_relation))

  if len(relation_tags) > 0:
      print('ERROR! Skipping a row. A tag left unassigned for the following sentence ')
      print(display_tagged_sent(en_raw, en_relations_sequence))
      print(display_tagged_sent(de_raw, de_relations_sequence))
      return np.nan

  if ind % 40 == 0:
    print(display_tagged_sent(en_raw, en_relations_sequence))
    print(display_tagged_sent(de_raw, de_relations_sequence))
    # print('after',de_relations_sequence)

  return de_relations_sequence


def read_task_2(
    data_dir: str,
    sep: str = '\t',
    columns: str = '+'.join(
        [
            'tokens', 'source', 'start_char',
            'end_char', 'tag', 'tag_id',
            'root_id', 'relation'
        ]
    ),
):
    columns = columns.split('+')

    result = defaultdict(list)
    sentences = defaultdict(list)
    num_columns = None

    for file in sorted(os.listdir(data_dir)):
      if file.startswith('task_2'):
        # print(file)
        with open(os.path.join(data_dir, file)) as fp:
            file_lines = [line.strip() for line in fp.readlines()]

        infile_offset = 0
        if num_columns is None:
            num_columns = len(file_lines[0].split(sep))

        for is_sep, lines in groupby(file_lines, key=lambda line: line == ''):
            lines = list(lines)
            if not is_sep:
                sentences[file].append((lines, [i + infile_offset for i in range(len(lines))]))
            infile_offset += len(lines)

    all_sentences = \
    [
        [
            [column.strip() for column in tokens.split(sep)] + [infile_offset]
            for tokens, infile_offset in zip(sentence, infile_offsets)
        ]
        for file_sentences in sentences.values()
        for (sentence, infile_offsets) in file_sentences
    ]
    print(f'Total of {len(all_sentences)} sentences in corpus')

    sentence_starts = \
    [
        i for i, sentence in enumerate(all_sentences)
        if len(sentence) == 2 and sentence[0][0].isdigit() and (sentence[1][0] == '.')
    ]
    print(f'Grouping into {len(all_sentences)-len(sentence_starts)} rows')

    window_sentences = []
    for i in range(len(all_sentences) - 1):
      sentence = all_sentences[i]
      if len(sentence) == 2 and sentence[0][0].isdigit() and (sentence[1][0] == '.'):
        window_sentence = []
        window_sentence.extend(sentence)
        saw_number = True
      elif saw_number:
        saw_number = False
        window_sentence.extend(sentence)
        window_sentences.append(window_sentence)
      else:
        window_sentence = []
        window_sentence.extend(sentence)
        window_sentences.append(window_sentence)

    columns = columns[:num_columns]

    def filter_value(value, eval_labels, neg_label):
        if value.strip() not in eval_labels and value.strip() != neg_label:
            value = neg_label
        return value

    for window_sentence in window_sentences:
        sentence = defaultdict(list)
        # print('sentence', window_sentence)
        for fields in window_sentence:
            for i, column_name in enumerate(columns):
                column_value = fields[i]
                # print('column_name', column_name)
                # print('field', fields[i])
                column_value = filter_value(
                    column_value, EVAL_TAGS, 'O'
                )
                sentence[column_name].append(fields[i])
            sentence['infile_offsets'].append(fields[-1])

        for column_name in sentence:
            result[column_name].append(sentence[column_name])

    return result


def load_de_translation(corpus, de_corpus_repo: str = '../data/de'):
   translation_corpus_dir=f'{de_corpus_repo}/{corpus}.tsv'
   dataset = pd.read_csv(translation_corpus_dir, on_bad_lines='skip', sep='\t')
   dataset['Text'] = dataset['Text'].str.strip()
   dataset['Translation'] = dataset['Translation'].str.strip()
   dataset = dataset.drop_duplicates(subset=['Text'])
   return dataset


def create_dataset(corpus, deft_corpus_repo: str = 'deft_corpus'):
    task_2_dir=f'../{deft_corpus_repo}/local_data/task_2/{corpus}'
    part = read_task_2(
            data_dir=task_2_dir
        )

    columns = [x for x in part.keys()]
    task_2_columns = [
                'tokens', 'source', 'start_char',
                'end_char', 'tag', 'infile_offsets',
                'part'
            ]
    for column in columns:
        if column not in task_2_columns:
            del part[column]

    dataset_en = pd.DataFrame(part)
    dataset_en['Text'] = dataset_en['tokens'].apply(' '.join)
    dataset_en = dataset_en.drop_duplicates(subset=['Text'])
    dataset_de = load_de_translation(corpus)
    print(f'Processing {len(dataset_de)} rows of German translations')

    bilingual_dataset = pd.merge(dataset_de, dataset_en, left_on='Text', right_on='Text', how='left')
    bilingual_dataset.dropna(inplace=True)
    bilingual_dataset = bilingual_dataset.rename(columns={"Text": "text_en", "Translation": "text_de", "tokens": "tokens_en", "tag": "tags_en", 'Label': "sent_type"}, errors="raise")
    # bilingual_dataset["tags_de"] = ""

    bilingual_dataset["tags_de"] = bilingual_dataset[:1000].progress_apply(lambda row: align_words(row['text_en'], row['text_de'], row['tags_en'], row['sent_type'], row.name), axis=1)
    bilingual_dataset["tokens_de"] = bilingual_dataset.apply(lambda row: word_tokenize(row['text_de']), axis=1)
    bilingual_dataset.dropna(inplace=True)
    bilingual_dataset = bilingual_dataset[['sent_type', 'text_en', 'tags_en', 'tokens_en', 'text_de', 'tags_de', 'tokens_de', 'source', 'start_char', 'end_char', 'infile_offsets']]
    bilingual_dataset.to_json(f'{target_dir}/{corpus}.json', orient='records')

    print(f'Combining into {len(bilingual_dataset)} rows of the bilingual dataset')
    return bilingual_dataset


   
if __name__ == '__main__':
    target_dir = 'data/bilingual-small'
    print('Creating a train dataset...')
    train_df = create_dataset('train')
    print('\nCreating a dev dataset...')
    dev_df = create_dataset('dev')
    print('\nCreating a test dataset...')
    test_df = create_dataset('test')