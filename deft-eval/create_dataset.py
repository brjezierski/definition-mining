import argparse
from collections import defaultdict, Counter
import os
from itertools import groupby
import string
import torch
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_aligner import SentenceAligner
import spacy


class bcolors:
    Default = "\033[39m"
    Black = "\033[30m"
    Red = "\033[31m"
    Green = "\033[32m"
    Yellow = "\033[33m"
    Blue = "\033[34m"
    Magenta = "\033[35m"
    Cyan = "\033[36m"
    LightGray = "\033[37m"
    DarkGray = "\033[90m"
    LightRed = "\033[91m"
    LightGreen = "\033[92m"
    LightYellow = "\033[93m"
    LightBlue = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan = "\033[96m"
    White = "\033[97m"


tqdm.pandas()

EVAL_TAGS = [
    'B-Term', 'I-Term', 'B-Definition', 'I-Definition',
    'B-Alias-Term', 'I-Alias-Term', 'B-Referential-Definition', 'I-Referential-Definition',
    'B-Referential-Term', 'I-Referential-Term', 'B-Qualifier', 'I-Qualifier'
]

tag_ids = {'O': '-1', 'B-Definition': 'T127', 'I-Definition': 'T127', 'B-Term': 'T128', 'I-Term': 'T128', 'B-Alias-Term': 'T137', 'I-Alias-Term': 'T137', 'B-Secondary-Definition': 'T138', 'I-Secondary-Definition': 'T138', 'B-Ordered-Term': 'T144', 'I-Ordered-Term': 'T144', 'B-Ordered-Definition': 'T145', 'I-Ordered-Definition': 'T145',
           'B-Referential-Definition': 'T153', 'I-Referential-Definition': 'T194', 'B-Qualifier': 'T35', 'I-Qualifier': 'T35', 'B-Referential-Term': 'T124', 'B-Definition-frag': 'T246-frag', 'I-Definition-frag': 'T246-frag', 'I-Referential-Term': 'T37', 'B-Term-frag': 'T298-frag', 'B-Alias-Term-frag': 'T219-frag', 'I-Term-frag': 'T61-frag'}


def cstr(s, color=bcolors.Default):
    return f"{color}{s}{color}"


def remove_order_tag(relation):
    return relation.replace('B-', '').replace('I-', '')


def display_tag_color_legend():
    tagged_sent = ["Color coding:"]
    tagged_sent.append(cstr("\tDefinition", color=bcolors.Blue))
    tagged_sent.append(cstr("\tSecondary-Definition", color=bcolors.LightBlue))
    tagged_sent.append(cstr("\tOrdered-Definition", color=bcolors.Cyan))
    tagged_sent.append(
        cstr("\tReferential-Definition", color=bcolors.LightCyan))
    tagged_sent.append(cstr("\tTerm", color=bcolors.Red))
    tagged_sent.append(cstr("\tOrdered-Term", color=bcolors.LightRed))
    tagged_sent.append(cstr("\tReferential-Term", color=bcolors.LightMagenta))
    tagged_sent.append(cstr("\tQualifier", color=bcolors.Green))
    tagged_sent.append(cstr("\tO", color=bcolors.Default))
    return "\n".join(tagged_sent)


def display_tagged_sent(sent, tags):
    tagged_sent = []
    for word, tag in zip(sent, tags):
        tag = remove_order_tag(tag)
        if tag == "Definition":
            tagged_sent.append(cstr(word, color=bcolors.Yellow))
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


def get_tag_id_dict(df):
    tag_dict = {}
    for index, row in df.iterrows():
        for tag, id in zip(row['tags_sequence'], row['tags_ids']):
            if tag not in tag_dict:
                tag_dict[tag] = id
    return get_tag_id_dict


def align_words(en_raw, de_raw, en_relations_sequence, label, ind, to_print=False):
    relation_tags = list(
        filter(lambda relation: relation.startswith("B-"), en_relations_sequence))
    relation_tags = [remove_order_tag(tag) for tag in relation_tags]
    en_relations_sequence = [remove_order_tag(
        seq) for seq in en_relations_sequence]

    de_sent = de_raw
    de_sent = word_tokenize(de_sent)
    de_raw = word_tokenize(de_raw)

    if label == "0":
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
                print(de_raw[de_ind], ":", en_raw[en_ind], '-', tag)

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
            (ind-1 >= 0) and de_relations_sequence[ind] != remove_order_tag(
                de_relations_sequence[ind-1])
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

    return de_relations_sequence


def assign_zero_relations(tags):
    return ["0"] * len(tags)


def assign_tag_ids(tags):
    assigned_tag_ids = []
    # Forth iteration to add tag ids
    for tag in tags:
        if tag in tag_ids:
            assigned_tag_ids.append(tag_ids[tag])
        else:
            print(f'ERROR! Skipping tag {tag} does not have an id assigned')
            return np.nan
    return assigned_tag_ids


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
            with open(os.path.join(data_dir, file)) as fp:
                file_lines = [line.strip() for line in fp.readlines()]

            infile_offset = 0
            if num_columns is None:
                num_columns = len(file_lines[0].split(sep))

            for is_sep, lines in groupby(file_lines, key=lambda line: line == ''):
                lines = list(lines)
                if not is_sep:
                    sentences[file].append(
                        (lines, [i + infile_offset for i in range(len(lines))]))
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
    window_sentence = []
    saw_number = False

    if args.sent_aggregation in ['window', 'both']:
        for i in range(len(all_sentences) - 1):
            sentence = all_sentences[i]
            if len(sentence) == 2 and sentence[0][0].isdigit() and (sentence[1][0] == '.'):
                if len(window_sentence) > 0:
                    window_sentences.append(window_sentence)
                window_sentence = []
                window_sentence.extend(sentence)
                saw_number = True
            elif saw_number:
                window_sentence.extend(sentence)
            else:
                window_sentence = []
                window_sentence.extend(sentence)
                window_sentences.append(window_sentence)
    window_sentences = []
    window_sentence = []
    if args.sent_aggregation in ['single', 'both']:
        for i in range(len(all_sentences) - 1):
            sentence = all_sentences[i]
            if len(sentence) == 2 and sentence[0][0].isdigit() and (sentence[1][0] == '.'):
                window_sentence = []
                window_sentence.extend(sentence)
                continue
            if window_sentence != []:
                window_sentence.extend(sentence)
                window_sentences.append(window_sentence)
                window_sentence = []
            else:
                window_sentences.append(sentence)

    columns = columns[:num_columns]

    def filter_value(value, eval_labels, neg_label):
        if value.strip() not in eval_labels and value.strip() != neg_label:
            value = neg_label
        return value

    for window_sentence in window_sentences:
        sentence = defaultdict(list)
        for fields in window_sentence:
            for i, column_name in enumerate(columns):
                column_value = fields[i]
                column_value = filter_value(
                    column_value, EVAL_TAGS, 'O'
                )
                sentence[column_name].append(fields[i])
            sentence['infile_offsets'].append(fields[-1])

        for column_name in sentence:
            result[column_name].append(sentence[column_name])
        sent_type = "0"

        for tag in sentence["tag"]:
            if "Definition" in tag:
                sent_type = "1"
                break
        result["sent_type"].append(sent_type)

    return result


def load_de_translation(corpus, create_dataset):
    translation_corpus_dir = f'{translation_dir}/{corpus}.tsv'
    dataset = pd.read_csv(translation_corpus_dir,
                          on_bad_lines='skip', sep='\t')

    dataset['Text'] = dataset['Text'].str.strip()
    dataset['Translation'] = dataset['Translation'].str.strip()
    dataset = dataset.drop_duplicates(subset=['Text'])
    return dataset


def create_dataset(corpus, target_dir, translation_dir, deft_corpus_repo: str = 'deft_corpus', lang="bilingual"):
    task_2_dir = f'../{deft_corpus_repo}/local_data/task_2/{corpus}'
    part = read_task_2(
        data_dir=task_2_dir
    )

    columns = [x for x in part.keys()]
    task_2_columns = [
        'tokens', 'source', 'start_char',
        'end_char', 'tag', 'infile_offsets',
        'part', 'sent_type'
    ]
    for column in columns:
        if column not in task_2_columns:
            del part[column]

    dataset_en = pd.DataFrame(part)
    if lang == "en":
        dataset_en = dataset_en.rename(
            columns={"tag": "tags_sequence"}, errors="raise")
        dataset_en["tags_ids"] = dataset_en.apply(
            lambda row: assign_tag_ids(row['tags_sequence']), axis=1)
        dataset_en["relations_sequence"] = dataset_en.apply(
            lambda row: assign_zero_relations(row['tags_sequence']), axis=1)
        dataset_en = dataset_en.drop_duplicates(subset=['tokens'])
        dataset_en['text'] = dataset_en['tokens'].apply(' '.join)
        dataset_en = dataset_en.drop(
            ['source', 'start_char', 'end_char', 'infile_offsets'], axis=1)
        return dataset_en
    elif lang == "de" or lang == "bilingual":
        dataset_en.drop(columns=['sent_type'], inplace=True, axis=1)
        dataset_en['Text'] = dataset_en['tokens'].apply(' '.join)
        dataset_en = dataset_en.drop_duplicates(subset=['Text'])
        dataset_de = load_de_translation(corpus, translation_dir)
        print(f'Processing {len(dataset_de)} rows of German translations')

        bilingual_dataset = pd.merge(
            dataset_de, dataset_en, left_on='Text', right_on='Text', how='left')
        bilingual_dataset.dropna(inplace=True)
        bilingual_dataset = bilingual_dataset.rename(
            columns={"Text": "text_en", "Translation": "text_de", "tokens": "tokens_en", "tag": "tags_sequence_en", 'Label': "sent_type"}, errors="raise")
        # convert sent_type to string
        bilingual_dataset['sent_type'] = bilingual_dataset['sent_type'].astype(
            str)
        # bilingual_dataset = bilingual_dataset[:(10 if (len(bilingual_dataset) > 100) else len(bilingual_dataset))]

        bilingual_dataset["tags_sequence_de"] = bilingual_dataset.progress_apply(lambda row: align_words(
            row['text_en'], row['text_de'], row['tags_sequence_en'], row['sent_type'], row.name), axis=1)
        bilingual_dataset.dropna(inplace=True)
        bilingual_dataset["tags_ids_de"] = bilingual_dataset.apply(
            lambda row: assign_tag_ids(row['tags_sequence_de']), axis=1)
        bilingual_dataset["relations_sequence_de"] = bilingual_dataset.apply(
            lambda row: assign_zero_relations(row['tags_sequence_de']), axis=1)
        bilingual_dataset["tokens_de"] = bilingual_dataset.apply(
            lambda row: word_tokenize(row['text_de']), axis=1)
        bilingual_dataset.dropna(inplace=True)
        bilingual_dataset = bilingual_dataset[['sent_type', 'text_en', 'tags_sequence_en', 'tokens_en', 'text_de', 'tokens_de',
                                               'tags_sequence_de', 'tags_ids_de', 'relations_sequence_de', 'source', 'start_char', 'end_char', 'infile_offsets']]
        bilingual_dataset.to_json(
            f'{target_dir}/{corpus}.json', orient='records')
        print(
            f'Combining into {len(bilingual_dataset)} rows of the bilingual dataset')
        if lang == "de":
            dataset_de = bilingual_dataset.drop(
                ['text_en', 'tags_sequence_en', 'tokens_en', 'source', 'start_char', 'end_char', 'infile_offsets'], axis=1)
            dataset_de = dataset_de.rename({'text_de': 'text', 'tokens_de': 'tokens', 'tags_sequence_de': 'tags_sequence',
                                           'tags_ids_de': 'tags_ids', 'relations_sequence_de': 'relations_sequence'}, axis=1)
            return dataset_de
        else:
            bilingual_dataset = bilingual_dataset.drop(
                ['source', 'start_char', 'end_char', 'infile_offsets'], axis=1)
            return bilingual_dataset
    else:
        print(f"ERROR! Pick lang as de, en or bilingual, not {lang}")
        return np.nan


def assign_zero_tags(sentence):
    tokens = word_tokenize(sentence)
    tags_sequence = ["O"] * len(tokens)
    return tags_sequence


def assign_one_sent_types(sentence):
    sent_type = "1"
    return sent_type


def convert_definitionen(df):
    df['tokens'] = df['terminus'].str.strip()
    df = df.dropna()
    df = df.drop_duplicates(subset=['tokens'])
    df = df.drop(['key', 'terminus', 'i'], axis=1)

    df["tags_sequence"] = df.progress_apply(
        lambda row: assign_zero_tags(row['tokens']), axis=1)
    df["sent_type"] = df.progress_apply(
        lambda row: assign_one_sent_types(row['tokens']), axis=1)
    df["relations_sequence"] = df.progress_apply(
        lambda row: assign_zero_relations(row['tags_sequence']), axis=1)

    df["tags_ids"] = df.apply(
        lambda row: assign_tag_ids(row['tags_sequence']), axis=1)

    return df


def get(df):
    tag_dict = {}
    for index, row in df.iterrows():
        for tag, id in zip(row['tags_sequence'], row['tags_ids']):
            if tag not in tag_dict:
                tag_dict[tag] = id


if __name__ == '__main__':
    nltk.download('punkt')
    myaligner = SentenceAligner(
        model="bert", token_type="bpe", matching_methods="mai")
    spacy.require_gpu()
    # os.system("spacy download de_dep_news_trf")
    nlp = spacy.load('de_dep_news_trf')

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='bilingual',
                        type=str, required=False)
    parser.add_argument("--target_dir", type=str, required=False)
    parser.add_argument("--sent_aggregation", default='single',
                        type=str, required=False)  # single or window or both
    parser.add_argument("--translation_dir", default='../data/de',
                        type=str, required=False)
    args = parser.parse_args()
    if args.lang in ["de", "bilingual"] and args.sent_aggregation in ["window", "both"]:
        print("ERROR! Right now only single sentence aggregation is supported for German")
        exit()
    target_dir = args.target_dir if args.target_dir else f'data/{args.lang}'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print('Creating a train dataset...')
    train_df = create_dataset(
        'train', args.translation_dir, target_dir, lang=args.lang)
    train_df.to_json(f'{target_dir}/train.json', orient='records')
    print('\nCreating a dev dataset...')
    dev_df = create_dataset('dev', args.translation_dir,
                            target_dir, lang=args.lang)
    dev_df.to_json(f'{target_dir}/dev.json', orient='records')
    print('\nCreating a test dataset...')
    test_df = create_dataset(
        'test', args.translation_dir, target_dir, lang=args.lang)
    test_df.to_json(f'{target_dir}/test.json', orient='records')
