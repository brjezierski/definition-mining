from transformers import BertConfig, BertTokenizer
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, TensorDataset
from .multitask_gbert import GBertForMultitaskLearning
from .multitask_bert import BertForMultitaskLearning
from transformers import XLNetConfig
import torch
import os
import json
from collections import Counter
from nltk.tokenize import word_tokenize


class InputExample(object):

    def __init__(
        self,
        tokens: list,
        sent_type: str,
        tags_sequence: list,
        tags_ids: list,
    ):
        self.tokens = tokens
        self.sent_type = sent_type
        self.tags_sequence = tags_sequence
        self.tags_ids = tags_ids


class DataProcessor(object):
    """Processor for the DEFTEVAL data set."""

    def __init__(
        self,
        filter_task_1: bool = False,
        filter_task_3: bool = False
    ):
        self.filter_task_1 = filter_task_1
        self.filter_task_3 = filter_task_3

    def _read_json(self, input_file):
        with open(input_file, "r", encoding='utf-8') as reader:
            data = json.load(reader)
            for (prefix, do_filter) in [
                ('sent', self.filter_task_1), ('subj', self.filter_task_3)
            ]:
                if do_filter:
                    new_data = []
                    for example in data:
                        new_data.append(example)
                    data = new_data

        return data

    def _read_test_json(self, input_file, text_column_title='text'):
        with open(input_file, "r", encoding='utf-8') as reader:
            data = json.load(reader)
            for (prefix, do_filter) in [
                ('sent', self.filter_task_1), ('subj', self.filter_task_3)
            ]:
                if do_filter:
                    new_data = []
                    for example in data:
                        example['tokens'] = word_tokenize(
                            example[text_column_title])
                        token_count = len(example['tokens'])
                        if 'tags_ids' not in example.keys():
                            example['tags_ids'] = ["-1"] * token_count
                        if 'tags_sequence' not in example.keys():
                            example['tags_sequence'] = ["O"] * token_count
                        if 'sent_type' not in example.keys():
                            example['sent_type'] = '0'
                        new_data.append(example)
                    data = new_data

        return data

    def get_train_examples(self, data_dir):
        return self.create_examples(
            self._read_json(
                os.path.join(data_dir, f"train.json")
            ),
            "train"
        )

    def get_dev_examples(self, data_dir):
        return self.create_examples(
            self._read_json(
                os.path.join(data_dir, f"dev.json")
            ),
            "dev"
        )

    def get_test_examples(self, test_file, text_column_title):
        return self.create_examples(
            self._read_test_json(test_file, text_column_title),
            "test"
        )

    def get_sent_type_labels(self, data_dir, logger=None):
        dataset = self._read_json(os.path.join(data_dir, f"train.json"))
        counter = Counter()
        labels = []
        for example in dataset:
            counter[example['sent_type']] += 1
        if logger is not None:
            logger.info(f"sent_type: {len(counter)} labels")
        for label, counter in counter.most_common():
            if logger is not None:
                logger.info("%s: %.2f%%" %
                            (label, counter * 100.0 / len(dataset)))
            if label not in labels:
                labels.append(label)
        return labels

    def get_hardcoded_sequence_labels(
        self,
        language,
        sequence_type: str = 'tags_sequence'
    ):
        if sequence_type == 'tags_sequence':
            return ['O', 'B-Definition', 'I-Definition', 'B-Term', 'I-Term', 'B-Alias-Term', 'I-Alias-Term', 'B-Secondary-Definition', 'I-Secondary-Definition', 'B-Ordered-Term', 'I-Ordered-Term', 'B-Ordered-Definition', 'I-Ordered-Definition', 'B-Referential-Definition', 'I-Referential-Definition', 'B-Qualifier', 'I-Qualifier', 'B-Referential-Term', 'B-Definition-frag', 'I-Definition-frag', 'I-Referential-Term', 'B-Term-frag', 'B-Alias-Term-frag', 'I-Term-frag']
        else:
            if language == 'en':
                return ['0'] * 7
            else:
                return ['0'] * 25

    def get_sequence_labels(
        self,
        data_dir: str,
        sequence_type: str = 'tags_sequence',
        logger=None
    ):
        tags_sequence_labels_list = ['O', 'B-Definition', 'I-Definition', 'B-Term', 'I-Term', 'B-Alias-Term', 'I-Alias-Term', 'B-Secondary-Definition', 'I-Secondary-Definition', 'B-Ordered-Term', 'I-Ordered-Term', 'B-Ordered-Definition',
                                     'I-Ordered-Definition', 'B-Referential-Definition', 'I-Referential-Definition', 'B-Qualifier', 'I-Qualifier', 'B-Referential-Term', 'B-Definition-frag', 'I-Definition-frag', 'I-Referential-Term', 'B-Term-frag', 'B-Alias-Term-frag', 'I-Term-frag']
        dataset = self._read_json(
            os.path.join(data_dir, "train.json")
        )
        denominator = len([
            lab for example in dataset for lab in example[sequence_type]
        ])
        counter = Counter()
        labels = []
        for example in dataset:
            for lab in example[sequence_type]:
                counter[lab] += 1
        if logger is not None:
            logger.info(f"{sequence_type}: {len(counter)} labels")

        for label, counter in counter.most_common():
            if logger is not None:
                logger.info("%s: %.2f%%" %
                            (label, counter * 100.0 / denominator))
            if label not in labels:
                labels.append(label)
        for tag in tags_sequence_labels_list:
            if tag not in labels:
                labels.append(tag)
        return labels

    def create_examples(self, dataset, set_type):
        examples = []
        for example in dataset:
            examples.append(
                InputExample(
                    tokens=example["tokens"],
                    sent_type=example["sent_type"],
                    tags_sequence=example["tags_sequence"],
                    tags_ids=example["tags_ids"],
                )
            )
        return examples


def get_dataloader_and_tensors(
        features: list,
        batch_size: int
):
    input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long
    )
    input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long
    )
    segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long
    )
    sent_type_labels_ids = torch.tensor(
        [f.sent_type_id for f in features],
        dtype=torch.long
    )
    tags_sequence_labels_ids = torch.tensor(
        [f.tags_sequence_ids for f in features],
        dtype=torch.long
    )
    token_valid_pos_ids = torch.tensor(
        [f.token_valid_pos_ids for f in features],
        dtype=torch.long
    )
    eval_data = TensorDataset(
        input_ids, input_mask, segment_ids,
        sent_type_labels_ids, tags_sequence_labels_ids, token_valid_pos_ids
    )

    dataloader = DataLoader(eval_data, batch_size=batch_size)

    return dataloader, sent_type_labels_ids, tags_sequence_labels_ids


tokenizers = {
    "bert-large-uncased": BertTokenizer,
    "deepset/gbert-large": AutoTokenizer.from_pretrained("deepset/gbert-large", use_fast=False)
}

models = {
    "bert-large-uncased": BertForMultitaskLearning,
    "deepset/gbert-large": GBertForMultitaskLearning
}

configs = {
    "bert-large-uncased": BertConfig,
    "deepset/gbert-large": AutoConfig.from_pretrained("deepset/gbert-large"),
}
