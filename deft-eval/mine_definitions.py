import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from tqdm import tqdm
from models.examples_to_features import (
    get_dataloader_and_tensors,
    models, tokenizers, DataProcessor
)
from collections import defaultdict

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
eval_logger = logging.getLogger("__scores__")


def evaluate(
        model, device, eval_dataloader,
        skip_every_n_examples=1
    ):
    model.eval()

    nb_eval_steps = 0
    preds = defaultdict(list)

    for batch_id, batch in enumerate(tqdm(
            eval_dataloader, total=len(eval_dataloader),
            desc='validation ... '
        )):

        if skip_every_n_examples != 1 and (batch_id + 1) % skip_every_n_examples != 1:
            continue

        batch = tuple([elem.to(device) for elem in batch])

        input_ids, input_mask, segment_ids, \
            sent_type_labels_ids, tags_sequence_labels_ids, \
            relations_sequence_labels_ids, token_valid_pos_ids = batch

        with torch.no_grad():
            outputs, _ = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                sent_type_labels=sent_type_labels_ids,
                tags_sequence_labels=tags_sequence_labels_ids,
                relations_sequence_labels=relations_sequence_labels_ids,
                token_valid_pos_ids=token_valid_pos_ids,
                device=device
            )

        sent_type_logits, tags_sequence_logits = outputs[:2]

        nb_eval_steps += 1
        preds['sent_type'].append(
            sent_type_logits.detach().cpu().numpy()
        )
        preds['tags_sequence'].append(
            tags_sequence_logits.detach().cpu().numpy()
        )

    preds['sent_type'] = np.concatenate(preds['sent_type'], axis=0)
    preds['tags_sequence'] = np.concatenate(preds['tags_sequence'], axis=0)

    scores = {}

    for key in preds:
        scores[key] = softmax(preds[key], axis=-1).max(axis=-1)
        preds[key] = preds[key].argmax(axis=-1)

    return preds, scores


def main(args):
    if (args.language == "de"):
        model_name = "deepset/gbert-large"
        model_dir = "training/de-gbert"
    else:
    # elif (args.language == "en"):
        model_name = "bert-large-uncased"
        model_dir = "training/en"

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else ("mps" if torch.has_mps else "cpu"))
    n_gpu = 1 if torch.has_mps else torch.cuda.device_count()

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

    processor = DataProcessor(
        filter_task_1=False,
        filter_task_3=True
    )

    sent_type_labels_list = ['0', '1']
    tags_sequence_labels_list = processor.get_hardcoded_sequence_labels(args.language, sequence_type='tags_sequence')
    relations_sequence_labels_list = processor.get_hardcoded_sequence_labels(args.language, sequence_type='relations_sequence')

    label2id = {
        'sent_type': {
            label: i for i, label in enumerate(sent_type_labels_list)
        },
        'tags_sequence': {
            label: i for i, label in enumerate(tags_sequence_labels_list, 1)
        },
        'relations_sequence': {
            label: i for i, label in enumerate(relations_sequence_labels_list, 1)
        }
     }

    id2label = {
        'sent_type': {
            i: label for i, label in enumerate(sent_type_labels_list)
        },
        'tags_sequence': {
            i: label for i, label in enumerate(tags_sequence_labels_list, 1)
        },
        'relations_sequence': {
            i: label for i, label in enumerate(relations_sequence_labels_list, 1)
        }
    }

    num_sent_type_labels = 2
    num_tags_sequence_labels = len(tags_sequence_labels_list) + 1
    num_relations_sequence_labels = len(relations_sequence_labels_list) + 1

    do_lower_case = 'uncased' in model_name
    tokenizer = tokenizers[model_name].from_pretrained(
        model_name, do_lower_case=do_lower_case
    )
    
    model = models[model_name].from_pretrained(
            model_dir,
            num_sent_type_labels=num_sent_type_labels,
            num_tags_sequence_labels=num_tags_sequence_labels,
            num_relations_sequence_labels=num_relations_sequence_labels,
            pooling_type=args.subtokens_pooling_type
        )

    model.to(device)
    
    test_examples = processor.get_test_examples(args.input_file, args.text_column)
    test_features, test_new_examples = model.convert_examples_to_features(
                test_examples, label2id, args.max_seq_length,
                tokenizer, logger, args.sequence_mode, context_mode=args.context_mode
            )

    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    test_dataloader, _, _, _ = get_dataloader_and_tensors(test_features, args.eval_batch_size)

    preds, scores = evaluate(model, device, test_dataloader)
    dest_file = 'labeled_' + args.input_file.split('/')[-1].replace('.json', '')
    write_predictions(
            model_dir, test_new_examples, test_features,
            preds, scores, dest_file,
            label2id=label2id, id2label=id2label
        )



def write_predictions(
    model_dir, examples, features, preds,
    scores, dest_file, label2id, id2label, metrics=None
):
    aggregated_results = {}
    orig_positions_map = [ex.orig_positions_map for ex in features]
    neg_label_mapper = {
        'tags_sequence': 'O',
    }
    for task in ['tags_sequence']:
        aggregated_results[task] = [
            list(pred[orig_positions]) + \
            [
                label2id[task][neg_label_mapper[task]]
            ] * (len(ex.tokens) - len(orig_positions))
            for pred, orig_positions, ex in zip(
                preds[task],
                orig_positions_map,
                examples
            )
        ]

        aggregated_results[f'{task}_scores'] = [
            list(score[orig_positions]) + \
            [0.999] * (len(ex.tokens) - len(orig_positions))
            for score, orig_positions, ex in zip(
                scores[task],
                orig_positions_map,
                examples
            )
        ]

    prediction_results = {
        'tokens': [
             ' '.join(ex.tokens) for ex in examples
        ],
        'sent_type_pred': [
            id2label['sent_type'][x] for x in preds['sent_type']
        ],
        'sent_type_scores': [
            str(score) for score in scores['sent_type']
        ],
        'tags_sequence_pred': [
            ' '.join([id2label['tags_sequence'][x] if x != 0 else 'O' for x in sent])
            for sent in aggregated_results['tags_sequence']
        ],
        'tags_sequence_scores': [
            ' '.join([str(score) for score in sent])
            for sent in aggregated_results['tags_sequence_scores']
        ],
        'tags_ids': [
            ' '.join(ex.tags_ids) for ex in examples
        ]
    }

    prediction_results = pd.DataFrame(prediction_results)

    print(f"Saving {model_dir}/{dest_file}.tsv")

    prediction_results.to_csv(
        os.path.join(
            model_dir,
            f"{dest_file}.tsv"),
        sep='\t', index=False
    )

    if metrics is not None:
        with open(
            os.path.join(
                model_dir,
                f"{dest_file}_eval_results.txt"
            ), "w"
        ) as f:
            for key in sorted(metrics.keys()):
                f.write("%s = %s\n" % (key, str(metrics[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default='', type=str, required=False)
    parser.add_argument("--language", default='en', type=str, required=True)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--label_tags", action="store_true",
                        help="label words with tags in the input data")
    parser.add_argument("--subtokens_pooling_type", type=str, default="first",
                        help="pooling mode in bert-ner, one of avg or first")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.\n"
                             "Sequences longer than this will be truncated, and sequences shorter\n"
                             "than this will be padded.")
    parser.add_argument("--sequence_mode", type=str, default="not-all",
                        help="train to predict for all subtokens or not"
                        "all or not-all")
    parser.add_argument("--context_mode", type=str, default="full",
                        help="context for task 1: one from center, full, left, right")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--text_column", default="text", type=str,
                        help="The title of the column with text.")

    parsed_args = parser.parse_args()
    main(parsed_args)
