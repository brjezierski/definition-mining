import argparse
import torch

from models.examples_to_features import (
    DataProcessor
)
from models.multitask_gbert import GBertForMultitaskLearning
from models.multitask_bert import BertForMultitaskLearning
max_seq_length = 256

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu"))
    n_gpu = 1 if torch.has_mps else torch.cuda.device_count()

    if n_gpu > 0:
        torch.cuda.manual_seed_all(43)
        print("device: {}, n_gpu: {}".format(device, n_gpu))

    processor = DataProcessor(
        filter_task_1=False,
        filter_task_3=True
    )

    tags_sequence_labels_list = processor.get_hardcoded_sequence_labels(args.language, sequence_type='tags_sequence')

    num_sent_type_labels = 2
    num_tags_sequence_labels = len(tags_sequence_labels_list) + 1

    if args.language == "de":
        model = GBertForMultitaskLearning.from_pretrained(
            args.model_dir,
            num_sent_type_labels=num_sent_type_labels,
            num_tags_sequence_labels=num_tags_sequence_labels,
            pooling_type="first",
            use_auth_token=True)
    elif args.language == "en":
        model = BertForMultitaskLearning.from_pretrained(
            args.model_dir,
            num_sent_type_labels=num_sent_type_labels,
            num_tags_sequence_labels=num_tags_sequence_labels,
            pooling_type="first",
            use_auth_token=True)
    else:
        print("Language not supported")
        return
    model.to(device)
    model.push_to_hub(f"{args.username}/{args.model_name}", commit_message=args.commit_msg)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default='en', type=str, required=True)
    parser.add_argument("--username", default='', type=str, required=True)
    parser.add_argument("--model_name", default='', type=str, required=True)
    parser.add_argument("--model_dir", default='', type=str, required=True)
    parser.add_argument("--commit_msg", default='', type=str, required=True)
    args = parser.parse_args()
    main(args)