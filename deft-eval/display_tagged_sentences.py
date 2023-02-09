from create_dataset import display_tagged_sent, display_tag_color_legend
import argparse
import os
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--substr", type=str)
    parser.add_argument("--show_only_def", action='store_true', default=False)
    parser.add_argument("--narrow_search", action='store_true', default=False)
    args = parser.parse_args()
    dataset = pd.read_csv(args.input_file, sep='\t')
    corpus_len = len(dataset)
    if args.show_only_def:
        dataset = dataset[dataset['sent_type_pred'] == 1]
    if args.substr:
        dataset = dataset[dataset['tokens'].str.contains(args.substr) == True]
    dataset['tokens'] = dataset['tokens'].apply(lambda x: x.split())
    dataset['tags_sequence_pred'] = dataset['tags_sequence_pred'].apply(
        lambda x: x.split())

    print(f"{display_tag_color_legend()}\n")
    sent_count = 0
    for index, row in dataset.iterrows():
        if args.show_only_def:
            if args.narrow_search:
                if ("I-Term" in row['tags_sequence_pred'] or "B-Term" in row['tags_sequence_pred']) and ("B-Definition" in row['tags_sequence_pred'] or "I-Definition" in row['tags_sequence_pred']):
                    print(display_tagged_sent(
                        row['tokens'], row['tags_sequence_pred']))
                    sent_count += 1
            else:
                print(display_tagged_sent(
                    row['tokens'], row['tags_sequence_pred']))
                sent_count += 1
        else:
            print(display_tagged_sent(
                row['tokens'], row['tags_sequence_pred']))
            sent_count += 1
    print(
        f"\n{sent_count} sentences found in the corpus of {corpus_len} sentences.\n")
