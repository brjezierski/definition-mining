import argparse
import spacy
import pandas as pd

# Load the Spacy English model
nlp = spacy.load("en_core_web_sm")

# Define a function to perform named entity recognition and mark the corresponding tags as "Term"


def tag_named_entities(sentence, tags_sequence_pred, sent_type):
    # Use Spacy to perform named entity recognition
    if (sent_type == 1):
        doc = nlp(sentence)

        # Extract the named entities and their positions in the sentence
        named_entities = [(ent.text, ent.start_char, ent.end_char,
                           ent.label_) for ent in doc.ents]

        # Convert the tags sequence to a list
        tags = tags_sequence_pred.split()

        # Mark the corresponding tags as "Term" for each named entity
        for entity in named_entities:
            start = entity[1]
            end = entity[2]
            for i in get_word_indices(sentence, start, end):
                tags[i] = "Term"

        # Convert the list of tags back to a string and return it
        return " ".join(tags)
    else:
        return tags_sequence_pred


def get_word_indices(sentence, start, end):
    words = sentence.split()
    word_indices = []
    for i, word in enumerate(words):
        word_start = sentence.find(word)
        word_end = word_start + len(word)
        if start <= word_start and end >= word_end:
            word_indices.append(i)
    return word_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input_file, sep='\t')
    df["tags_sequence_pred"] = df.apply(lambda row: tag_named_entities(
        row["tokens"], row["tags_sequence_pred"], row["sent_type_pred"]), axis=1)
    df.to_csv(index=False, sep="\t")
