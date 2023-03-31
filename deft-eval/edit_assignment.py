import spacy
import streamlit as st
import pandas as pd
import base64

# Set up the logger
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a StreamHandler to display logs in Streamlit


class StreamHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        logs = st.session_state.get("logs", [])
        logs.append(log_entry)
        st.session_state["logs"] = logs


stream_handler = StreamHandler()
logger.addHandler(stream_handler)


def colorize_text(sentence, tags_sequence_pred):
    """
    Given a sentence and its predicted tags, return the sentence with each word colored based on its tag.
    """
    words = sentence.split()
    tags = tags_sequence_pred.split()
    colored_words = []
    for i in range(len(words)):
        word = words[i]
        tag = tags[i]
        if tag == "O":
            colored_word = f'{word}'
            colored_words.append(colored_word)
        elif "Term" in tag:
            colored_word = f'<span style="color:red">{word}</span>'
            colored_words.append(colored_word)
        elif "Definition" in tag:
            colored_word = f'<span style="color:blue">{word}</span>'
            colored_words.append(colored_word)
    colored_sentence = " ".join(colored_words)
    return colored_sentence


# Load the Spacy English model
nlp = spacy.load("en_core_web_sm")

# Define a function to perform named entity recognition and mark the corresponding tags as "Term"


def get_word_indices(sentence, start, end):
    words = sentence.split()
    word_indices = []
    for i, word in enumerate(words):
        word_start = sentence.find(word)
        word_end = word_start + len(word)
        if start <= word_start and end >= word_end:
            word_indices.append(i)
    return word_indices


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


def prepend_tags(tags):
    tags = tags.split()
    updated_tags = []
    prev_tag = ""
    for tag in tags:
        if "Term" in tag:
            if "Term" not in prev_tag:
                updated_tags.append("B-Term")
            else:
                updated_tags.append("I-Term")
        elif "Definition" in tag:
            if "Definition" not in prev_tag:
                updated_tags.append("B-Definition")
            else:
                updated_tags.append("I-Definition")
        else:
            updated_tags.append(tag)
        prev_tag = tag
    return " ".join(updated_tags)


def main():
    st.set_page_config(page_title="Tag Editor", page_icon=":pencil2:")

    # Get the file from the user
    uploaded_file = st.file_uploader("Choose a TSV file", type=["tsv"])

    if uploaded_file is not None:
        # Read the file into a DataFrame
        df = pd.read_csv(uploaded_file, sep="\t")
        df = df.sort_values(
            by=["sent_type_pred", "sent_type_scores"], ascending=[False, True])

        # Apply the tagging function to each row of the DataFrame
        df["tags_sequence_pred"] = df.apply(lambda row: tag_named_entities(
            row["tokens"], row["tags_sequence_pred"], row["sent_type_pred"]), axis=1)

        df["tags_sequence_pred"] = df.apply(
            lambda row: prepend_tags(row["tags_sequence_pred"]), axis=1)

        # Colorize the sentences based on their tags
        df["colored_tokens"] = df.apply(lambda row: colorize_text(
            row["tokens"], row["tags_sequence_pred"]), axis=1)

        # Create a copy of the DataFrame to store the user's changes
        df_edited = df.copy()

        # Display the sentences with their tags
        st.write("## Sentences")
        for i in range(len(df)):
            sentence = df.iloc[i]["colored_tokens"]
            st.markdown(sentence, unsafe_allow_html=True)

            # Allow the user to edit the tags
            tags = df_edited.iloc[i]["tags_sequence_pred"].split()
            tags_input = st.empty()
            tags_input.text_input("Tags", " ".join(tags), key=f"tags-{i}")

            # Add a button to update the coloring based on the edited tags
            if st.button(f"Update colors for sentence {i+1}"):
                new_tags = tags_input.text_input(
                    "Tags", " ".join(tags), key=f"new-tags-{i}")
                df_edited.at[i, "tags_sequence_pred"] = new_tags
                if len(new_tags.split()) != len(df_edited.iloc[i]["tokens"].split()):
                    st.markdown("Should be same length",
                                unsafe_allow_html=True)
                else:
                    colored_sentence = colorize_text(
                        df_edited.iloc[i]["tokens"], new_tags)
                    st.markdown(colored_sentence, unsafe_allow_html=True)

        # Allow the user to download the edited file
        if st.button("Download edited file"):
            csv = df_edited.to_csv(index=False, sep="\t")
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="edited_file.tsv">Download edited file</a>'
            st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":

    main()
