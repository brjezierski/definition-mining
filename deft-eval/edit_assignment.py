import streamlit as st
import pandas as pd
import base64


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
            colored_word = word
            colored_words.append(colored_word)
        elif tag == "Term":
            colored_word = "<span style='color:red'>{}</span>".format(word)
            colored_words.append(colored_word)
        elif tag == "Definition":
            colored_word = "<span style='color:yellow'>{}</span>".format(word)
            colored_words.append(colored_word)
    colored_sentence = " ".join(colored_words)
    return colored_sentence


def main():
    st.set_page_config(page_title="Tag Editor", page_icon=":pencil2:")

    # Get the file from the user
    uploaded_file = st.file_uploader("Choose a TSV file", type=["tsv"])

    if uploaded_file is not None:
        # Read the file into a DataFrame
        df = pd.read_csv(uploaded_file, sep="\t")

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
            edited_tags = st.text_input("Tags", " ".join(tags))
            df_edited.at[i, "tags_sequence_pred"] = edited_tags

        # Allow the user to download the edited file
        if st.button("Download edited file"):
            csv = df_edited.to_csv(index=False, sep="\t")
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="edited_file.tsv">Download edited file</a>'
            st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
