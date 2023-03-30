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
            colored_word = f'<button onclick="set_tag(this, \'Term\')">{word}</button>'
            colored_words.append(colored_word)
        elif "Term" in tag:
            colored_word = f'<button onclick="set_tag(this, \'Definition\')"><span style="color:red">{word}</span></button>'
            colored_words.append(colored_word)
        elif "Definition" in tag:
            colored_word = f'<button onclick="set_tag(this, \'O\')"><span style="color:blue">{word}</span></button>'
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

        # Define the JavaScript function to set the tag of a button
        set_tag_js = """
        <script>
        function set_tag(button, tag) {
            var sentence_index = button.getAttribute('data-sentence-index');
            var word_index = button.getAttribute('data-word-index');
            var tags_input = document.getElementById('tags-' + sentence_index);
            var tags = tags_input.value.split(' ');
            tags[word_index] = tag;
            tags_input.value = tags.join(' ');
            button.innerHTML = button.innerHTML.replace(/<[^>]*>/g, '');
            if (tag === 'Term') {
                button.innerHTML = '<span style="color:red">' + button.innerHTML + '</span>';
                button.setAttribute('onclick', 'set_tag(this, \'Definition\')');
            } else if (tag === 'Definition') {
                button.innerHTML = '<span style="color:yellow">' + button.innerHTML + '</span>';
                button.setAttribute('onclick', 'set_tag(this, \'O\')');
            } else {
                button.setAttribute('onclick', 'set_tag(this, \'Term\')');
            }
        }
        </script>
        """

        # Display the sentences with their tags
        st.write("## Sentences")
        for i in range(len(df)):
            sentence = df.iloc[i]["colored_tokens"]
            st.markdown(sentence, unsafe_allow_html=True)

            # Allow the user to edit the tags
            tags = df_edited.iloc[i]["tags_sequence_pred"].split()
            tags_input = st.empty()
            tags_input.text_input("Tags", " ".join(tags), key=f"tags-{i}")
            st.markdown(set_tag_js, unsafe_allow_html=True)

        # Allow the user to download the edited file
        if st.button("Download edited file"):
            csv = df_edited.to_csv(index=False, sep="\t")
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="edited_file.tsv">Download edited file</a>'
            st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":

    main()
