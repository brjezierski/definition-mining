import streamlit as st
import pandas as pd
import base64
import numpy as np

# Set up the logger
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a StreamHandler to display logs in Streamlit
# streamlit run edit_assignment.py


# Instantiate the Session State Variables
if 'cache' not in st.session_state:
    st.session_state.cache = {'start': 0, 'end': 0}


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
        elif "Alias-Term" in tag:
            colored_word = f'<span style="color:pink">{word}</span>'
            colored_words.append(colored_word)
        elif "Term" in tag and "Alias-Term" not in tag:
            colored_word = f'<span style="color:red">{word}</span>'
            colored_words.append(colored_word)
        elif "Definition" in tag:
            colored_word = f'<span style="color:blue">{word}</span>'
            colored_words.append(colored_word)
    colored_sentence = " ".join(colored_words)
    return colored_sentence


@st.experimental_memo
def simplify_tags(tags):
    tags = tags.split()
    updated_tags = []
    for tag in tags:
        updated_tags.append(tag.replace("B-", "").replace("I-", ""))
    return " ".join(updated_tags)


# @st.experimental_memo
def prepend_tags(tags, tokens):
    tokens = tokens.split()
    if tags == "-":
        tags = ""
        for _ in tokens:
            tags += "O "
        return tags.strip()
    tags = tags.split()
    updated_tags = []
    prev_tag = ""
    for tag in tags:
        tag = tag.lower()
        if "term" in tag or "t" == tag:
            if "Term" not in prev_tag:
                updated_tags.append("B-Term")
            else:
                updated_tags.append("I-Term")
        elif "definition" in tag or "d" == tag:
            if "Definition" not in prev_tag:
                updated_tags.append("B-Definition")
            else:
                updated_tags.append("I-Definition")
        elif "alias-term" in tag or "at" == tag:
            if "Alias-Term" not in prev_tag:
                updated_tags.append("B-Alias-Term")
            else:
                updated_tags.append("I-Alias-Term")
        elif "ordered-term" in tag or "ot" == tag:
            if "Ordered-Term" not in prev_tag:
                updated_tags.append("B-Ordered-Term")
            else:
                updated_tags.append("I-Ordered-Term")
        elif "referential-term" in tag or "rt" == tag:
            if "Referential-Term" not in prev_tag:
                updated_tags.append("B-Referential-Term")
            else:
                updated_tags.append("I-Referential-Term")
        elif "definition" in tag or "d" == tag:
            if "Definition" not in prev_tag:
                updated_tags.append("B-Definition")
            else:
                updated_tags.append("I-Definition")
        elif "secondary definition" in tag or "sd" == tag:
            if "Secondary Definition" not in prev_tag:
                updated_tags.append("B-Secondary Definition")
            else:
                updated_tags.append("I-Secondary Definition")
        elif "ordered definition" in tag or "od" == tag:
            if "Ordered Definition" not in prev_tag:
                updated_tags.append("B-Ordered Definition")
            else:
                updated_tags.append("I-Ordered Definition")
        elif "referential definition" in tag or "rd" == tag:
            if "Referential Definition" not in prev_tag:
                updated_tags.append("B-Referential Definition")
            else:
                updated_tags.append("I-Referential Definition")
        elif "qualifier" in tag or "q" == tag:
            if "Qualifier" not in prev_tag:
                updated_tags.append("B-Qualifier")
            else:
                updated_tags.append("I-Qualifier")
        elif "o" == tag:
            updated_tags.append("O")
        else:
            updated_tags.append(tag)
        prev_tag = tag
    if len(updated_tags) < len(tokens):
        updated_tags.extend(["O"] * (len(tokens) - len(updated_tags)))
    if len(updated_tags) > len(tokens):
        updated_tags[:-(len(updated_tags) - len(tokens))]
    return " ".join(updated_tags)


def update_sent_type(tags, sent_type, sent_type_score):
    for tag in tags.split():
        if tag != "O":
            new_sent_type = 1
            new_sent_type_score = sent_type_score if sent_type == new_sent_type else 0
            return new_sent_type, new_sent_type_score
    new_sent_type = 0
    new_sent_type_score = sent_type_score if sent_type == new_sent_type else 0
    return new_sent_type, new_sent_type_score


@st.experimental_memo
def sort(df):
    df = df.sort_values(
        by=["sent_type_pred", "sent_type_scores"], ascending=[False, True])
    return df


def form_callback(df_edited, i, tokens):
    tags_input_value = st.session_state[f"tags-{i}"]
    tags_input_value = prepend_tags(tags_input_value, tokens)
    df_edited.at[i, "tags_sequence_pred"] = tags_input_value
    st.session_state.cache["df"] = df_edited


def upload_file_callback():
    st.experimental_memo.clear()


def main():
    st.set_page_config(page_title="Tag Editor", page_icon=":pencil2:")

    # Get the file from the user
    uploaded_file = st.file_uploader("Choose a TSV file", type=[
                                     "tsv"], on_change=upload_file_callback)

    if uploaded_file is not None:
        # Read the file into a DataFrame
        if "df" not in st.session_state.cache:
            df = pd.read_csv(uploaded_file, sep="\t")
            df = sort(df)
            df = df.drop_duplicates(subset='tokens', keep="first")
            df = df.reset_index(drop=True)
            for index, row in df.iterrows():
                if isinstance(row['tokens'], float):
                    df.at[index, 'tokens'] = np.nan
            df = df.dropna()
            # Create a copy of the DataFrame to store the user's changes
            df_edited = df.copy()
            st.session_state.cache["df"] = df_edited
        else:
            df_edited = st.session_state.cache["df"]
            df_edited = sort(df_edited)

        df_edited["tags_sequence_pred"] = df_edited.apply(
            lambda row: simplify_tags(row["tags_sequence_pred"]), axis=1)

        # Colorize the sentences based on their tags
        df_edited["colored_tokens"] = df_edited.apply(lambda row: colorize_text(
            row["tokens"], row["tags_sequence_pred"]), axis=1)

        # Set up pagination
        page_size = 20
        page_count = int((len(df_edited) / page_size) + 1)
        current_page = st.session_state.get("current_page", 1)

        st.write("## Sentences")

        def update_page():
            col2.write(f"Page {current_page}/{page_count}")
            st.session_state.cache["start"] = max(
                (current_page - 1) * page_size, 0)
            st.session_state.cache["end"] = min(
                current_page * page_size, len(df_edited))
            st.session_state.cache["df"] = df_edited

        # Display the current page number
        if st.button("Update"):
            current_page = current_page

        # Add navigation buttons
        col1, col2, col3 = st.columns(3)
        if col1.button("Prev", key="prev1") and current_page > 1:
            current_page -= 1
            update_page()
            if col3.button("Next", key="next1") and current_page < page_count - 1:
                current_page += 1
                update_page()
        elif col3.button("Next", key="next1") and current_page < page_count - 1:
            current_page += 1
            update_page()
        else:
            update_page()
        st.session_state["current_page"] = current_page

        to_refresh = False
        for i in range(st.session_state.cache["start"], st.session_state.cache["end"]):
            sentence = df_edited.iloc[i]["colored_tokens"]
            st.markdown(f"<br />No. {i} <br />{sentence}",
                        unsafe_allow_html=True)
            st.markdown("Pred: "+str(df_edited.iloc[i]["sent_type_pred"])+", score:"+str(df_edited.iloc[i]
                                                                                         ["sent_type_scores"]), unsafe_allow_html=True)

            tags = df_edited.iloc[i]["tags_sequence_pred"].split()
            tokens = df_edited.iloc[i]["tokens"]
            tags_input = st.empty()

            tags_input.text_input(
                "Tags", value=" ".join(
                    tags), key=f"tags-{i}", on_change=form_callback, args=(df_edited, i, tokens))

            if st.button("Delete", key=f"delete-{i}"):
                df_edited_new = df_edited.drop(i)
                df_edited_new.reset_index(drop=True, inplace=True)
                st.session_state.cache["df"] = df_edited_new
                to_refresh = True
            if to_refresh and i == st.session_state.cache["end"]-1:
                st.experimental_rerun()

        # Allow the user to download the edited file
        if st.button("Download edited file"):
            df_edited["tags_sequence_pred"] = df_edited.apply(
                lambda row: prepend_tags(row["tags_sequence_pred"], row["tokens"]), axis=1)
            df_edited["sent_type_pred"], df_edited["sent_type_scores"] = zip(*df_edited.apply(
                lambda row: update_sent_type(row["tags_sequence_pred"], row["sent_type_pred"], row["sent_type_scores"]), axis=1))
            st.session_state.cache["df"] = df_edited
            csv = df_edited.to_csv(index=False, sep="\t")
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="edited_file.tsv">Download edited file</a>'
            st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":

    main()
