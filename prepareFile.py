import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk import word_tokenize


def prepareFile(uncleaned_df):
    # uncleaned_df = pd.read_csv(filename)

    # Selecting specific columns (Timestamp, Event, Utterance) and creating a new DataFrame
    cleaned_df = uncleaned_df[["Timestamp", "Event", "Utterance"]].copy()
    cleaned_df["selected"] = 0
    cleaned_df["missed"] = 0
    cleaned_df["selected / missed"] = "-"
    cleaned_df["WP or SP?"] = "-"

    df = uncleaned_df

    # get range of text to determine whether missed /selected
    possible_matched_Utterance = getSimilarText(df)

    # selected text computation
    # this removes text that are not in the output-updated for the specfied range
    # eg. t,th,and,a, this -> t, th, this
    matched_Utterance = removeUnrelatedText(possible_matched_Utterance)

    # checks which text was selected and assigns the 4 columns(selected,missed, selected / missed, WP/SP?)
    # it assigns values of 1, 0 , the selected text from Utterance and whether it is a word or sentence
    determineSelectedText(matched_Utterance, cleaned_df)

    # missed text computation

    nearestTTSRows = findNearestRows(possible_matched_Utterance, cleaned_df)
    determineMissedText(matched_Utterance, nearestTTSRows, cleaned_df)

    # cleaned_df.to_csv('prepared.csv',index=False)
    # return
    return cleaned_df


def getSimilarText(df):
    matched_Utterance = []
    for index, row in df.iterrows():
        current_event = row["Event"]
        current_Utterance = row["Utterance"]
        current_timestamp = row["Timestamp"]

        # Find the row above with Utterance = "Current Location and Identification Tags Updated"
        previous_row = df[
            (df.index < index)
            & (df["Utterance"] == "Current Location and Identification Tags Updated")
            ].tail(1)

        if not previous_row.empty:
            previous_Utterance = previous_row["Utterance"].iloc[0]

            # Check if any rows within the specified range have the same Utterance value
            Utterance_match_rows = df[
                (df.index > previous_row.index[0]) & (df.index <= index)
                ]

            if not Utterance_match_rows.empty:
                # Add matching event and Utterance values to the matched_Utterance list as values in a dictionary
                matched_dict = {
                    current_Utterance: Utterance_match_rows[
                        ["Event", "Utterance"]
                    ].to_dict(orient="records")
                }

                matched_Utterance.append(matched_dict)
            else:
                # If no match, add a dictionary with the output-updated Utterance value as key and an empty list as the value
                no_match_dict = {current_Utterance: []}
                matched_Utterance.append(no_match_dict)


    return matched_Utterance


def removeUnrelatedText(possible_matched_Utterance):
    # Filter out unrelated input to determine selected and missed later
    for result in possible_matched_Utterance:
        outer_key = list(result.keys())[0]
        output_updated = outer_key

        # Access the inner list in each dictionary
        inner_list = result.get(outer_key, [])
        # Use list comprehension to create a new list without undesired sub-dictionaries
        result[outer_key] = [
            item for item in inner_list if str(output_updated) in item["Utterance"]
        ]

        break
    return possible_matched_Utterance


def determineSelectedText(matched_Utterance, cleaned_df):
    for result in matched_Utterance:
        outer_key = list(result.keys())[0]
        output_updated = outer_key

        # Access the inner list in each dictionary
        inner_list = result.get(outer_key, [])

        for item in inner_list:
            event = item["Event"]
            Utterance = item["Utterance"]

            if event != "output-updated":
                # Split the Event value by '-'
                event_split = event.split("-")

                # Check if the last item in the split list is 'selected'
                if event_split[-1] == "selected":
                    # Check if the first item in the split list is either "word" or "sentence"
                    if event_split[0] in ["word", "sentence"]:
                        # Update cleaned_df based on the criteria
                        cleaned_df.loc[cleaned_df["Event"] == event, "selected"] = 1
                        cleaned_df.loc[cleaned_df["Event"] == event, "missed"] = 0
                        cleaned_df.loc[
                            cleaned_df["Event"] == event, "selected / missed"
                        ] = Utterance
                        cleaned_df.loc[cleaned_df["Event"] == event, "WP or SP?"] = (
                            event_split[0]
                        )
    positionSelectedWell(cleaned_df)


def positionSelectedWell(cleaned_df):
    # Iterate through the rows of cleaned_df
    for index, row in cleaned_df.iterrows():
        if row["selected"] == 1:
            # Copy values for selected, missed, selected / missed, WP or SP?
            selected_value = row["selected"]
            missed_value = row["missed"]
            selected_missed_value = row["selected / missed"]
            wp_sp_value = row["WP or SP?"]

            # Find the next row below with Event value 'output-updated'
            next_row_index = cleaned_df.index[
                (cleaned_df.index > index) & (cleaned_df["Event"] == "output-updated")
                ].min()

            if next_row_index is not None:
                # Update the next row's selected, missed, selected / missed, WP or SP? values
                cleaned_df.at[next_row_index, "selected"] = selected_value
                cleaned_df.at[next_row_index, "missed"] = missed_value
                cleaned_df.at[next_row_index, "selected / missed"] = (
                    selected_missed_value
                )
                cleaned_df.at[next_row_index, "WP or SP?"] = wp_sp_value

                # Set its own values to 0, 0, "", and ""
                cleaned_df.at[index, "selected"] = 0
                cleaned_df.at[index, "missed"] = 0
                cleaned_df.at[index, "selected / missed"] = "-"
                cleaned_df.at[index, "WP or SP?"] = "-"


def findNearestRows(matched_Utterance, df):
    pairOfNearestRows = []
    for index, row in df.iterrows():
        current_event = row["Event"]
        current_Utterance = row["Utterance"]
        nearestTTSRow_Utterance = ""

        if current_event == "output-updated":
            # Find the row below with event 'TTS-button-selected'
            next_row = df[(df.index > index) & (df["Event"] == "TTS-button-selected")].head(1)

            if not next_row.empty:
                nearestTTSRow_Utterance = next_row["Utterance"].iloc[0]
                if nearestTTSRow_Utterance not in pairOfNearestRows:
                    pairOfNearestRows.append(nearestTTSRow_Utterance)

    return pairOfNearestRows


def determineMissedText(matched_Utterance, nearestTTSRows, cleaned_df):
    for result in matched_Utterance:
        outer_key = list(result.keys())[0]
        inner_list = result.get(outer_key, [])

        for item in inner_list:
            event = item["Event"]
            Utterance = item["Utterance"]

            # Check if any of the Utterance values are equal to any of the values in the nearestTTSRows list
            for nearestTTSRow in nearestTTSRows:
                nearestTTSRow_words = nearestTTSRow.split()

                if Utterance in nearestTTSRow_words and len(Utterance.split()) == len(nearestTTSRow_words):
                    # Check if the event selected is the same as the value that matches the word in the nearestTTSRows list
                    if event == f"{nearestTTSRow_words.index(Utterance) + 1}-selected":
                        # Update cleaned_df based on the criteria
                        cleaned_df.loc[cleaned_df["Event"] == event, "selected"] = 0
                        cleaned_df.loc[cleaned_df["Event"] == event, "missed"] = 1
                        cleaned_df.loc[
                            cleaned_df["Event"] == event, "selected / missed"
                        ] = Utterance
                        cleaned_df.loc[cleaned_df["Event"] == event, "WP or SP?"] = (event.split("-")[0])
    positionMissedWell(cleaned_df)


def positionMissedWell(cleaned_df):
    # Iterate through the rows of cleaned_df
    for index, row in cleaned_df.iterrows():
        if row["missed"] == 1:
            # Copy values for selected, missed, selected / missed, WP or SP?
            selected_value = row["selected"]
            missed_value = row["missed"]
            selected_missed_value = row["selected / missed"]
            wp_sp_value = row["WP or SP?"]

            # Find the next row below with Event value 'output-updated'
            next_row_index = cleaned_df.index[
                (cleaned_df.index > index) & (cleaned_df["Event"] == "output-updated")
                ].min()

            if next_row_index is not None:
                # Update the next row's selected, missed, selected / missed, WP or SP? values
                cleaned_df.at[next_row_index, "selected"] = selected_value
                cleaned_df.at[next_row_index, "missed"] = missed_value
                cleaned_df.at[next_row_index, "selected / missed"] = (
                    selected_missed_value
                )
                cleaned_df.at[next_row_index, "WP or SP?"] = wp_sp_value

                # Set its own values to 0, 0, "", and ""
                cleaned_df.at[index, "selected"] = 0
                cleaned_df.at[index, "missed"] = 0
                cleaned_df.at[index, "selected / missed"] = "-"
                cleaned_df.at[index, "WP or SP?"] = "-"


def predict_next_sentence(text, model, max_length=6, temperature=None):
    # Preprocess text (e.g., add start/end tokens)
    processed_text = preprocess_text(text)
    predictions = model.generate(processed_text, max_length=max_length, temperature=temperature)
    return predictions[0]  # Return the generated sentence


def preprocess_text(text):
    """
    Preprocesses text for word or sentence prediction tasks.

    Args:
        text: The text to preprocess.

    Returns:
        The preprocessed text.
    """

    # Lowercase
    text = text.lower()

    # Remove punctuation and special characters (customizable)
    import string
    punctuation = string.punctuation + "’“”"  # Add additional characters if needed
    text = text.replace(f"[^{''.join(word_tokenize(' '.join(c for c in punctuation if c.isalpha())))}]", "", regex=True)

    # Remove stopwords (optional)
    # uncomment if you want to remove stopwords
    # stop_words = stopwords.words("english")
    # text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# prepareFile(r"C:\Users\hp\Videos\CSVMetricsForKingsley\new_files\27112019-101440.csv")
