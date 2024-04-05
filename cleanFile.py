import pandas as pd
import os
import re
from prepareFile import *


# Data Extraction and Loading
def convert_to_csv(filename):
    read_file = pd.read_excel(filename)
    original_file_name = os.path.splitext(os.path.basename(filename))[0]

    read_file.to_csv(f"{original_file_name}.csv", index=None, header=True)
    return original_file_name


def removeTTSDuplicates(df):
    rows_to_remove = []

    for row in range(len(df) - 1):  # Adjust loop condition
        current_event = df.at[row, "Event"]
        next_row = row + 1

        if next_row < len(df):  # Check if next row is within DataFrame index range
            next_event = df.at[next_row, "Event"]

            if current_event == "TTS-button-selected":
                if next_event != "output-updated":
                    # Mark the next row for removal
                    rows_to_remove.append(next_row)
                else:
                    # Skip to the next 'output-updated' event
                    while (
                            next_row < len(df)
                            and df.at[next_row, "Event"] == "output-updated"
                    ):
                        next_row += 1
            elif current_event == "output-updated":
                # Reset the check when encountering 'output-updated'
                continue

    # Remove the identified rows
    df.drop(index=rows_to_remove, inplace=True)


def filterTTSRows(df):
    # Remove the 'RowNo' column
    if 'RowNo' in df.columns:
        df.drop(columns=["RowNo"], inplace=True)
    return df[df["Event"] == "TTS-button-selected"].copy()


# Data Transformation -> Column creation
def getWordCount(cell_value):
    if pd.isnull(cell_value):
        return 0
    return len(cell_value.strip().split())


# Timestamp Format
def timeStamp(df):
    try:
        # Try parsing with format for date and time
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M:%S.%f").dt.time
    except ValueError:
        try:
            # If parsing fails, try format for time only
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%M:%S.%f").dt.time
        except ValueError:
            # If both formats fail, raise an error
            print("Invalid datetime format")


# Typing speed computations
def calculateTimeInMinutesAndSeconds(df):
    df["Time (mins)"] = "-"
    df["Time (secs)"] = "-"
    # print(df)
    # Convert 'Timestamp' column to datetime format (assuming format "%M:%S.%f")
    timeStamp(df)

    for index, row in df.iterrows():
        current_event = row["Event"]
        current_Utterance = row["Utterance"]
        current_timestamp = row["Timestamp"]

        if current_event == "TTS-button-selected":
            # Find the previous row with length 1 in the 'Utterance' column (assuming 'Utterance' is a string)
            previous_row = df.loc[(df.index < index) & (df["Utterance"].str.len() == 1), :]
            if not previous_row.empty:
                previous_Utterance = previous_row["Utterance"].iloc[0]
                previous_timestamp = previous_row["Timestamp"].iloc[0]

                # Calculate time difference in minutes and seconds
                current_minutes = current_timestamp.minute
                current_seconds = current_timestamp.second + current_timestamp.microsecond / 1e6  # convert microseconds to seconds
                previous_minutes = previous_timestamp.minute
                previous_seconds = previous_timestamp.second + previous_timestamp.microsecond / 1e6

                # Handle minute rollover
                if current_minutes < previous_minutes:
                    current_minutes += 60

                time_difference_minutes = current_minutes - previous_minutes
                time_difference_seconds = current_seconds - previous_seconds

                # # Ensure positive values and handle potential negative seconds from rollover
                if time_difference_seconds < 0:
                    time_difference_minutes -= 1
                    time_difference_seconds += 60

                df.at[index, "Time (mins)"] = time_difference_minutes
                df.at[index, "Time (secs)"] = round(time_difference_seconds, 2)


def calculateWordsPerMinute(df):
    df["Time (mins)"] = pd.to_numeric(df["Time (mins)"], errors="coerce")
    df["Words per minute"] = round(df["Word Count"] / df["Time (mins)"], 2)


def calculateCharacterCount(df):
    df["Character Count"] = df["Utterance"].str.len()


def calculateTotalCharacterCount(df):
    # Add 1 to the character count if the 'Event' is 'TTS-button-selected'
    df["Total Character Count"] = df.apply(
        lambda row: (
            len(row["Utterance"]) + 1
            if row["Event"] == "TTS-button-selected"
            else len(row["Utterance"])
        ),
        axis=1,
    )


def calculateTimeSeconds(df):
    df["Time (sec)"] = df["Time (mins)"] * 60


def calculateCharactersPerSecond(df):
    df["Time (secs)"] = pd.to_numeric(df["Time (secs)"], errors='coerce')

    df["Characters per Second"] = round(
        df["Character Count"] / df["Time (secs)"], 2)


# these functions get the WPM and CPS from the dataset and compute the time accordingly
def getTypingRates(df):
    for index, row in df.iterrows():
        current_event = row["Event"]
        current_Utterance = row["Utterance"]
        current_timestamp = row["Timestamp"]

        if current_event == "TTS-button-selected":
            # Check if there are two subsequent rows with the specified events
            subsequent_rows = df[(df.index > index) & (df.index <= index + 2)]
            if len(subsequent_rows) == 2 and subsequent_rows["Event"].tolist() == [
                "words-per-minute-value_word_count",
                "words-per-minute-value_character_count",
            ]:
                # Extract Utterance values
                words_per_minute = subsequent_rows.iloc[0]["Utterance"]
                characters_per_second = subsequent_rows.iloc[1]["Utterance"]

                # Update the TTS row with the extracted values
                df.at[index, "Words per minute"] = words_per_minute
                df.at[index, "Characters per Second"] = characters_per_second


def derivedTimeInMins(df):
    # Compute Time (mins) based on Word Count and Words per minute
    df["Derived Time (mins)"] = df["Word Count"] / df["Words per minute"]


def derivedTimeInSecs(df):
    # Compute Time (sec) based on Character Count and Characters per Second
    df["Derived Time (sec)"] = df["Character Count"] / df["Characters per Second"]


# lexicon and keystrokes
def computePredictedText(df):
    for index, row in df.iterrows():
        current_event = row["Event"]
        current_Utterance = row["Utterance"]
        current_timestamp = row["Timestamp"]

        if current_event == "TTS-button-selected":
            # Find the row above with length 1 in the 'Utterance' column
            previous_row = df[(df.index < index) & (df["Utterance"].str.len() == 1)].tail(
                1
            )

            if not previous_row.empty:
                previous_Utterance = previous_row["Utterance"].iloc[0]
                # Find all rows within the range and with 'selected' value of 1
                selected_rows = df[
                    (df.index > previous_row.index[0])
                    & (df.index <= index)
                    & (df["selected"] == 1)
                    ]

                # Extract words and sentences based on 'WP or SP?' column
                predicted_words = selected_rows[selected_rows["WP or SP?"] == "word"][
                    "selected / missed"
                ].tolist()
                predicted_sentences = selected_rows[
                    selected_rows["WP or SP?"] == "sentence"
                    ]["selected / missed"].tolist()
                # print(predicted_words)
                # print(predicted_sentences)

                # Update the current row with the extracted values
                df.at[index, "Predicted Words Used"] = ", ".join(
                    predicted_words)
                df.at[index, "No. of Predicted Words Used"] = len(
                    predicted_words)
                df.at[index, "Predicted Sentences Used"] = ", ".join(
                    predicted_sentences
                )
                num_words_used_in_sentences = sum(
                    len(sentence.split()) for sentence in predicted_sentences
                )
                df.at[index, "No. of Words Used in Predicted Sentences"] = (
                    num_words_used_in_sentences
                )
                df.at[index, "No. of Predicted Sentences Used"] = len(
                    predicted_sentences
                )


def determineKeyStrokes(df):
    zero_one_pairs = []

    for index, row in df.iterrows():
        current_event = row["Event"]
        current_Utterance = row["Utterance"]
        current_selected = row["selected"]

        if current_event == "TTS-button-selected":
            # Find the row above with length 1 in the 'Utterance' column
            previous_row = df[(df.index < index) & (df["Utterance"].str.len() == 1)].tail(
                1
            )

            if not previous_row.empty:
                previous_Utterance = previous_row["Utterance"].iloc[0]

                # Find all rows within the range and with 'selected' value of 1
                selected_rows = df[
                    (df.index > previous_row.index[0])
                    & (df.index <= index)
                    & (df["selected"] == 1)
                    ]

                row_above_selected_text_index = selected_rows.index - 1
                row_above_selected_text = df.loc[row_above_selected_text_index]
                # print(row_above_selected_text)
                pressed = row_above_selected_text["Utterance"].tolist()
                underscored = [entry.split()[-1] for entry in pressed]
                actual_keys_pressed = "_".join(underscored)
                # zero_one_pairs.append(pressed)
                if not selected_rows.empty:
                    temp = [pressed, selected_rows["Utterance"].tolist()]
                    zero_one_pairs.append(temp)
                    df.at[index, "Actual Keys pressed"] = actual_keys_pressed + "_>"


def handle_length(value, np):
    if isinstance(value, str):
        # Option 1: Exclude special characters using regular expressions (adjust pattern as needed)
        # cleaned_value = re.sub(r'[^\w\s]', '', value)  # Remove non-alphanumeric characters and spaces
        # print(len(cleaned_value))
        # return len(cleaned_value)

        # Option 2: Replace special characters (adjust replacement as needed)
        cleaned_value = value.replace('_', '')  # Replace underscore with empty string
        print(len(cleaned_value))
        return len(cleaned_value)
    else:
        return np.nan


def countKeystrokes(df):
    df["Actual Keystrokes"] = df["Actual Keys pressed"].str.len()
    df["Actual Keystroke Savings"] = (
            df["Total Character Count"] - df["Actual Keystrokes"]
    )


def underscoredWord(input_text):
    words = input_text.split()
    result = "_".join(word[0].lower() for word in words) + "_>"
    return result


def altKeystrokes(df):
    df["Keys pressed if user used predictions the first time they appeared"] = df[
        "Actual Keys pressed"
    ]
    selected_rows = df[df["Event"] == "TTS-button-selected"]
    df.loc[
        selected_rows.index,
        "Min. keys user has to press if system could predict all current words and next words/phrases in the lexicon (at each moment) ASAP and user uses all those predictions the first time they appear",
    ] = selected_rows["Utterance"].apply(underscoredWord)
    df["Keystrokes"] = df[
        "Min. keys user has to press if system could predict all current words and next words/phrases in the lexicon (at each moment) ASAP and user uses all those predictions the first time they appear"
    ].str.len()
    df["Keystrokes Saved"] = df["Total Character Count"] - df["Keystrokes"]


def lexiconText(df):
    # Words used in prediction (WP)
    df["Words in the lexicon (WP)"] = df["Predicted Words Used"]
    df["Theoretical min character (WP)"] = df["Words in the lexicon (WP)"].apply(
        lambda x: x.count(",") + 1 if pd.notna(x) and x.strip() != "" else 0
    )

    # Create temporary columns for split operation
    df["Temp Split"] = df["Predicted Sentences Used"].apply(
        lambda x: x.split(",") if pd.notna(x) else []
    )

    # Separate single words and phrases
    df["Words in the lexicon (SP)"] = df["Temp Split"].apply(
        lambda x: ", ".join([word.strip()
                             for word in x if len(word.split()) == 1])
    )
    # Theoretical min character for SP
    df["Theoretical min character (Word SP)"] = df["Words in the lexicon (SP)"].apply(
        lambda x: x.count(",") + 1 if pd.notna(x) and x.strip() != "" else 0
    )

    df["Phrases in the lexicon (SP)"] = df["Temp Split"].apply(
        lambda x: ", ".join([word.strip()
                             for word in x if len(word.split()) > 1])
    )

    df["Theoretical min character (Phrase SP)"] = df[
        "Phrases in the lexicon (SP)"
    ].apply(lambda x: x.count(",") + 1 if pd.notna(x) and x.strip() != "" else 0)

    # Drop temporary columns
    df.drop(columns=["Temp Split"], inplace=True)


def theoreticalKeyStrokes(df):
    df["Keystrokes involved in selecting predictions"] = (
            df["Theoretical min character (WP)"]
            + df["Theoretical min character (Word SP)"]
            + df["Theoretical min character (Phrase SP)"]
    )

    df["Total min. char (including TTS button)"] = (
            1  # tts buttton
            + df["Theoretical min character (WP)"]
            + df["Theoretical min character (Word SP)"]
            + df["Theoretical min character (Phrase SP)"]
            + df["Keystrokes involved in selecting predictions"]
    )

    df["Keystrokes saved (lexicon)"] = (
            df["Total Character Count"] -
            df["Total min. char (including TTS button)"]
    )


# Define a function to check if the value appears elsewhere
# Define a function to check if the value appears elsewhere (excluding the row itself)
def is_referenced(value, df, event_col="Event"):
    # Check if the value appears in any other column of any other row
    return df[df[event_col] != value][event_col].isin([value]).any()


def removeNonTTSRows(df):
    # Filter data based on the event column and referenced values
    filtered_data = df[df["Event"] == "TTS-button-selected"]
    temp = df[df.apply(lambda row: is_referenced(row["Event"], df), axis=1)]
    filtered_data = pd.concat([filtered_data, temp], ignore_index=True)
    return filtered_data
    # Save the filtered data as a new CSV
    # filtered_data.to_csv("filtered_data.csv", index=False)


# file indexing
def finalOutputforMultipleFiles(df, originalFile, fileCount):
    # Get the current filename
    file_name = os.path.basename(originalFile)

    # Get the appropriate file count for this dataframe
    current_file_count = next(fileCount)  # Get the next count from the generator

    # Add File Count and File Name columns
    df.insert(0, "File Count", current_file_count)
    df.insert(1, "File Name", file_name)

    # Generate a unique filename with count (optional)
    unique_filename = f"static/uploads/cleaned_{file_name}.csv"  # Modify extension if needed

    # Save the dataframe to CSV
    df.to_csv(unique_filename, index=False)  # Don't save index

# def calculateForwporsp(df):
#     df["wp/sp"] = df[""]

# if __name__ == "__main__":
#     filename = "sample.xlsx"
#     process_excel_file(filename)
