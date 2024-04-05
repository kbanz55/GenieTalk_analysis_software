import os  # for path manipulation
import tempfile
import uuid  # for generating unique filenames
from datetime import datetime, timedelta
from io import StringIO  # for converting dataframe to HTML
from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd  # for CSV reading
import numpy as np
from flask import render_template, request
from google.cloud import storage, client
import shutil
import zipfile

from tqdm import tqdm
from werkzeug.utils import secure_filename  # for secure filename handling
from cleanFile import *
from prepareFile import *
import plotly.express as px

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

app = Flask(__name__)

# Replace with your bucket name
bucket_name = "saved_datasets1"


def compile_csv():
    # Get the path to the 'static/uploads' folder
    uploads_folder = os.path.join(app.root_path, "static", "uploads")

    # List all files that start with "cleaned_"
    cleaned_files = [f for f in os.listdir(uploads_folder) if f.startswith("cleaned_")]

    if not cleaned_files:
        print("No cleaned files found for joining.")
        return

    # Initialize an empty list to store DataFrames
    dfs_to_concat = []

    # Iterate through each cleaned file and append its DataFrame to the list
    for cleaned_file in cleaned_files:
        cleaned_filepath = os.path.join(uploads_folder, cleaned_file)
        df = pd.read_csv(cleaned_filepath)
        dfs_to_concat.append(df)

    # Concatenate all DataFrames in the list into the final DataFrame
    final_df = pd.concat(dfs_to_concat, ignore_index=True)

    # Save the final result to a new CSV file
    final_result_filename = "final_result.csv"
    final_result_filepath = os.path.join(uploads_folder, final_result_filename)
    # Filtered for TTS
    TTS = filterTTSRows(final_df)
    TTS.to_csv(final_result_filepath, index=False)
    print(f"Final result saved to: {final_result_filepath}")


def compute_metrics():
    # Get the path to the 'static/uploads' folder
    uploads_folder = os.path.join(app.root_path, "static", "uploads")

    # List all files that start with "cleaned_"
    cleaned_files = [f for f in os.listdir(uploads_folder) if f.startswith("cleaned_")]
    if not cleaned_files:
        print("No cleaned files found for joining.")
        return

    # Load the final_result.csv file
    final_result_path = os.path.join(uploads_folder, "final_result.csv")
    final_df = pd.read_csv(final_result_path)

    # Initialize an empty DataFrame to store metrics and values
    metrics_df = pd.DataFrame(columns=["Metrics", "Value"])

    # Compute Total Word Count
    total_word_count = final_df["Word Count"].sum()
    metrics_df.loc[len(metrics_df.index)] = ["Total Word Count", total_word_count]

    # Compute Total Time (mins)
    total_time_mins = final_df["Time (mins)"].sum()
    metrics_df.loc[len(metrics_df.index)] = ["Total Time (mins)", total_time_mins]

    # Compute Total Utterance Count for TTS-button-selected
    column_to_access = 'Event'
    # Filter by name (replace 'name_to_filter' with the actual name)
    filtered_data = final_df[final_df[column_to_access] == 'TTS-button-selected']
    # Sum the filtered data (assuming numerical column)
    total_Utterance_count = filtered_data[column_to_access].count()
    metrics_df.loc[len(metrics_df.index)] = ["Total Utterance Count (TTS-button-selected)", total_Utterance_count]

    # Compute Words Per Minute
    words_per_minute = round(total_word_count / total_time_mins, 2)
    metrics_df.loc[len(metrics_df.index)] = ["Words Per Minute", words_per_minute]

    # Compute Total Character Count
    total_char_count = final_df["Character Count"].sum()
    metrics_df.loc[len(metrics_df.index)] = ["Total Character Count", total_char_count]

    # Compute Total Character Count (TTS)
    total_character_count_tts = final_df["Total Character Count"].sum()
    metrics_df.loc[len(metrics_df.index)] = [
        "Total Character Count (TTS)",
        total_character_count_tts,
    ]

    # Compute Total Time (sec)
    total_time_sec = final_df["Time (secs)"].sum()
    metrics_df.loc[len(metrics_df.index)] = ["Total Time (sec)", round(total_time_sec, 2)]

    # Compute Total Character Per Second
    total_char_per_sec = round(total_char_count / total_time_sec, 2)
    metrics_df.loc[len(metrics_df.index)] = [
        "Total Character Per Second",
        total_char_per_sec,
    ]

    # # Compute Total Actual Keystrokes
    total_actual_keystrokes = final_df["Actual Keystrokes"].sum()
    metrics_df.loc[len(metrics_df.index)] = [
        "Total Actual Keystrokes",
        total_actual_keystrokes,
    ]

    # Compute Total Actual Keystroke Savings
    total_actual_keystroke_savings = final_df["Actual Keystroke Savings"].sum()
    metrics_df.loc[len(metrics_df.index)] = [
        "Total Actual Keystroke Savings",
        total_actual_keystroke_savings,
    ]

    # Compute Actual Keystroke Savings (%)
    actual_keystroke_savings_percentage = round((
                                                        total_actual_keystroke_savings / total_character_count_tts
                                                ) * 100, 2)
    metrics_df.loc[len(metrics_df.index)] = [
        "Actual Keystroke Savings (%)",
        actual_keystroke_savings_percentage,
    ]

    # Compute Total Keystrokes
    total_keystrokes = final_df["Keystrokes"].sum()
    metrics_df.loc[len(metrics_df.index)] = ["Total Keystrokes", total_keystrokes]

    # Compute Total Keystrokes Saved
    total_keystrokes_saved = final_df["Keystrokes Saved"].sum()
    metrics_df.loc[len(metrics_df.index)] = ["Total Keystrokes Saved", total_keystrokes_saved - total_keystrokes_saved]

    # Compute Theoretical Keystroke Savings (Liberal Estimate for Vocab Limit) (%)
    theoretical_keystroke_savings_percentage = round((
                                                             total_keystrokes_saved / total_character_count_tts
                                                     ) * 100, 2)
    metrics_df.loc[len(metrics_df.index)] = [
        "Theoretical Keystroke Savings (Liberal Estimate for Vocab Limit) (%)",
        theoretical_keystroke_savings_percentage,
    ]

    # Compute Aggregated Total Min. characters (including TTS button)
    aggregated_total_min_characters = final_df[
        "Total min. char (including TTS button)"
    ].sum()
    metrics_df.loc[len(metrics_df.index)] = [
        "Aggregated Total Min. characters (including TTS button)",
        aggregated_total_min_characters,
    ]

    # Compute Total Keystrokes saved (lexicon)
    total_keystrokes_saved_lexicon = final_df["Keystrokes saved (lexicon)"].sum()
    metrics_df.loc[len(metrics_df.index)] = [
        "Total Keystrokes saved (lexicon)",
        total_keystrokes_saved_lexicon,
    ]

    # Compute Theoretical Keystroke Savings (Conservative Estimate for Vocab Limit) (%)
    theoretical_keystroke_savings_conservative_percentage = round((
                                                                          total_keystrokes_saved_lexicon / total_character_count_tts
                                                                  ) * 100, 2)
    metrics_df.loc[len(metrics_df.index)] = [
        "Theoretical Keystroke Savings (Conservative Estimate for Vocab Limit) (%)",
        theoretical_keystroke_savings_conservative_percentage,
    ]

    # Save the computed metrics to a new CSV file
    metrics_output_path = os.path.join(uploads_folder, "computed_metrics.csv")
    metrics_df.to_csv(metrics_output_path, index=False)


def rename(df):
    # Rename the column 'Content' to 'Utterance'
    df.rename(columns={'Content': 'Utterance'}, inplace=True)
    return df


def upload_files():
    upload_folder = "static/uploads"  # Replace with your upload folder path

    # Get filenames from the folder
    filenames = [f for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]
    # Path to your JSON credentials file
    credentials_path = "myflix-411119-161bc9896b4f.json"  # Replace with actual path

    # Authenticate with service account credentials
    try:
        storage_client = storage.Client.from_service_account_json(credentials_path)
    except FileNotFoundError:
        print(f"Error: Could not find credentials file at {credentials_path}")
        return False  # Indicate failure

    for filename in filenames:
        # Create full path with folder structure
        full_path = os.path.join(upload_folder, filename)
        with open(full_path, "rb") as file:
            # Upload from file (binary mode for various file types)
            # storage_client = storage.Client()
            blob = storage_client.bucket(bucket_name).blob(filename)
            blob.upload_from_file(file)


# Assuming a file upload field named "files" that allows multiple files
def uploadFile(request, upload_folder="static/uploads"):
    files = request.files.getlist('files')

    if not files:
        return render_template('home.html', message='No files selected')

    all_data = []  # List to store dataframes from all CSV files
    all_data2 = []  # List to store dataframes from all CSV files
    all_data3 = []  # List to store dataframes from all CSV files
    all_data_html = []  # List to store HTML representations of dataframes
    all_data_html2 = []  # List to store HTML representations of dataframes
    all_data_html3 = []  # List to store HTML representations of dataframes
    enumerated_data = []
    enumerated_data2 = []
    enumerated_data3 = []
    # num_files = len(files)
    fileCounts = iter(range(1, len(files) + 1))  # Create a generator for file counts
    # Process each uploaded file
    for file in files:
        if not file:  # Check if the file object exists
            continue  # Skip to the next iteration if no file was uploaded

        filename = secure_filename(file.filename)

        # Save the uploaded file with the unique filename
        file.save(os.path.join(upload_folder, filename))

        if filename.lower().endswith('.csv'):
            df = pd.read_csv(os.path.join(upload_folder, filename))  # Read using unique path
            rename(df)
            all_data.append(df)
            # Convert dataframe to HTML string
            buffer = StringIO()
            df.to_html(buffer, classes='table table-bordered h6 datatables no-wrap', index=False)
            data_html = buffer.getvalue()
            buffer.close()
            all_data_html.append(data_html)

            df.fillna("-", inplace=True)
            df = prepareFile(df)
            # Word count
            df["Word Count"] = df["Utterance"].apply(getWordCount)
            removeTTSDuplicates(df)
            # Time calculation
            # Typing speed computations
            # Character Count calculation
            calculateCharacterCount(df)
            calculateTotalCharacterCount(df)

            # Calculate Metrics from Sheet
            # Time Minutes&Seconds Calculation
            calculateTimeInMinutesAndSeconds(df)

            # Words Per Minute calculation
            calculateWordsPerMinute(df)
            # Characters per second calculation
            calculateCharactersPerSecond(df)

            # Derive Metrics from Sheet
            # go to cleanFile.py and uncomment definitions
            getTypingRates(df)
            derivedTimeInSecs(df)
            derivedTimeInMins(df)
            # Remove non-TTS rows
            removeNonTTSRows(df)

            computePredictedText(df)
            determineKeyStrokes(df)
            countKeystrokes(df)
            altKeystrokes(df)
            lexiconText(df)
            theoreticalKeyStrokes(df)

            all_data2.append(df)
            buffer = StringIO()
            df.to_html(buffer, classes='table table-bordered h6 datatables no-wrap', index=False)
            data_html2 = buffer.getvalue()
            buffer.close()
            all_data_html2.append(data_html2)

            finalOutputforMultipleFiles(df, filename, fileCounts)
            compile_csv()
            compute_metrics()
            # To filter the data and remove the Non TTS Rows
            filteredData = removeNonTTSRows(df)
            all_data3.append(filteredData)
            buffer = StringIO()
            filteredData.to_html(buffer, classes='table table-bordered h6 datatables no-wrap', index=False)
            data_html3 = buffer.getvalue()
            buffer.close()
            all_data_html3.append(data_html3)

            enumerated_data.append((len(enumerated_data) + 1, data_html))  # Use len for counting
            enumerated_data2.append((len(enumerated_data2) + 1, data_html2))  # Use len for counting
            enumerated_data3.append((len(enumerated_data3) + 1, data_html3))  # Use len for counting

            # Load the computed metrics from the CSV file
            metrics_path = os.path.join(upload_folder, "computed_metrics.csv")
            metrics_df = pd.read_csv(metrics_path)

            # Convert metrics DataFrame to a dictionary for easy rendering in HTML
            metrics_dict = dict(zip(metrics_df["Metrics"], metrics_df["Value"]))

            # print(all_data)
            #
            barchart_keystroke = scatterchart_keystrokes(df)
            words_per_minutes = words_per_minute(df)
            wordc_ch_s = wordc_ch(df)
            actualKeystrokes = actualKeystroke(df)
            char_per_sec = char_per_se(df)

            firstGraphInsight = sumy('A scatter plot titled "Keystroke Saved over Keystroke" likely analyzes how a new method saves time on tasks compared to the original method.')
            secondGraphInsight = sumy('Words per minute over Timestamp suggests a graph tracking typing speed (words per minute) as it changes over time (timestamp).')
            thirdGraphInsight = sumy('This kind of scatter plot likely shows the relationship between typing speed and the total written content.')
            fourthGraphInsight = sumy('This graph is a scatter plot of typing speeds, showing how many instances that is typed at each character per second speed.')

        else:
            # Handle files with different extensions (optional)
            pass

    # After processing all files, return data for template
    if all_data:
        # Combine dataframes if needed (e.g., using pd.concat)
        combined_df = pd.concat(all_data2, ignore_index=False)  # Combine dataframes
        # # Perform operations on the combined_df dataframe

        finalresult_path = os.path.join(upload_folder, "final_result.csv")
        finalresult_df = pd.read_csv(finalresult_path)
        final = finalresult_df.to_html(classes='table table-bordered h6 datatables no-wrap',
                                       index=False)
        # upload_files()
        today = datetime.now().strftime("%Y")
        return render_template('analyze.html', today=today, data_html=enumerated_data, data_html2=enumerated_data2,
                               data_html3=enumerated_data3, combined_df=combined_df,
                               finalresult=final,
                               metrics=metrics_dict,
                               barchart_keystroke=barchart_keystroke,
                               words_per_minutes=words_per_minutes,
                               wordc_ch_s=wordc_ch_s,
                               actualKeystrokes=actualKeystrokes,
                               char_per_sec=char_per_sec,
                               firstGraphInsight = firstGraphInsight,
                               secondGraphInsight = secondGraphInsight,
                               thirdGraphInsight = thirdGraphInsight,
                               fourthGraphInsight = fourthGraphInsight
                               )
    else:
        return render_template('home.html', message='No CSV files uploaded')


def scatterchart_keystrokes(data):
    fig = px.scatter(data, x="Keystrokes", y="Keystrokes Saved", title='Keystrokes vs Keystrokes Saved',
                     height=700, width=1000, color='Keystrokes')
    return fig.to_json()


def wordc_ch(df):
    wordc_ch_s = px.scatter(df, x="Characters per Second", y="Word Count",
                            title='Words Count vs Characters per second', height=600, width=1000)
    return wordc_ch_s.to_json()


def words_per_minute(df):
    words_per_minutes = px.scatter(df, x="Timestamp", y="Words per minute",
                                   title="Words per Minute Distribution", height=600, width=1000
                                   )
    return words_per_minutes.to_json()


def actualKeystroke(df):
    actualKeystrokes = px.box(df, x="Actual Keystrokes", title="BoxPlot for Actual Keystrokes", height=600,
                              width=1000)
    return actualKeystrokes.to_json()


def char_per_se(df):
    char_per_sec = px.scatter(df["Characters per Second"], title="Distribution of Characters per Second",
                              height=600, width=1000)
    char_per_sec.update_xaxes(title="Count")
    char_per_sec.update_yaxes(title="Characters per Second")
    return char_per_sec.to_json()


def sumy(text):
    tokenizer = Tokenizer(language="english")
    parser = PlaintextParser.from_string(text, tokenizer)
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count=2)
    for sentence in summary:
        return sentence


@app.route("/")
def home():
    return render_template("home.html")


# Example usage in your route handler (assuming Flask)
@app.route('/upload', methods=["GET", "POST"])
def upload():
    return uploadFile(request)


if __name__ == "__main__":
    app.run(debug=True)
