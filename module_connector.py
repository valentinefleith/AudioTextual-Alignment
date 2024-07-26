import os
import glob
import json
import bert_embedding_extractor as be

def read_transcripts(folder_path: str):
    """
    Read all text files from the specified folder and return their content as a list of strings.

    Parameters:
    folder_path (str): The path to the folder containing transcript files.

    Returns:
    List[str]: A list of transcript texts.
    """
    transcripts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                transcripts.append(file.read())
    return transcripts

def read_words(json_file: str):
    """
    Read the JSON file and return a list of words.

    Parameters:
    json_file (str): The path to the JSON file containing words.

    Returns:
    List[str]: A list of words to contextualize.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return [x["word"] for x in data['high']]
    # words = list(data["word"].values())
    # print(words)
    # return words

def read_indices(json_file: str):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return [x["index"] for x in data["high"]]

if __name__ == "__main__":
    # Define paths
    transcripts_folder = './data_sample/transcripts/'
    words_json = glob.glob('./data_sample/words/*.json')

    # Read transcripts and words
    transcripts = read_transcripts(transcripts_folder)
    # for json_file in words_json:

    for json_file, transcript in zip(words_json, transcripts):
        print(f"\nCurrently dealing {json_file}")
        words_to_contextualize = read_words(json_file)
        indices_to_contextualize = read_indices(json_file)
        embeddings = be.get_contextualized_bert_embeddings(transcript, indices_to_contextualize)

        # Print embeddings for each word
        for word, embedding in embeddings:
            print(f"Word: {word}, Embedding: {embedding[:5]}...")  # Print first 5 values for brevity
