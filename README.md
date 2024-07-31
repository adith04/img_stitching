# EECS486Project

Welcome EECS 486 Staff!

## Motivation
Football podcasts can get quite long, and can cover many topics in one episode. Michigan fans may only want to hear talk about specific topics, such as position groups or players. Our goal is to create these segments automatically, so people can know where to look for the topics they want. 

## Project Pipeline
There are 3 main components of the project that work in sequence to process a podcast episode - transcription, labeling, and merging the results. The podcast is first transcribed using transcribe_google.py, which sends the mp3 file to google's text-to-speech AI. This API returns a list of sentences, as well as timestamps for the beginning and end of each sentence. Then, in merge_classifications.py, each sentence is fed into our pre-trained, fine-tuned TinyBert instance, which returns a classification for the sentence using the functions in classify_sentences.py. This model was trained in a Google Colab notebook (linked further below), and the final model was downloaded to be used for evaluation on a local machine. To merge the results in merge_classification.py, we combine the sentences into groups of 10, find the most common topic classification across those 10 sentences, and label the entire podcast segment with that label. This process is repeated for all segment groups in the podcast. Consecutive segments with the same topic are merged into one larger segment, and the final list of segment classifications and corresponding timestamps is output to a .txt file. 

## Repository Structure:

This repository contains a variety of folders and files. Below is a high level explanation of what all these folder contain and their purpose towards the project. 

deprecated/ - Contains scripts ultimately not used in the final project

evaluation/ - Contains scripts and manually verified podcast segment data

in_the_trenches_episode_... - These folders were output by transcribe_google.py and contain audio transcription files used to effectively segment a podcast episode by topic 

model_directory/ - Contains configuration information related to our custom langugage model

training/ - All of the project's article training data and related scripts are here

.gitignore - Ensures credentials and virtual environment are not pushed to GitHub

classify_sentences.py - Uses custom language model to classify a sentence into predetermined categories

example_sentences.txt - Sample input file for classify_sentences.py

merge_classification.py - Uses sentence information of a .mp3 file and the custom language model's classifications of those sentences to create podcast topic segments

README.md - File containing project information and instructions for running on a local machine

requirements.txt - Contains all Python packages required 

transcribe_google.py - Transcribes .mp3 file into sentences

transcribe_podcast_options.txt - Lists all .mp3 files that transcribe_google.py can operate on since they are stored in a Google Cloud bucket associated with 'credentials.json'

## Data Sources

The Google Drive link below contains all of the original .mp3 files that our project is based on. 

Podcast Google Drive Link: https://drive.google.com/drive/folders/19mEX01TQiWGeWlyIJ3WRptGCBnGYmI4-?usp=sharing

## Setting Up the Project:

1) Clone this GitHub repository to your local machine.

```bash
git clone git@github.com:rchandra20/eecs486-final-project.git
```

2) Install all the necessary Python packages needed to run this project via pip3

```bash
pip3 install -r requirements.txt
```

## Running the Project:

There are a couple key Python programs that make up the project pipeline. This guide takes you step by step through it along with relevant explanations and instructions for each script.  

### training/crawl.py

This script is used to crawl the URLs of football related articles found by the team for model training purposes. Al of these URLs are contained in training/training_articles_urls.txt. This file is a list of links labeled with their corresponding topic, which then gets crawled and processed into the format needed to train our model. 

Navigate to the training/ folder and run this script to crawl the training URLs:
```bash
cd training
python3 crawl.py training_article_urls.txt
```

The raw training data for this run should be shown in training/training_data_test/ 

### clean_training_data.py

This script simply takes the raw training data contained in training/training_data_raw/ and outputs the cleaned version into training/training_data_processed/

```bash
python3 clean_training_data.py
```

### training/naivebayes.py

This script runs the Naive Bayes algorithm on the processed training data. We wanted to use this as an evaluation metric, to compare our custom trained model with a more naive approach to the same problem. 

Navigate to the naivebayes.py script and run it on the cleaned training data folder as an input as shown below:
```bash
python3 naivebayes.py training_data_processed/
```
In the terminal you can see the number of training files analyzed as well as the accuracy of the system using Naive Bayes.

### classify_sentences.py

Using the training data contained in training/training_data_processed/ the team was able to train a model.

Google Colab Link: https://colab.research.google.com/drive/1q6gOAOhHm3mVFlxmRt_nCp3P5aD8Te-3?usp=sharing

Above is the Google Colab that contains scripts used to train the language model that classify_sentences.py uses. The Colab can be run by
running every cell in sequence and uploading the appropriate training files (articles) when prompted. The downloaded model is placed in the model_directory/ folder for local inference by classify_sentences.py.

Navigate to the root of this repository and run classify_sentences.py on a sample test file created by the team:
```bash
cd ..
python3 classify_sentences.py example_sentences.txt
```

As you can see, each of the sentences in example_sentences.txt is classified by the custom trained language model. Take a look at the output file:
```bash
cat classified_sentences.txt
```

### transcribe_google.py

This script takes in a .mp3 file name as input and produces a audio transcription files containing sentences. The Google Cloud credential file 'credentials.json' needs to be at the root of the repository for the audio transcription files to be properly generated. This script takes in 1 command line argument, the .mp3 file to be transcribed. The actual .mp3 file is located in a Google Cloud bucket associated with the 'credentials.json'. All the .mp3 files that this script can transcribe are contained in 'transcribe_podcast_options.txt'.

Verify that the credentials.json is contained in the repository.

```bash
cat credentials.json
```

The audio transcription usually takes ~20 minutes for a podcast episode. Thus, the team created a small snippet of an episode called 'podcast_snippet.mp3' that is around 4 minutes long so that the transcription process can be faster for you. It should take ~5 minutes.

```bash
python3 transcribe_google.py podcast_snippet.mp3
```

Now take a look at the podcast_snippet/ folder that was generated. Inside are all the sentence transcription files and sentence timestamp information generated by Google Cloud.

```bash
cd podcast_snippet
ls 
cat sentences_timestamps.txt 
cat sentences_transcribed_raw.txt
```

### merge_classification.py

This script takes in the audio transcription files generated by Google Cloud for a specific podcast episode and splits up the podcast into segmented categories. Simply run the script with the podcast episode name as the argument. This name must match that of a podcast contained in 'transcribe_podcast_options.txt'. Let's do the podcast episode snippet 'podcast_snippet.mp3' and 'in_the_trenches_episode_428.mp3' as examples. 

```bash
cd ..
python3 merge_classification.py podcast_snippet.mp3
python3 merge_classification.py in_the_trenches_episode_428.mp3
```

Take a look at the output. It showcases how a podcast changes topic as time goes on. A timeline plot generated by matplotlib should also be shown on your machine that visualize these topic changes. 

```bash
cat in_the_trenches_episode_428_segments.txt
```

### evaluate.py

This script was used for efficient calculation of the IoU metrics for our manually reviewed segment evaluations contained in the evaluation/ folder. 

Navigate to the evaluation/ folder and run it on a labeled podcast segments file manually verified by a team member. 
```bash
cd evaluation
python3 evaluate.py in_the_trenches_episode_428_segments.txt
```

The results of this evaluation should be shown in the terminal. These metrics are included in a table in our project report. 

## Contact
For any questions/concerns about this project please contact the team at the following email addresses:
bbrdak@umich.edu
rajchan@umich.edu
radithya@umich.edu
rohanxg@umich.edu
ncurdo@umich.edu
