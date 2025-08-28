# youtube-sentiment-analysis

## Project Overview
1. Get your comments from the Youtube trailer. One option would manually getting the comments, get them directly from the API, you can save them in a file and
load it in your colab.

2.Setup your colab to do the job for you, you will need to install the required libraries.

We have a huge number of comments from YouTube for a latest trailer from a worldwide production house, you as an AI
service provider are supposed to analyse all the comments on that trailer, get the sentiment and the score, and give
a consolidated report for that trailer about how it might perform on the box office.

## Dataset
- YouTube comments (collected via API).
- Sample data file: `snowwhite.xlsx`

## Tech Stack
- **Python**, **PyTorch**
- **NLTK**, **VADER**
- **Transformers (HuggingFace)**
- **Matplotlib**, **WordCloud**, **Pandas**

App Script code to get comments for any YouTube video:
https://script.google.com/u/0/home/projects/1zggWmt60gNVbFEvg_V9tE2iXnOcH7NW_eyd-Azh4QWoMrkCiV-yLmT6/edit
  
1.Do all the necessary imports
2.create a function for removing stop words
3.create a function to calculate the sentiment score and the sentiment(positive/negative)
4.Loop through the Comments that you will get from your input excel file
4.5 Seggreate the words into positive and negative, so you can make a word cl
5.Calculate all the sentiments in loop and return only one final result

## Results
- Each comment is labeled **Positive / Negative / Neutral**.
- Majority sentiment distribution shows how audiences are reacting.
- Example use case: Predicting trailer success and box-office impact.

Future Improvements
Multilingual Support – Detect and translate non-English comments into English before analysis.
Advanced Models – Experiment with LLM-based sentiment classifiers (e.g., GPT fine-tuning).
Trend Analysis – Track sentiment changes over time (e.g., before/after movie release).
Aspect-Based Sentiment – Identify which aspects (actors, music, VFX, story) audiences mention positively or negatively.
Deployment – Build a Flask/Streamlit web app to analyze new trailer comments in real time.
