# Rule-Based AI Chatbot Using Natural Language Processing

A web-based chatbot built with **Python, NLTK and Streamlit** as an internship minor project
(*Simple Chatbot* from the AI minor-project list).

It understands questions using NLP techniques (tokenization, stemming, weighted pattern
matching and typo tolerance), answers from a JSON knowledge base, and can give either a
**short** or a **detailed** explanation depending on how you ask.

## Features

- Greetings, small talk and 12 knowledge topics (AI, machine learning, deep learning,
  neural networks, NLP, computer vision, Python, chatbots, tokenization, stemming,
  sentiment analysis, and how this chatbot works)
- **Short answers by default; detailed answers** with *"explain in detail"*, *"elaborate"*,
  *"in depth"*, *"tell me more"* and similar phrases
- **Conversation memory**: *"tell me more"* continues the previous topic
- **Typo tolerance**: *"what is machne learnig"* still works
- **Honest fallback**: if confidence is below 60%, the bot says it doesn't know instead of guessing
- Optional **NLP analysis panel** that shows tokens, stems and match scores for each reply
- 33 automated unit tests and an accuracy evaluation script
- No API key, no internet connection and no GPU needed

## Technology stack

| Part | Tool |
|---|---|
| Language | Python 3.9+ |
| NLP | NLTK (tokenizer and Porter stemmer) |
| Web interface | Streamlit |
| Knowledge base | JSON (`intents.json`) |
| Testing | `unittest` (built into Python) |

## Project structure

```
AI_Chatbot/
├── app.py               Streamlit web interface
├── chatbot.py           Chatbot logic (preprocessing, matching, answers)
├── intents.json         Knowledge base: patterns, short and detailed answers
├── test_chatbot.py      Automated unit tests
├── evaluate.py          Accuracy evaluation (precision, recall, F1)
├── requirements.txt     Python dependencies
└── README.md            This file
```

## Setup (Windows, VS Code, PowerShell)

Open the `AI_Chatbot` folder in VS Code, open a terminal (**Terminal > New Terminal**) and run:

```powershell
python -m pip install -r requirements.txt
```

If `python` is not recognised, use `py` instead of `python` in every command below.

## Run the chatbot

```powershell
python -m streamlit run app.py
```

Open the address shown in the terminal (normally http://localhost:8501).
Stop the server with **Ctrl + C**.

You can also chat in the terminal without the web page:

```powershell
python chatbot.py
```

## Run the tests

```powershell
python -m unittest -v
```

## Measure accuracy

```powershell
python evaluate.py
```

This prints overall accuracy plus precision, recall and F1 for every intent, and saves all
results to `evaluation_results.csv`, which you can open in Excel for your report.

The test sentences were written by the project author. The score therefore shows that the
chatbot handles many different phrasings, typos and unknown questions as designed. It is not
a benchmark on unseen real-world conversations.

## How it works

Every message goes through this pipeline:

1. **Normalisation**: lower-case the text and remove punctuation.
2. **Tokenization**: split the message into words with NLTK's `wordpunct_tokenize`.
3. **Stemming**: reduce each word to its root with the Porter stemmer
   (*learning, learned, learns* all become *learn*).
4. **Weighted pattern matching**: common filler words (*is, the, what*) count 0.2 and topic
   words (*nlp, python*) count 1.0. Each known pattern is scored by how much of it appears
   in the message. Topic words in the message that the pattern does not explain reduce the
   score. Small spelling mistakes are accepted using edit distance (Damerau-Levenshtein).
5. **Confidence check**: the best score must be at least 0.60, otherwise the bot uses a
   fallback reply.
6. **Answer selection**: a short answer by default, or a detailed answer if the message
   contains detail words such as *detail*, *elaborate* or *in depth*.
7. **Context memory**: the last topic is remembered so follow-ups such as *"tell me more"* work.

## Adding a new topic

You do not need to change any Python code. Add an entry to the `intents` list in `intents.json`:

```json
{
  "tag": "data_science",
  "title": "Data Science",
  "patterns": ["what is data science", "data science", "explain data science"],
  "short": "Data science is ...",
  "detailed": [
    "### What is Data Science?",
    "A longer explanation ...",
    "**Examples**\n- Example one\n- Example two"
  ],
  "related": ["machine_learning"]
}
```

Then run `python -m unittest` to check that the knowledge base is still valid.

## Limitations

- The chatbot only knows the topics in `intents.json`; it cannot generate new answers.
- It does not learn from conversations.
- It matches words, not meaning, so unusual wording or questions that mix several topics
  may get a fallback reply or the answer for only one of the topics.
- Word-by-word matching does not handle negation or sarcasm.

## Possible future work

- Train a machine-learning intent classifier (TF-IDF with Logistic Regression) on the patterns
- Add sentiment analysis of the user's messages
- Connect the chatbot to a larger knowledge base or an external API
- Add voice input and output