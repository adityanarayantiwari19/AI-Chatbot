"""
evaluate.py
===========
Measures how accurately the chatbot recognises the intent of a message.

Run it with:   python evaluate.py

It prints an accuracy summary and per-intent precision / recall, and saves
every result to evaluation_results.csv (open it in Excel for your report).

Note: these test sentences were written by the project author, so the score
shows that the chatbot behaves as designed on a wide range of phrasings,
typos and unknown questions. It is not a benchmark on unseen real-world data.
"""

import csv
from pathlib import Path

from chatbot import chat

# (message, expected intent). 'fallback' means the bot should admit it doesn't know.
TEST_CASES = [
    # --- small talk ---
    ("hello", "greeting"),
    ("Hi", "greeting"),
    ("hey there", "greeting"),
    ("good morning", "greeting"),
    ("Good evening!", "greeting"),
    ("hii", "greeting"),
    ("namaste", "greeting"),
    ("hello bot", "greeting"),
    ("how are you", "how_are_you"),
    ("how are you doing today", "how_are_you"),
    ("what's up", "how_are_you"),
    ("bye", "goodbye"),
    ("goodbye", "goodbye"),
    ("see you later", "goodbye"),
    ("ok bye", "goodbye"),
    ("quit", "goodbye"),
    ("good night", "goodbye"),
    ("thanks", "thanks"),
    ("thank you", "thanks"),
    ("thank you so much", "thanks"),
    ("thanks a lot", "thanks"),
    ("who are you", "bot_identity"),
    ("what is your name", "bot_identity"),
    ("introduce yourself", "bot_identity"),
    ("who created you", "bot_identity"),
    ("tell me about yourself", "bot_identity"),
    ("what can you do", "capabilities"),
    ("help", "capabilities"),
    ("how can you help me", "capabilities"),
    ("what topics do you know", "capabilities"),
    # --- knowledge topics ---
    ("what is ai", "artificial_intelligence"),
    ("What is artificial intelligence?", "artificial_intelligence"),
    ("explain ai", "artificial_intelligence"),
    ("define artificial intelligence", "artificial_intelligence"),
    ("tell me about AI", "artificial_intelligence"),
    ("WHAT IS AI", "artificial_intelligence"),
    ("what is artifical inteligence", "artificial_intelligence"),
    ("what is machine learning", "machine_learning"),
    ("explain machine learning", "machine_learning"),
    ("machine learning", "machine_learning"),
    ("what is ml", "machine_learning"),
    ("what is machne learnig", "machine_learning"),
    ("how does machine learning work", "machine_learning"),
    ("what is deep learning", "deep_learning"),
    ("deep learning", "deep_learning"),
    ("explain deep learning in detail", "deep_learning"),
    ("define deep learnig", "deep_learning"),
    ("what is a neural network", "neural_network"),
    ("neural networks", "neural_network"),
    ("explain neural netwrk", "neural_network"),
    ("how do neural networks work", "neural_network"),
    ("what is nlp", "nlp"),
    ("natural language processing", "nlp"),
    ("explain nlp", "nlp"),
    ("explain in detail about nlp", "nlp"),
    ("how does nlp work", "nlp"),
    ("what is nlp used for", "nlp"),
    ("tell me about natural language processing", "nlp"),
    ("what is computer vision", "computer_vision"),
    ("computer vision", "computer_vision"),
    ("explain computer vision in detail", "computer_vision"),
    ("what is python", "python"),
    ("python programming", "python"),
    ("explain python", "python"),
    ("what is pyhton", "python"),
    ("why is python popular", "python"),
    ("what is a chatbot", "chatbot"),
    ("chatbots", "chatbot"),
    ("how do chatbots work", "chatbot"),
    ("explain chatbot in detail", "chatbot"),
    ("types of chatbots", "chatbot"),
    ("what is tokenization", "tokenization"),
    ("explain tokenization", "tokenization"),
    ("what is a token", "tokenization"),
    ("tokenization in detail", "tokenization"),
    ("what is tokanization", "tokenization"),
    ("what is stemming", "stemming"),
    ("explain stemming", "stemming"),
    ("porter stemmer", "stemming"),
    ("what is lemmatization", "stemming"),
    ("what is sentiment analysis", "sentiment_analysis"),
    ("sentiment analysis", "sentiment_analysis"),
    ("explain sentiment analysis in detail", "sentiment_analysis"),
    ("how do you work", "about_chatbot"),
    ("how does this chatbot work", "about_chatbot"),
    ("what technology do you use", "about_chatbot"),
    ("how were you built", "about_chatbot"),
    ("about this project", "about_chatbot"),
    # --- things the chatbot should NOT pretend to know ---
    ("tell me about cooking", "fallback"),
    ("what is the weather today", "fallback"),
    ("who won the cricket match", "fallback"),
    ("what is the capital of france", "fallback"),
    ("recommend a movie", "fallback"),
    ("how to cook pasta", "fallback"),
    ("asdfgh", "fallback"),
    ("what is the price of gold", "fallback"),
    ("can you book a flight", "fallback"),
    ("play some music", "fallback"),
]


def run_evaluation(cases=TEST_CASES):
    """Run every test case through the chatbot and return a list of result rows."""
    rows = []
    for message, expected in cases:
        result = chat(message)
        rows.append({
            "message": message,
            "expected": expected,
            "predicted": result["intent"],
            "confidence_percent": round(result["confidence"] * 100, 1),
            "correct": result["intent"] == expected,
        })
    return rows


def accuracy(rows):
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def per_intent_metrics(rows):
    """Precision, recall and F1 for every intent."""
    labels = sorted({r["expected"] for r in rows} | {r["predicted"] for r in rows})
    metrics = []
    for label in labels:
        tp = sum(1 for r in rows if r["expected"] == label and r["predicted"] == label)
        fp = sum(1 for r in rows if r["expected"] != label and r["predicted"] == label)
        fn = sum(1 for r in rows if r["expected"] == label and r["predicted"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append((label, tp + fn, precision, recall, f1))
    return metrics


def save_csv(rows, path="evaluation_results.csv"):
    path = Path(__file__).resolve().parent / path
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    rows = run_evaluation()
    correct = sum(r["correct"] for r in rows)
    print(f"\nTest cases : {len(rows)}")
    print(f"Correct    : {correct}")
    print(f"Accuracy   : {accuracy(rows):.1%}\n")

    print(f"{'Intent':<26}{'Cases':>6}{'Precision':>11}{'Recall':>9}{'F1':>7}")
    print("-" * 59)
    for label, support, precision, recall, f1 in per_intent_metrics(rows):
        print(f"{label:<26}{support:>6}{precision:>11.2f}{recall:>9.2f}{f1:>7.2f}")

    failures = [r for r in rows if not r["correct"]]
    print(f"\nIncorrect predictions: {len(failures)}")
    for r in failures:
        print(f"  '{r['message']}'  expected={r['expected']}  got={r['predicted']}"
              f"  ({r['confidence_percent']}%)")

    print(f"\nFull results saved to: {save_csv(rows)}")


if __name__ == "__main__":
    main()