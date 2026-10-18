"""
chatbot.py
==========
Core logic of the Rule-Based AI Chatbot (there is no web code in this file).

How one message flows through the chatbot
-----------------------------------------
1. tokenize()       lower-case the text and split it into words
2. stemming         'learning' / 'learned' / 'learns'  ->  'learn'
3. rank_intents()   score the message against every known pattern
4. chat()           pick the best intent, then a short or a detailed answer

The knowledge base lives in intents.json, so new topics can be added
without touching this file.
"""

import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INTENTS_FILE = BASE_DIR / "intents.json"

CONFIDENCE_THRESHOLD = 0.60   # below this score the bot says "I don't know"
FILLER_WEIGHT = 0.2           # common words ('is', 'the') count less than topic words
CONTENT_WEIGHT = 1.0
UNMATCHED_PENALTY = 0.4       # penalty for topic words that match no known pattern
FUZZY_MIN_LENGTH = 4          # typo tolerance only for words of this length or more

# Very common words. They help a little, but they should not decide the topic.
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
    "what", "whats", "who", "how", "why", "which", "when", "where",
    "of", "to", "in", "on", "for", "with", "about", "and", "or",
    "me", "my", "i", "you", "your", "it", "its", "this", "that",
    "please", "can", "could", "would", "do", "does", "did",
    "tell", "explain", "define", "describe", "give", "show", "go",
    "meaning", "mean", "know", "want", "like", "some", "us", "we",
    "ok", "okay", "yes", "yeah", "sure", "now", "also", "just", "there",
}

# Words that tell the bot HOW MUCH detail the user wants.
DETAIL_WORDS = {
    "detail", "details", "detailed", "detailing", "depth", "elaborate",
    "elaboration", "thorough", "thoroughly", "comprehensive",
    "comprehensively", "extensive", "extensively", "deeper", "more",
    "longer", "fully", "full", "complete", "completely", "lengthy",
    "exhaustive",
}
DETAIL_PHRASES = ("in detail", "in depth", "tell me more", "go deeper",
                  "full explanation")
FUZZY_DETAIL_ROOTS = ("detail", "detailed", "elaborate", "thorough",
                      "comprehensive")
BRIEF_WORDS = {
    "brief", "briefly", "short", "shortly", "summary", "summarize",
    "summarise", "tldr", "quick", "quickly", "concise", "concisely",
    "simple", "simply", "oneline",
}

stemmer = PorterStemmer()


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
def load_knowledge_base(path=INTENTS_FILE):
    """Read intents.json and check that every intent is complete."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    for key in ("fallback_responses", "detail_without_topic_responses", "intents"):
        if key not in data:
            raise ValueError(f"intents.json is missing the '{key}' section")

    for intent in data["intents"]:
        tag = intent.get("tag", "<no tag>")
        if not intent.get("patterns"):
            raise ValueError(f"Intent '{tag}' has no patterns")
        is_topic = "short" in intent
        if is_topic and not intent.get("detailed"):
            raise ValueError(f"Topic '{tag}' needs a 'detailed' answer")
        if not is_topic and not intent.get("responses"):
            raise ValueError(f"Intent '{tag}' needs 'responses' or 'short'")
    return data


KB = load_knowledge_base()
INTENTS = KB["intents"]
INTENTS_BY_TAG = {intent["tag"]: intent for intent in INTENTS}
TOPIC_TITLES = [i["title"] for i in INTENTS if "short" in i]


# ---------------------------------------------------------------------------
# Step 1 and 2: text preprocessing
# ---------------------------------------------------------------------------
def tokenize(text):
    """Lower-case the text and split it into word tokens (punctuation removed)."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"(\w+)'s\b", r"\1", text)   # what's -> what
    text = text.replace("'", "")               # don't  -> dont
    return [t for t in wordpunct_tokenize(text) if re.search(r"\w", t)]


def preprocess_text(text):
    """Tokenize the text and stem every token."""
    return [stemmer.stem(token) for token in tokenize(text)]


def weighted_stems(tokens):
    """Return [(stem, weight, token), ...]. Filler words get a lower weight."""
    items = []
    for token in tokens:
        weight = FILLER_WEIGHT if token in STOPWORDS else CONTENT_WEIGHT
        items.append((stemmer.stem(token), weight, token))
    # A pattern made only of filler words (e.g. 'who are you') is still
    # meaningful, so its words all count fully.
    if items and all(w == FILLER_WEIGHT for _, w, _ in items):
        items = [(stem, CONTENT_WEIGHT, token) for stem, _, token in items]
    return items


# ---------------------------------------------------------------------------
# Typo tolerance
# ---------------------------------------------------------------------------
def edit_distance(a, b):
    """Damerau-Levenshtein distance: swapping two letters counts as 1 edit."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def words_match(a, b):
    """True if two stems are equal or differ only by a small typo."""
    if a == b:
        return True
    if min(len(a), len(b)) < FUZZY_MIN_LENGTH:
        return False
    if edit_distance(a, b) <= 1:
        return True
    return min(len(a), len(b)) >= 5 and SequenceMatcher(None, a, b).ratio() >= 0.83


# ---------------------------------------------------------------------------
# Detail level detection
# ---------------------------------------------------------------------------
def is_detail_word(token):
    if token in DETAIL_WORDS:
        return True
    return len(token) >= 6 and any(words_match(token, root) for root in FUZZY_DETAIL_ROOTS)


def is_style_word(token):
    return is_detail_word(token) or token in BRIEF_WORDS


def detect_style(tokens):
    """Return 'detailed', 'brief' or 'normal'."""
    text = " ".join(tokens)
    if any(phrase in text for phrase in DETAIL_PHRASES) or any(is_detail_word(t) for t in tokens):
        return "detailed"
    if any(t in BRIEF_WORDS for t in tokens):
        return "brief"
    return "normal"


# ---------------------------------------------------------------------------
# Step 3: intent matching
# ---------------------------------------------------------------------------
# Patterns never change, so they are preprocessed once at start-up.
for _intent in INTENTS:
    _intent["_compiled"] = [weighted_stems(tokenize(p)) for p in _intent["patterns"]]


def pattern_score(user_items, pattern_items):
    """
    Score (0 to 1) how well a message matches one known pattern.

    coverage  = how much of the pattern's weight is found in the message
    penalty   = reduced when the message has topic words the pattern lacks
    """
    total_pattern = sum(w for _, w, _ in pattern_items)
    matched_pattern = 0.0
    matched_content = 0.0
    used = set()

    for p_stem, p_weight, p_token in pattern_items:
        for i, (u_stem, u_weight, u_token) in enumerate(user_items):
            if i in used:
                continue
            if p_weight == CONTENT_WEIGHT:
                # Compare stems, and also the original words, because
                # stemming can turn two spellings of one word into
                # different stems ('artifical' vs 'artificial').
                found = words_match(u_stem, p_stem) or words_match(u_token, p_token)
            else:
                found = (u_stem == p_stem)
            if found:
                used.add(i)
                matched_pattern += p_weight
                if u_weight == CONTENT_WEIGHT:
                    matched_content += 1
                break

    coverage = matched_pattern / total_pattern if total_pattern else 0.0
    user_content = sum(1 for _, w, _ in user_items if w == CONTENT_WEIGHT)
    unmatched = 1 - matched_content / user_content if user_content else 0.0
    return coverage * (1 - UNMATCHED_PENALTY * unmatched), total_pattern


def rank_intents(user_items):
    """Return [(score, tag), ...] for every intent, best first."""
    ranked = []
    for intent in INTENTS:
        best = (0.0, 0.0)
        for pattern_items in intent["_compiled"]:
            score, weight = pattern_score(user_items, pattern_items)
            # On equal scores, prefer the longer (more specific) pattern.
            if (round(score, 4), weight) > (round(best[0], 4), best[1]):
                best = (score, weight)
        ranked.append((best[0], best[1], intent["tag"]))
    ranked.sort(reverse=True)
    return [(score, tag) for score, _, tag in ranked]


# ---------------------------------------------------------------------------
# Step 4: building the reply
# ---------------------------------------------------------------------------
def topic_reply(intent, detailed, show_hint=True):
    """Short or detailed answer for a knowledge topic."""
    if detailed:
        text = "\n\n".join(intent["detailed"])
        related = [INTENTS_BY_TAG[t]["title"] for t in intent.get("related", [])]
        if related:
            text += "\n\n**Related topics:** " + ", ".join(related)
        return text
    text = intent["short"]
    if show_hint:
        text += (f"\n\n\U0001F4A1 *Tip: say **“explain in detail”** "
                 f"to learn more about {intent['title']}.*")
    return text


def make_result(reply, intent, confidence, topic, tokens=None, style="normal", ranked=None):
    """Bundle the reply with the information needed for the NLP-analysis view."""
    tokens = tokens or []
    return {
        "reply": reply,
        "intent": intent,
        "confidence": confidence,
        "topic": topic,                       # remembered for follow-up questions
        "style": style,
        "tokens": tokens,
        "stems": [stemmer.stem(t) for t in tokens],
        "matches": [{"intent": tag, "score": round(score * 100, 1)}
                    for score, tag in (ranked or [])[:3]],
    }


def chat(user_input, last_topic=None):
    """
    Main function. Takes the user's message (and the topic of the previous
    answer) and returns a dictionary that contains the reply.
    """
    text = (user_input or "").strip()
    tokens = tokenize(text)

    if not tokens:
        return make_result("Please type a message, for example: *What is AI?*",
                           "empty", 0.0, last_topic)

    style = detect_style(tokens)
    topic_tokens = [t for t in tokens if not is_style_word(t)]
    content_words = [t for t in topic_tokens if t not in STOPWORDS]

    # Follow-up such as "tell me more" or "explain in detail" with no topic.
    if style != "normal" and not content_words:
        if last_topic:
            reply = topic_reply(INTENTS_BY_TAG[last_topic], style == "detailed",
                                show_hint=False)
            return make_result(reply, last_topic, 1.0, last_topic, tokens, style)
        return make_result(random.choice(KB["detail_without_topic_responses"])
                           + "\n\n**Topics I can explain:** " + ", ".join(TOPIC_TITLES),
                           "need_topic", 0.0, last_topic, tokens, style)

    ranked = rank_intents(weighted_stems(topic_tokens))
    best_score, best_tag = ranked[0]

    # Not confident enough: admit it instead of guessing.
    if best_score < CONFIDENCE_THRESHOLD:
        reply = (random.choice(KB["fallback_responses"])
                 + "\n\n**Topics I can explain:** " + ", ".join(TOPIC_TITLES))
        return make_result(reply, "fallback", best_score, last_topic, tokens, style, ranked)

    intent = INTENTS_BY_TAG[best_tag]
    if "short" in intent:                               # a knowledge topic
        reply = topic_reply(intent, style == "detailed", show_hint=(style == "normal"))
        new_topic = best_tag
    else:                                               # small talk
        reply = random.choice(intent["responses"]).replace("{topics}", ", ".join(TOPIC_TITLES))
        new_topic = last_topic
    return make_result(reply, best_tag, best_score, new_topic, tokens, style, ranked)


def get_response(user_input, last_topic=None):
    """Convenience wrapper that returns only the reply text."""
    return chat(user_input, last_topic)["reply"]


# ---------------------------------------------------------------------------
# Optional: run the chatbot in the terminal with `python chatbot.py`
# ---------------------------------------------------------------------------
def main():
    print(f"{KB['bot_name']} (type 'quit' to leave)\n")
    last_topic = None
    while True:
        try:
            message = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        result = chat(message, last_topic)
        last_topic = result["topic"]
        print(f"\nBot: {result['reply']}\n")
        if result["intent"] == "goodbye":
            break


if __name__ == "__main__":
    main()