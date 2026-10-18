"""
app.py
======
Streamlit web interface for the chatbot.

Run it with:   python -m streamlit run app.py
"""

import streamlit as st

from chatbot import KB, chat

st.set_page_config(page_title=KB["bot_name"], page_icon="\U0001F916", layout="centered")

WELCOME = (
    "\U0001F44B Hello! I'm **AI Chatbot Assistant**.\n\n"
    "Ask me about AI, machine learning, deep learning, NLP, Python and more. "
    "I give a short answer by default, and a longer one if you say "
    "*“explain in detail”*."
)

EXAMPLES = [
    "Hello",
    "What is AI?",
    "Explain machine learning in detail",
    "What is NLP?",
    "What is tokenization?",
    "How do you work?",
]


def new_conversation():
    """Reset the chat history and the remembered topic."""
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
    st.session_state.last_topic = None


def render_analysis(analysis):
    """Show how the chatbot processed a message (useful in a demo or viva)."""
    with st.expander("\U0001F50D NLP analysis of this reply"):
        st.markdown(f"**1. Tokens:** `{analysis['tokens']}`")
        st.markdown(f"**2. Stems:** `{analysis['stems']}`")
        st.markdown(f"**3. Detail level requested:** `{analysis['style']}`")
        st.markdown(
            f"**4. Matched intent:** `{analysis['intent']}` "
            f"({analysis['confidence']:.0%} confidence)"
        )
        if analysis["matches"]:
            st.markdown("**Top candidate intents:**")
            st.table({
                "Intent": [m["intent"] for m in analysis["matches"]],
                "Score (%)": [m["score"] for m in analysis["matches"]],
            })


if "messages" not in st.session_state:
    new_conversation()

# ----------------------------------------------------------------- sidebar
clicked_example = None
with st.sidebar:
    st.header("About this project")
    st.write(
        "A rule-based chatbot that uses tokenization, stemming, weighted "
        "pattern matching and typo tolerance to understand your questions."
    )
    show_analysis = st.toggle(
        "Show NLP analysis",
        value=False,
        help="Display the tokens, stems and match scores behind each reply.",
    )

    st.subheader("Try asking")
    for example in EXAMPLES:
        if st.button(example, key=f"example_{example}"):
            clicked_example = example

    st.divider()
    if st.button("\U0001F5D1️ Clear conversation"):
        new_conversation()
        st.rerun()

# -------------------------------------------------------------------- main
st.title("\U0001F916 " + KB["bot_name"])
st.caption("Rule-Based AI Chatbot using Natural Language Processing")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_analysis and message.get("analysis"):
            render_analysis(message["analysis"])

prompt = st.chat_input("Type your message here...") or clicked_example

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = chat(prompt, st.session_state.last_topic)
    st.session_state.last_topic = result["topic"]

    analysis = {
        "tokens": result["tokens"],
        "stems": result["stems"],
        "style": result["style"],
        "intent": result["intent"],
        "confidence": result["confidence"],
        "matches": result["matches"],
    }
    st.session_state.messages.append(
        {"role": "assistant", "content": result["reply"], "analysis": analysis}
    )
    with st.chat_message("assistant"):
        st.markdown(result["reply"])
        if show_analysis:
            render_analysis(analysis)