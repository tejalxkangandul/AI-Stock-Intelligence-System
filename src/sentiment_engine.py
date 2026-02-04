from textblob import TextBlob
import nltk

# Self-Healing: Check if the required dictionary is present
try:
    TextBlob("test").sentiment
except Exception:
    import subprocess
    print("📥 Downloading sentiment data...")
    subprocess.run(["python", "-m", "textblob.download_corpora", "lite"])


def get_headline_sentiment(text):
    """
    Returns a color and a score for a headline.
    Polarity > 0.1 = Positive (Green)
    Polarity < -0.1 = Negative (Red)
    Otherwise = Neutral (White)
    """
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity

    if score > 0.1:
        return "🟢", score
    elif score < -0.1:
        return "🔴", score
    else:
        return "⚪", score
