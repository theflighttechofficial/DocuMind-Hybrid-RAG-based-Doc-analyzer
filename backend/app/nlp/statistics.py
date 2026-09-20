"""Real document statistics: counts, vocabulary, readability."""
import re
from nltk.tokenize import word_tokenize, sent_tokenize

from app.nlp.nltk_setup import ensure_nltk_data

ensure_nltk_data()


def compute_statistics(full_text: str) -> dict:
    try:
        sentences = sent_tokenize(full_text)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", full_text)

    try:
        words = [w for w in word_tokenize(full_text) if any(c.isalnum() for c in w)]
    except Exception:
        words = re.findall(r"\b\w+\b", full_text)

    paragraphs = [p for p in re.split(r"\n\s*\n", full_text) if p.strip()]

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    vocab = {w.lower() for w in words}
    vocab_size = len(vocab)
    unique_word_ratio = round(vocab_size / word_count, 4) if word_count else 0.0
    avg_sentence_len = round(word_count / sentence_count, 2)

    # Flesch Reading Ease (approximate syllable counting)
    def count_syllables(word: str) -> int:
        word = word.lower()
        vowels = "aeiouy"
        count, prev_was_vowel = 0, False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    total_syllables = sum(count_syllables(w) for w in words) if words else 0
    if word_count and sentence_count:
        flesch = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (total_syllables / word_count)
        flesch = round(max(0.0, min(100.0, flesch)), 1)
    else:
        flesch = 0.0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": max(len(paragraphs), 1),
        "avg_sentence_length": avg_sentence_len,
        "vocabulary_size": vocab_size,
        "unique_word_ratio": unique_word_ratio,
        "flesch_reading_ease": flesch,
    }
