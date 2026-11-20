import json
import re
import sys
from pathlib import Path

from nltk.stem import PorterStemmer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Sample texts 
SAMPLE_TEXTS = {
    "text1": "Friends, Romans, countrymen, lend me your ears;",
    "text2": (
        "John's new U.S.A. laptop is a state-of-the-art device. "
        "The price is high, but the quality is amazing!"
    ),
    "text3": (
        "The authorization for this authorized project is pending. "
        "He authorizes the payments regularly."
    ),
}

#stop-word list

STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "from",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "me",
        "my",
        "your",
        "you",
        "we",
        "our",
        "i",
        "not",
        "no",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
    }
)

#Ordinary words
TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:\.[A-Za-z]+)+\.?"
    r"|[A-Za-z]+(?:-[A-Za-z]+)+"
    r"|[A-Za-z]+(?:['\u2019]s)?"
)

_STEMMER = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """Split raw text into tokens according to the project rules."""
    raw_tokens = TOKEN_PATTERN.findall(text)
    tokens = []
    for token in raw_tokens:
        if len(token) > 2 and token[-2:] in ("'s", "\u2019s"):
            token = token[:-2]
        tokens.append(token)
    return tokens


def normalize(tokens: list[str]) -> list[str]:
    """Lowercase tokens and unify abbreviations; keep internal hyphens."""
    normalized = []
    for token in tokens:
        value = token.lower()
        if "." in value:
            value = value.replace(".", "")
        normalized.append(value)
    return normalized


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove English stop words using the fixed STOP_WORDS list."""
    return [token for token in tokens if token not in STOP_WORDS]


def stemming(tokens: list[str]) -> list[str]:
    """Stem tokens with Porter Stemmer; keep order and duplicates."""
    return [_STEMMER.stem(token) for token in tokens]


def preprocess(text: str) -> list[str]:
    """
    Run the full preprocessing pipeline in the required order:
    Tokenization -> Normalization -> Stop Word Removal -> Stemming
    The result is a list (not a set) so order and duplicates are kept.
    """
    tokens = tokenize(text)
    tokens = normalize(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens

GOLD_STANDARD = {

    "text1": ["friend", "roman", "countrymen", "lend", "ear"],
    "text2": [
        "john",
        "new",
        "usa",
        "laptop",
        "state-of-the-art",
        "devic",
        "price",
        "high",
        "qualiti",
        "amaz",
    ],
    "text3": [
        "author",
        "author",
        "project",
        "pend",
        "author",
        "payment",
        "regularli",
    ],
}


def evaluate(output: list[str], gold: list[str]) -> dict:
    """
    Compute Precision, Recall, Extra Terms, and Missing Terms.
    Metrics use sets so duplicates do not affect the scores.
    """
    output_set = set(output)
    gold_set = set(gold)
    intersection = output_set & gold_set

    precision = (len(intersection) / len(output_set)) if output_set else 0.0
    recall = (len(intersection) / len(gold_set)) if gold_set else 0.0

    extra_terms = sorted(output_set - gold_set)
    missing_terms = sorted(gold_set - output_set)

    return {
        "intersection": sorted(intersection),
        "precision": precision,
        "recall": recall,
        "extra_terms": extra_terms,
        "missing_terms": missing_terms,
    }


def _print_section(title: str, value) -> None:
    """Print one labeled output section."""
    print(f"{title}: {value}")


def run_pipeline_and_report() -> dict[str, list[str]]:
    """Run the pipeline on all three texts, print results, and prepare JSON output."""
    final_outputs: dict[str, list[str]] = {}

    for text_id, raw_text in SAMPLE_TEXTS.items():
        print("=" * 70)
        print(f"Results for {text_id}")
        print("=" * 70)

        tokens = tokenize(raw_text)
        normalized = normalize(tokens)
        without_stops = remove_stopwords(normalized)
        stemmed = stemming(without_stops)

        gold = GOLD_STANDARD[text_id]
        metrics = evaluate(stemmed, gold)

        _print_section("Input text", raw_text)
        _print_section("Tokenization output", tokens)
        _print_section("Normalization output", normalized)
        _print_section("Stop Word Removal output", without_stops)
        _print_section("Final Stemming output", stemmed)
        _print_section("Gold Standard", gold)
        _print_section("Intersection", metrics["intersection"])
        _print_section("Precision", metrics["precision"])
        _print_section("Recall", metrics["recall"])
        _print_section("Extra Terms", metrics["extra_terms"])
        _print_section("Missing Terms", metrics["missing_terms"])
        print()

        final_outputs[text_id] = stemmed

    return final_outputs


def save_outputs(outputs: dict[str, list[str]], path: Path | None = None) -> Path:
    """Save the final processed output to outputs.json."""
    output_path = path or Path(__file__).resolve().parent / "outputs.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(outputs, file, ensure_ascii=False, indent=4)
    return output_path


def main() -> None:
    outputs = run_pipeline_and_report()
    saved_path = save_outputs(outputs)
    print(f"Final output saved to: {saved_path}")


if __name__ == "__main__":
    main()
