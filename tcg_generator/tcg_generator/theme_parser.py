import json
import re
from pathlib import Path

from loguru import logger

from tcg_generator.progress_tracker import track_progress
from tcg_generator.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

THEME_REGEX_MAP = {
    "Flying": r"Flying"
}

PARSED_CARD_THEMES: dict[str, set[str]] = dict()

def parse_dataset(entry):
    card_name = entry.get("name", "")
    oracle_text = entry.get("oracle_text", "")

    for [theme, regex] in THEME_REGEX_MAP.items():
        match = re.search(regex, oracle_text)
        if match is not None:
            PARSED_CARD_THEMES.setdefault(card_name, set())
            PARSED_CARD_THEMES[card_name].add(theme)

def main(
    input_path: Path = RAW_DATA_DIR / "oracle-cards-20250127220801.json",
    output_path: Path = PROCESSED_DATA_DIR / "card_themes.json"
):
    with open(input_path, "r") as dataset_file:
        dataset = dataset_file.read()
        dataset_json = json.loads(dataset)

    track_progress(dataset_json, parse_dataset)
    card_themes = {card_name: list(themes) for card_name, themes in PARSED_CARD_THEMES.items()}
    with open(output_path, "w") as output_file:
        output_file.write(json.dumps(card_themes))

if __name__ == "__main__":
    main()