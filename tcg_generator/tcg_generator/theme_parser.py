import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from tcg_generator.progress_tracker import track_progress
from tcg_generator.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

THEME_REGEX_MAP = {
    "Activated Abilities": [r"activated abilities"],
    "Adapt": [r"adapt"],
    "Affinity": [r"affinity"],
    "Alternative Cost": [r"may pay (?:.*) rather than pay"],
    "Amass": [r"amass"],
    "Any Number Of": [r"a deck can have any number of"],
    "Aura": [r"(?<!non-)aura"],
    "Cascade": [r"cascade"],
    "Combat Damage": [r"combat damage"],
    "Counter Theme": [r"\b(?:[a-z]+)\bcounter"], # not yet working.
    "Craft": [r"\bcraft"],
    "Curse": [r"curse\b"],
    "Cycling": [r"cycl(?:e|es|ing)"],
    "Damage": [r"deals (.*)(?<!combat )damage"],
    "Deathtouch": [r"deathtouch"],
    "Decayed": [r"decayed"],
    "Defender": [r"defender"],
    "Desert": [r"\bdesert\b"],
    "Devotion": [r"devotion"],
    "Discover": [r"\bdiscover\b"],
    "Dredge": [r"\bdredge\b"],
    "Dungeon": [r"venture into the dungeon", r"complete a dungeon"],
    "Double Strike": [r"double strike"],
    "Energy": [r"{e}"],
    "Enrage": [r"enrage", r"whenever (.*) is dealt damage"],
    "Equipment": [r"(?<!non-)equip(?:ment)?"],
    "Escape": [r"escape—", r"gains (?:\")escape", r"has escape"],
    "Evoke": [r"\bevoke\b"],
    "Exalted": [r"\bexalted\b", r"attacks alone"],
    "Exchange": [r"\bexchange\b"],
    "First Strike": [r"first strike"],
    "Flash": [r"flash"],
    "Food": [r"food"],
    "Flying": [r"flying"],
    "Gain Control": [r"gain(?:s)? control"],
    "Haste": [r"haste"],
    "Lifelink": [r"lifelink"],
    "Menace": [r"menace"],
    "Reach": [r"reach"],
    "Plot": [r"plot(?! counter)"],
    "Populate": [r"populate"],
    "Proliferate": [r"proliferate"],
    "Scry": [r"scry"],
    "Trample": [r"trample"],
    "Vigilance": [r"vigilance"],
    "Ward": [r"ward"],
    "+1/+1 Counters": [r"\+1/\+1 counter"],
    "-1/-1 Counters": [r"\-1/\-1 counter"],
}

PARSED_CARD_THEMES: dict[str, set[str]] = dict()

def parse_dataset(entry: Any):
    card_name = entry.get("name", "")
    oracle_text = entry.get("oracle_text", "").lower()

    reminder_text = r"\(.*\)"
    oracle_text = re.sub(reminder_text, "", oracle_text)

    for [theme, regexes] in THEME_REGEX_MAP.items():
        for regex in regexes:
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