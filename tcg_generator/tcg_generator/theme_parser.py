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
    "Counter Theme": [r"\b([a-z]+) counter"], # not yet working.
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
    "Explore": [r"\bexplores\b"],
    "Fights": [r"fight(?:s)?(?: up to one| another)? target creature", r"fight each other"],
    "First Strike": [r"first strike"],
    "Flash": [r"flash\b"],
    "Flashback": [r"flashback"],
    "Flip a Coin": [r"flip [a-zA-Z0-9]+ coin(?:s)?"],
    "Flying": [r"flying"],
    "Fog": [r"prevent all (?:combat )?damage"],
    "Food": [r"food"],
    "Foretell": [r"foretell"],
    "Free Spells": [r"without paying (?:its|their) mana cost(?:s)?"],
    "Gain Control": [r"gain(?:s)? control"],
    "Goad": [r"goad"],
    "Haste": [r"haste"],
    "Hexproof": [r"hexproof"],
    "Improvise": [r"improvise"],
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
                if (theme == "Counter Theme"):
                    theme = match.group(1).capitalize() + " Counters"
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
    print()
    print(f"Total cards with themes: {len(card_themes)}")
    with open(output_path, "w") as output_file:
        output_file.write(json.dumps(card_themes))

if __name__ == "__main__":
    main()