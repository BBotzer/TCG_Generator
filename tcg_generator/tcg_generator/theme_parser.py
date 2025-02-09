import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from tcg_generator.progress_tracker import track_progress
from tcg_generator.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

THEME_REGEX_MAP = {
    "Activated Abilities": [r"activated abilities"],
    "Adamant": [r"adamant"],
    "Adapt": [r"adapt"],
    "Additional Combat": [r"additional combat"],
    "Affinity": [r"affinity"],
    "Afterlife": [r"afterlife"],
    "Alliance": [r"alliance —"],
    "Alternative Cost": [r"may (?:.*) rather than pay"],
    "Amass": [r"amass"],
    "Annihilator": [r"annihilator"],
    "Any Number Of": [r"a deck can have any number of"],
    "Ascend": [r"ascend\b", r"you have the city's blessing"],
    "Aura": [r"(?<!non-)aura", r"enchant\b", r"bestow\b"],
    "Awaken": [r"awaken (?:[0-9]+)"],
    "Backup": [r"backup (?:[0-9]+)"],
    "Banding": [r"banding\b", r"band(?:s)?\b"],
    "Bargain": [r"bargain\s+\n"],
    "Battalion": [r"battalion —", r"and at least two other creatures attack"],
    "Battle Cry": [r"battle cry (?!goblin)"],
    "Bestow": [r"bestow\b"],
    "Blitz": [r"blitz {", r"has blitz"],
    "Bloodthirst": [r"bloodthirst [0-9x]+"],
    "Boast": [r"boast\b"],
    "Bolster": [r"bolster [0-9x]+"],
    "Bushido": [r"bushido [0-9x]+"],
    "Buyback": [r"buyback(?:—| {)"],
    "Cascade": [r"cascade"],
    "Casualty": [r"casualty"],
    "Celebration": [r"celebration —"],
    "Champion": [r"champion a(?:n)?\b"],
    "Changeling": [r"changeling"],
    "Channel": [r"channel —"],
    "Choose a Background": [r"choose a background"],
    "Chroma": [r"chroma —"],
    "Cipher": [r"\bcipher"],
    "Clash": [r"\bclash\b"],
    "Cleave": [r"cleave {"],
    "Cloak": [r"cloak(?:s)? the top card", r"cloak those cards", r"cloak a card", r"cloak two"],
    "Clone": [r"enter(?:s)? (tapped )?as a copy"],
    "Cohort": [r"cohort —"],
    "Collect Evidence": [r"collect evidence [0-9]+"],
    "Combat Damage": [r"combat damage"],
    "Commit a Crime": [r"commit a crime"],
    "Companion": [r"companion —"],
    "Compleated": [r"compleated\s+\n"],
    "Connive": [r"connive"],
    "Conspire": [r"conspire"],
    "Constellation": [r"constellation —"],
    "Convoke": [r"convoke"],
    "Counter Theme": [r"\b([a-z]+) counter(?:s)?\b"], # not yet working.
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
    "Unblockable": [r"can't be blocked"],
    "Uncounterable": [r"can't be countered"],
    "Vigilance": [r"vigilance"],
    "Ward": [r"ward(?:—| {)"],
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
    print(f"Total themes checked: {len(THEME_REGEX_MAP.keys())}")
    print(f"Total cards with themes: {len(card_themes)}")
    with open(output_path, "w") as output_file:
        output_file.write(json.dumps(card_themes))

if __name__ == "__main__":
    main()