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
    "Corrupted": [r"corrupted —"],
    "Counter Theme": [r"\b([a-z]+) counter(?:s)?\b"], # not yet working.
    "Coven": [r"coven —"],
    "Craft": [r"craft with"], # Doesn't work. Multi-faced cards have different oracle text location.
    "Crew": [r"crew [0-9]+", r"crew(?:s)? vehicle"],
    "Cumulative Upkeep": [r"cumulative upkeep"],
    "Curse": [r"curse\b"],
    "Cycling": [r"cycl(?:e|es|ing)"],
    "Damage": [r"deals (.*)(?<!combat )damage"],
    "Dash": [r"dash {"],
    "Daybound": [r"ddaybound"], # Doesn't work. Multi-faced cards have different oracle text location.
    "Deathtouch": [r"deathtouch"],
    "Decayed": [r"decayed"],
    "Defender": [r"defender"],
    "Delirium": [r"delirium —"],
    "Delve": [r"\bdelve\b"],
    "Demonstrate": [r"\bdemonstrate\b"],
    "Descend": [r"descend [0-9]+", r"if you descended"],
    "Desert": [r"\bdesert\b"],
    "Detain": [r"\bdetain\b"],
    "Dethrone": [r"dethrone"],
    "Devoid": [r"devoid"],
    "Devotion": [r"devotion"],
    "Devour": [r"devour\s(?:.*\s)?[0-9]+"],
    "Discover": [r"\bdiscover\b"],
    "Disguise": [r"disguise {"],
    "Disturb": [r"disturb {"],
    "Domain": [r"domain —"],
    "Drain": [r"lose(?:s)? ([0-9]+|X) life.*you gain ([0-9]+|X) life", r"deal(?:s)? ([0-9]+|X) damage.*you gain ([0-9]+|X) life", r"lose(?:s)? ([0-9]+|X) life.*you gain life equal to", r"deal(?:s)? ([0-9]+|X) damage.*you gain life equal to"],
    "Dredge": [r"\bdredge\b"],
    "Dungeon": [r"venture into the dungeon", r"complete a dungeon"],
    "Double Strike": [r"double strike"],
    "Echo": [r"echo {", r"echo cost"],
    "Eerie": [r"eerie —"],
    "Embalm": [r"embalm {", r"gains embalm", r"embalm ability"],
    "Emerge": [r"emerge\b.*{", r"with emerge\b"],
    "Eminence": [r"eminence —"],
    "Encore": [r"encore {", r"(gains|has) encore"],
    "Energy": [r"{e}"],
    "Enlist": [r"\benlist\b"],
    "Enrage": [r"enrage", r"whenever (.*) is dealt damage"],
    "Entwine": [r"entwine {", r"entwine—"],
    "Epic": [r"\bepic\b"],
    "Equipment": [r"(?<!non-)equip(?:ment)?"],
    "Escalate": [r"escalate {", r"escalate—"],
    "Escape": [r"escape—", r"gains (?:\")escape", r"has escape"],
    "Eternalize": [r"eternalize", r"eternalize ability"],
    "Evolve": [r"\bevolve\b"],
    "Evoke": [r"\bevoke\b"],
    "Exalted": [r"\bexalted\b", r"attacks alone"],
    "Exchange": [r"\bexchange\b"],
    "Exert": [r"\bexert\b"],
    "Exploit": [r"\bexploit\b"],
    "Explore": [r"\bexplores\b"],
    "Extort": [r"\bextort\b"],
    "Fabricate": [r"fabricate [0-9]+"],
    "Fading": [r"fading [0-9]+", r"with fading"],
    "Fateseal": [r"fateseal [0-9]+"],
    "Fear": [r"\bfear\b", r"gains fear"],
    "Ferocious": [r"ferocious —"],
    "Flanking": [r"\bflanking\b"],
    "Fights": [r"fight(?:s)?(?: up to one| another)? target creature", r"fight each other"],
    "First Strike": [r"first strike"],
    "Flash": [r"flash\b"],
    "Flashback": [r"flashback"],
    "Flip a Coin": [r"flip [a-zA-Z0-9]+ coin(?:s)?"],
    "Flying": [r"flying"],
    "Fog": [r"prevent all (?:combat )?damage"],
    "Food": [r"food"],
    "For Mirrodin": [r"for mirrodin!"],
    "Forage": [r"\bforage\b"],
    "Forecast": [r"forecast — {"],
    "Foretell": [r"foretell"],
    "Formidable": [r"formidable —"],
    "Fortify": [r"fortify {"],
    "Free Spells": [r"without paying (?:its|their) mana cost(?:s)?"],
    "Freerunning": [r"freerunning {", r"(has|have|with) freerunning", r"freerunning—"],
    "Gain Control": [r"gain(?:s)? control"],
    "Gift": [r"gift a", r"give a gift"],
    "Goad": [r"goad"],
    "Graft": [r"graft [0-9]+"],
    "Gravestorm": [r"\bgravestorm\b"],
    "Haste": [r"haste"],
    "Haunt": [r"\bhaunt\s+\n"],
    "Hellbent": [r"hellbent —"],
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