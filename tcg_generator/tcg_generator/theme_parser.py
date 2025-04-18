import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from tcg_generator.progress_tracker import track_progress
from tcg_generator.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

THEME_REGEX_MAP = {
    "Activated Abilities": [r"(?<!counter target )activated abilities.*(?!can't be activated)"],
    "Adamant": [r"adamant —"],
    "Adapt": [r"adapt"],
    "Addendum": [r"addendum —"],
    "Additional Combat": [r"additional combat"],
    "Affinity": [r"affinity"],
    "Afflict": [r"afflict (x|[0-9]+)"],
    "Afterlife": [r"afterlife"],
    "Alliance": [r"alliance —"],
    "Alternative Cost": [r"may (?:.*) rather than pay"],
    "Amass": [r"amass"],
    "Amplify": [r"amplify [0-9]+"],
    "Annihilator": [r"annihilator"],
    "Any Number Of": [r"a deck can have any number of"],
    "Artifact Based": [r"(?<!non)(?<!this )artifact"],
    "Ascend": [r"ascend\b", r"you have the city's blessing"],
    "Assist": [r"^\bassist\b", r"\n\bassist\b"],
    "Attacking": [r"(?:whenever you )?attack(?:s|ing)"],
    "Aura": [r"(?<!non-)aura", r"enchant\b", r"bestow\b"],
    "Awaken": [r"awaken (?:[0-9]+)"],
    "Backup": [r"backup (?:[0-9]+)"],
    "Banding": [r"banding\b", r"band(?:s)?\b"],
    "Bargain": [r"bargain\s+\n"],
    "Battalion": [r"battalion —", r"and at least two other creatures attack"],
    "Battle Cry": [r"battle cry (?!goblin)"],
    "Becomes Tapped": [r"whenever .* becomes tapped"],
    "Bestow": [r"bestow\b"],
    "Blitz": [r"blitz {", r"has blitz"],
    "Bloodthirst": [r"bloodthirst [0-9x]+"],
    "Boardwipe": [r"destroy all"],
    "Boast": [r"boast\b"],
    "Bolster": [r"bolster [0-9x]+"],
    "Bushido": [r"bushido [0-9x]+"],
    "Buyback": [r"buyback(?:—| {)"],
    "Card Draw": [r"draw(?:s)? .*card(?:s)?"],
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
    "Council's Dilemma": [r"council's dilemma —"],
    "Counterspell": [r"counter target.* spell", r"counter target.* ability", r"counter all", r"counter it"],
    "Counter Theme": [r"\b([a-z]+) counter(?:s)?\b"], # not yet working.
    "Coven": [r"coven —"],
    "Craft": [r"craft with"], # Doesn't work. Multi-faced cards have different oracle text location.
    "Creature Based": [r"(?<!non)(?<!this )creature"],
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
    "Dies": [r"\bdies\b"],
    "Discard": [r"\bdiscard\b"],
    "Discover": [r"\bdiscover\b"],
    "Disguise": [r"disguise {"],
    "Disturb": [r"disturb {"],
    "Domain": [r"domain —"],
    "Drain": [r"lose(?:s)? ([0-9]+|x) life.*you gain ([0-9]+|x) life", r"deal(?:s)? ([0-9]+|x) damage.*you gain ([0-9]+|x) life", r"lose(?:s)? ([0-9]+|x) life.*you gain life equal to", r"deal(?:s)? ([0-9]+|x) damage.*you gain life equal to"],
    "Dredge": [r"\bdredge\b"],
    "Dungeon": [r"venture into the dungeon", r"complete a dungeon"],
    "Double Strike": [r"double strike"],
    "Echo": [r"echo {", r"echo cost"],
    "Eerie": [r"eerie —"],
    "Embalm": [r"embalm {", r"gains embalm", r"embalm ability"],
    "Emerge": [r"emerge\b.*{", r"with emerge\b"],
    "Eminence": [r"eminence —"],
    "Enchantment Based": [r"(?<!non)(?<!this )enchantment"],
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
    "Flicker": [r"exile.*return it to the battlefield"],
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
    "Heroic": [r"heroic — "],
    "Hexproof": [r"hexproof"],
    "Hideaway": [r"hideaway [0-9]+"],
    "Horsemanship": [r"(?<!without )\bhorsemanship\b"],
    "Imprint": [r"imprint —"],
    "Improvise": [r"improvise"],
    "Incubate": [r"incubate(?:s)? (x|[0-9]+)"],
    "Indestructible": [r"indestructible"],
    "Infect": [r"\binfect\b"],
    "Initiative": [r"(?:take|have|has) the initiative"],
    "Inspired": [r"inspired —"],
    "Intimidate": [r"\bintimidate\b"],
    "Investigate": [r"\binvestigate\b"],
    "Jump-start": [r"\bjump-start\b"],
    "Kicker": [r"kicker {", r"kicker—", r"kicked spell"],
    "Kinship": [r"kinship —"],
    "Landfall": [r"landfall", r"whenever a land you control enters"],
    "Extra Lands": [r"play (?:an|two|up to three) additional land(?:s)?"],
    "Landwalk": [r"landwalk", r"islandwalk", r"swampwalk", r"forestwalk", r"plainswalk", r"mountainwalk"],
    "Level Up": [r"level up {", r"with level up"],
    "Lieutenant": [r"lieutenant —"],
    "Lifegain": [r"\bgain.*\blife\b"],
    "Lifelink": [r"lifelink"],
    "Living Weapon": [r"living weapon"],
    "Madness": [r"madness {", r"has madness", r"madness—pay six {c}"],
    "Magecraft": [r"magecraft —"],
    "Manifest": [r"\bmanifest\b"],
    "Megamorph": [r"megamorph"],
    "Meld": [r"\bmeld\b"],
    "Melee": [r"(?<!redcap )\bmelee\b"],
    "Menace": [r"\bmenace\b"],
    "Mentor": [r"(?<!proud )\bmentor\b", r"\bmentors\b"],
    "Metalcraft": [r"metalcraft —"],
    "Mill": [r"\bmill(?:s)?\b"],
    "Miracle": [r"miracle {", r"has miracle"],
    "Modular": [r"modular [0-9]+", r"modular—"],
    "Monarch": [r"become(?:s)? the monarch", r"you're the monarch"],
    "Monstrosity": [r"monstrosity (x|[0-9]+)"],
    "Morbid": [r"morbid —"],
    "Morph": [r"\bmorph\b"],
    "Mounts": [r"\bmount(?:s|ed)?\b", r"saddle [0-9]+", r"\bsaddle(?:d|s)\b"],
    "Multicolored": [r"\bmulticolored\b"],
    "Multikicker": [r"\bmultikicker\b"],
    "Mutate": [r"\bmutate(?:s)?\b"],
    "Myriad": [r"\bmyriad\b"],
    "Ninjutsu": [r"ninjutsu {"],
    "Offering": [r"(?:artifact|goblin|fox|moonfolk|rat|snake) offering"],
    "Offspring": [r"offspring {"],
    "Outlaw": [r"(?<!dunerider |glamorous )\boutlaw(?:s)?\b(?! stitcher)"],
    "Overload": [r"overload {"],
    "Pack Tactics": [r"pack tactics —"],
    "Paradox": [r"paradox —"],
    "Parley": [r"parley —"],
    "Partner": [r"\bpartner\b"],
    "Party": [r"in your party", r"a full party", r"choose(?:s)? a party"],
    "Pay Life": [r"pay (?:x |[0-9]+ )?life"],
    "Persist": [r"\bpersist\b"],
    "Phasing": [r"\bphasing\b"],
    "Reach": [r"reach"],
    "Plot": [r"plot(?! counter)"],
    "Poison": [r"\bpoison(?:ed)?\b"],
    "Populate": [r"populate"],
    "Proliferate": [r"proliferate"],
    "Protection": [r"protection from"],
    "Prototype": [r"prototype {"],
    "Provoke": [r"\bprovoke\b(?! the Trolls)"],
    "Prowess": [r"\bprowess\b"],
    "Prowl": [r"prowl {"],
    "Radiance": [r"radiance —"],
    "Raid": [r"raid —", r"if you attached.*this turn"],
    "Rally": [r"rally —"],
    "Rampage": [r"rampage [0-9]+"],
    "Ravenous": [r"\bravenous\b\s*\n"],
    "Read Ahead": [r"read ahead"],
    "Reanimation": [r"(?:put|return).* from.* graveyard (?:on)?to the battlefield"],
    "Rebound": [r"\brebound\b"],
    "Reconfigure": [r"reconfigure {", r"reconfigure—"],
    "Recover": [r"recover {", r"recover—"],
    "Regenerate": [r"(?<!can't be )regenerate"],
    "Reinforce": [r"reinforce (x|[0-9]+)—{"],
    "Renown": [r"renown [0-9]+", r"renowned"],
    "Retrace": [r"\bretrace\b"],
    "Revolt": [r"revolt —"],
    "Riot": [r"\briot\b"],
    "Ripple": [r"ripple [0-9]+"],
    "Roll Dice": [r"roll .* d20", r"roll .* d12", r"roll .* d10", r"roll .* dice"],
    "Sacrifice": [r"(?<!can't be )sacrifice(?:s|d)?"],
    "Scavenge": [r"scavenge {", r"has scavenge"],
    "Scry": [r"scry"],
    "Secret Council": [r"secret council —"],
    "Shadow": [r"\bshadow\b"],
    "Shroud": [r"\bshroud\b"],
    "Skulk": [r"\bskulk\b"],
    "Snow": [r"\bsnow\b(?: permanent| land| creature)?"],
    "Soulbond": [r"\bsoulbond\b"],
    "Soulshift": [r"\bsoulshift (x|[0-9]+)"],
    "Spectacle": [r"\bspectacle {"],
    "Spell Mastery": [r"spell mastery —"],
    "Splice": [r"splice onto"],
    "Split Second": [r"split second"],
    "Spree": [r"\bspree\b"],
    "Squad": [r"\bsquad\b {", r"\bsquad—"],
    "Storm": [r"^\bstorm\b", r"\n\bstorm\b"],
    "Strive": [r"\bstrive —"],
    "Sunburst": [r"\bsunburst\b"],
    "Support": [r"\bsupport (x|[0-9]+)"],
    "Surge": [r"\bsurge {"],
    "Surveil": [r"\bsurveil (x|[0-9]+)", r"whenever you surveil", r"surveilled this turn"],
    "Suspect": [r"\bsuspect(?:s|ed)?\b"],
    "Suspend": [r"\bsuspend (x|[0-9]+)—{", r"suspended card", r"is suspended", r"gains suspend"],
    "Tempting Offer": [r"tempting offer —"],
    "The Ring Tempts": [r"the ring tempts you"],
    "Threshold": [r"threshold —"],
    "Time Travel": [r"time travel"],
    "Toxic": [r"toxic (x|[0-9]+)"],
    "Training": [r"^\btraining\b", r"\n\btraining\b", r"have training"],
    "Trample": [r"trample"],
    "Transfigure": [r"transfigure {"],
    "Transform": [r"transform this", r"transform(?:s|ed)"],
    "Transmute": [r"transmute {"],
    "Turned Face Up": [r"turned face up"],
    "Tutor": [r"search (?:your|their) library"],
    "Umbra Armor": [r"umbra armor"],
    "Unblockable": [r"can't be blocked"],
    "Uncounterable": [r"can't be countered"],
    "Undaunted": [r"^\bundaunted\b", r"has undaunted"],
    "Undergrowth": [r"undergrowth —"],
    "Undying": [r"^\bundying\b", r"\n\bundying\b", r"gains undying", r"has undying"],
    "Unearth": [r"\bunearth {", r"\bunearth—"],
    "Unleash": [r"^\bunleash\b(?! shell| the)", r"\n\bunleash\b"],
    "Upkeep": [r"\bupkeep\b"],
    "Valiant": [r"valiant —"],
    "Vanishing": [r"vanishing (x|[0-9]+)", r"gains vanishing", r"\n\bvanishing\b"],
    "Vigilance": [r"vigilance"],
    "Voting": [r"vot(?:es|ing)"],
    "Ward": [r"ward(?:—| {)"],
    "Will of the Council": [r"will of the council —"],
    "Will of the Planeswalkers": [r"will of the planeswalkers —"],
    "Wither": [r"\bwither\b"],
    "+1/+1 Counters": [r"\+1/\+1 counter"],
    "-1/-1 Counters": [r"\-1/\-1 counter"],
}

PARSED_CARD_THEMES: dict[str, set[str]] = dict()
TYPE_LINE_OPTIONS: set[str] = set()
THEME_OPTIONS: set[str] = set()

def parse_dataset(entry: Any):
    card_name = entry.get("name", "")
    oracle_text = entry.get("oracle_text", "").lower()

    PARSED_CARD_THEMES.setdefault(card_name, set())
    reminder_text = r"\(.*\)"
    oracle_text = re.sub(reminder_text, "", oracle_text)

    for [theme, regexes] in THEME_REGEX_MAP.items():
        for regex in regexes:
            match = re.findall(regex, oracle_text)
            if len(match) > 0:
                if theme == "Counter Theme":
                    theme = match[0].capitalize() + " Counters"
                PARSED_CARD_THEMES[card_name].add(theme)
                THEME_OPTIONS.add(theme)

    all_parts = entry.get("all_parts", list())
    if len(all_parts) > 0:
        for part in all_parts:
            if part["object"] == "related_card" and part["component"] == "token":
                token_name = part["name"]
                theme = f"{token_name} Tokens"
                PARSED_CARD_THEMES[card_name].add(theme)
                THEME_OPTIONS.add(theme)

    card_colors = entry.get("colors", list())
    color = "".join(card_colors)
    if color == "":
        # It is a colorless card, let's add some special representation.
        color = "N"
    color_theme = f"{color} Color"
    PARSED_CARD_THEMES[card_name].add(color_theme)
    THEME_OPTIONS.add(color_theme)

    card_identity_colors = entry.get("color_identity", list())
    card_identity = "".join(card_identity_colors)
    if card_identity == "":
        # It is a colorless identity, let's add some special representation.
        card_identity = "N"
    identity_theme = f"{card_identity} Identity"
    PARSED_CARD_THEMES[card_name].add(identity_theme)
    THEME_OPTIONS.add(identity_theme)

    type_line = entry.get("type_line", "")
    types = type_line.split(" ")
    for type in types:
        if type == "—":
            continue
        elif type == "\\\\":
            continue
        PARSED_CARD_THEMES[card_name].add(type)
        TYPE_LINE_OPTIONS.add(type)

    mana_cost = entry.get("mana_cost", "")
    mana_cost_theme = f"{mana_cost} Cost"
    PARSED_CARD_THEMES[card_name].add(mana_cost_theme)
    THEME_OPTIONS.add(mana_cost_theme)


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

    with open(PROCESSED_DATA_DIR / "theme_options.json", "w") as theme_options_file:
        theme_options_file.write(json.dumps(sorted(list(THEME_OPTIONS)), indent=4))

    with open(PROCESSED_DATA_DIR / "type_options.json", "w") as type_options_file:
        type_options_file.write(json.dumps(sorted(list(TYPE_LINE_OPTIONS)), indent=4))

if __name__ == "__main__":
    main()