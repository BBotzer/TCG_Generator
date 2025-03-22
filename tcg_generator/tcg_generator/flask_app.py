# call with python flask_app.py

import os
import re
from flask import Flask, render_template, request, jsonify
from tokenizers import Tokenizer
import torch
from tcg_generator.theme_options import THEME_OPTIONS, TYPE_LINE_OPTIONS

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer

app = Flask(__name__)

# I hard coded this... we'll have to make it relative to the project
MODEL_DIRECTORY = "../models/hf_gpt2_style_theme_model_v2_10_epochs"
TOKENIZER_FILE = "../models/hf_gpt2_style_theme_model_v2_10_epochs"

THEMES_FORMAT = r"<THEMES> ([a-zA-Z0-9,/\s{}-]+)"
CARD_NAME_FORMAT = r"<CARD_NAME> ([a-zA-Z0-9<>]+)"
MANA_COST_FORMAT = r"<MANA_COST> ([\{\}0-9WUBRG\s]+)"
TYPE_LINE_FORMAT = r"<TYPE_LINE> ([a-zA-Z0-9\s,—]+)"
ORACLE_TEXT_FORMAT = r"<ORACLE_TEXT> ([a-zA-Z0-9,\s|<name>]+)"
POWER_FORMAT = r"<POWER> ([0-9X+]+)"
TOUGHNESS_FORMAT = r"<TOUGHNESS> ([0-9X+]+)"
LOYALTY_FORMAT = r"<LOYALTY> ([0-9X+]+)"


def generate_text(
        prompt, max_length=300, num_return_sequences=1, temperature=1.0):
    """
    Generate text using the trained GPT2 model.

    Args:
        model: The trained GPT2 model
        tokenizer: The tokenizer used for encoding/decoding text
        prompt: The input prompt text to generate from
        max_length: Maximum length of generated sequence
        num_return_sequences: Number of sequences to generate
        temperature: Controls randomness (higher = more random)

    Returns:
        List of generated text sequences
    """
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained(MODEL_DIRECTORY)
    model = model.to(device)
    model.eval()

    # Need to load tokenizer from saved folder.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FILE)
    # Encode the input prompt
    encoded_prompt = tokenizer(prompt, return_tensors='pt').to(device)

    # Generate text
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=encoded_prompt['input_ids'],
            attention_mask=encoded_prompt['attention_mask'],
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            # do_sample=True,
        )

    # Decode and return the generated sequences
    generated_sequences = []
    for generated_sequence in output_sequences:
        generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        generated_sequences.append(generated_text)

    return generated_sequences[0]

@app.route('/')
def index():
    """Render the main page with the form"""
    return render_template('index.html', themes=THEME_OPTIONS, card_types=TYPE_LINE_OPTIONS)

@app.route('/generate', methods=['POST'])
def generate():
    """Process the form data and generate a card"""

    # This is the form data in the form of a dict
    # It will need to be processed for the model input
    form_data = request.form.to_dict()

    try:
        # Process the form_data for the model input
        selected_themes = list()
        for i in range(int(form_data["num_themes"])):
            theme_key = f"theme_{i}"
            if not theme_key in form_data:
                continue

            selected_themes.append(form_data[theme_key])

        selected_card_type = form_data["card_type"]
        selected_themes.append(selected_card_type)
        themes_text = f"<THEMES> {" , ".join(selected_themes)} <CARD_NAME>"
        print(themes_text)

        # Run our model here
        if (
            (MODEL_DIRECTORY != "" and os.path.exists(MODEL_DIRECTORY))
            and (TOKENIZER_FILE != "" and os.path.exists(TOKENIZER_FILE))
        ):
            # load the tokenizer from a saved folder as well.
            print("Attempting to generate text...")
            print(f"Theme text: {themes_text}")
            model_output = generate_text(themes_text)
            print(f"Model output: {model_output}")
        else:
            print("Model not found!")
            model_output = f"This is a {selected_card_type.lower()} card combining the themes: {themes_text}."

        #TODO: Output parsing
        # Do all the parsing within here.
        # Mock card generation
        # This is where we'll have to parse out our generated card text and slap it in
        model_output = model_output.split("[end]")[0]
        print(model_output)
        print("Matching theme")
        theme_match = re.match(THEMES_FORMAT, model_output)
        themes = theme_match.group(0) if theme_match is not None else None
        print(f"Matched theme: {themes != None}")
        print("Matching card name")
        card_name_match = re.match(CARD_NAME_FORMAT, model_output)
        card_name = card_name_match.group(0) if card_name_match is not None else None
        print(f"Matched card name: {card_name != None}")
        print("Matching mana cost")
        mana_cost_match = re.match(MANA_COST_FORMAT, model_output)
        mana_cost = mana_cost_match.group(0) if mana_cost_match is not None else None
        print(f"Matched mana cost: {mana_cost != None}")
        print("Matching type line")
        type_line_match = re.match(TYPE_LINE_FORMAT, model_output)
        type_line = type_line_match.group(0) if type_line_match is not None else None
        print(f"Matched type line: {type_line != None}")
        print("Matching oracle text")
        oracle_text_match = re.match(ORACLE_TEXT_FORMAT, model_output)
        oracle_text = oracle_text_match.group(0) if oracle_text_match is not None else None
        print(f"Matched oracle_text: {oracle_text != None}")
        print("Matching power")
        power_match = re.match(POWER_FORMAT, model_output)
        power = power_match.group(0) if power_match is not None else None
        print(f"Matched power: {power != None}")
        print("Matching toughness")
        toughness_match = re.match(TOUGHNESS_FORMAT, model_output)
        toughness = toughness_match.group(0) if toughness_match is not None else None
        print(f"Matched toughness: {toughness != None}")
        print("Matching loyalty")
        loyalty_match = re.match(LOYALTY_FORMAT, model_output)
        loyalty = loyalty_match.group(0) if loyalty_match is not None else None
        print(f"Matched loyalty: {loyalty != None}")

        card = {
            "themes": themes,
            "name": card_name,
            "mana_cost": mana_cost,
            "type": type_line,
            "text": oracle_text,
            "power": power,
            "toughess": toughness,
            "loyalty": loyalty,
            "flavor_text": f"The blended essence of {', '.join(selected_themes)} flows through this {selected_card_type.lower()}."
        }

        return jsonify({"success": True, "card": card})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)