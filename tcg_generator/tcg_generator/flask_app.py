# call with python flask_app.py

import os
import re
from flask import Flask, render_template, request, jsonify
from tokenizers import Tokenizer
import torch
import sys
sys.path.append('../')
from tcg_generator.theme_options import THEME_OPTIONS, TYPE_LINE_OPTIONS

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer



app = Flask(__name__)

# I hard coded this... we'll have to make it relative to the project
MODEL_DIRECTORY = "../models/hf_gpt2_style_theme_model_quick_v4_15_epochs"
TOKENIZER_FILE = "../models/hf_gpt2_style_theme_model_quick_v4_15_epochs"
GENERATE_IMG = False

if(GENERATE_IMG):
    from diffusers import StableDiffusion3Pipeline

def prepare_for_html(string):
    if string is None:
        return None
    return string.replace("<", "&lt;").replace(">", "&gt;")


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
    # with torch.no_grad():
    #     output_sequences = model.generate(
    #         input_ids=encoded_prompt['input_ids'],
    #         attention_mask=encoded_prompt['attention_mask'],
    #         max_length=max_length,
    #         temperature=temperature,
    #         num_return_sequences=num_return_sequences,
    #         pad_token_id=tokenizer.pad_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #         bos_token_id=tokenizer.bos_token_id,
    #         # do_sample=True,
    #     )

# Testing NEW generation method
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=encoded_prompt['input_ids'],
            attention_mask=encoded_prompt['attention_mask'],
            max_length=max_length,
            do_sample=True,
            temperature=0.6,
            top_k=5,
            top_p=0.75,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )

    # Decode and return the generated sequences
    generated_sequences = []
    for generated_sequence in output_sequences:
        generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        generated_sequences.append(generated_text)

    return generated_sequences[0]

def generate_image(prompt):
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
    # If not using cuda, remove this line
    pipe = pipe.to("cuda")

    image = pipe(prompt, num_inference_steps=40, guidance_scale=4.5, ).images[0]
    image.save("static/img.png")


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
        themes_text = f"<THEMES> {' , '.join(selected_themes)} <CARD_NAME>"
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
            if (GENERATE_IMG):
                print("Generating image...")
                generate_image(model_output)
                print("Image generated!")
        else:
            print("Model not found!")
            model_output = f"This is a {selected_card_type.lower()} card combining the themes: {themes_text}."

        #TODO: Output parsing
        # Do all the parsing within here.
        # Mock card generation
        # This is where we'll have to parse out our generated card text and slap it in
        model_output = model_output.split("end]")[0]
        themes = model_output.split("<THEMES>")[-1].split("<CARD_NAME>")[0].replace("<", "&lt;")
        card_name = model_output.split("<CARD_NAME>")[-1].split("<MANA_COST>")[0]
        mana_cost = model_output.split("<MANA_COST>")[-1].split("<TYPE_LINE>")[0]
        # type_line = model_output.split("<TYPE_LINE>")[-1].split("<ORACLE_TEXT>")[0]
        # With the new generation method, the type line is generating more than one word.
        type_line = model_output.split("<TYPE_LINE>")[-1].split()[0]
        # If the card is neither a creature or planeswalker, it should end with the oracle text.
        if "<ORACLE_TEXT>" in model_output:
            oracle_text = model_output.split("<ORACLE_TEXT>")[-1].split("end]")[0]
        else:
            oracle_text = "No Generated Oracle Text."
        power = None
        toughness = None
        loyalty = None
        if "<POWER>" in model_output:
            # power_text = model_output.split("<POWER>")[0]
            power = model_output.split("<POWER>")[-1].split()[0] # Just get the first number after the <POWER> tag
        if "<TOUGHNESS>" in model_output:
            toughness = model_output.split("<TOUGHNESS>")[-1].split()[0] # Just get the first number after the <TOUGHNESS> tag
            print(f"\n\n{toughness}\n\n")
        elif "<LOYALTY>" in oracle_text:
            oracle_text = oracle_text.split("<LOYALTY>")[0]
            # Loyalty is expected to be the last part of a planeswalker
            loyalty = model_output.split("<TOUGHNESS>")[-1]

        card = {
            "themes": prepare_for_html(themes),
            "name": prepare_for_html(card_name),
            "mana_cost": prepare_for_html(mana_cost),
            "type": prepare_for_html(type_line),
            "text": prepare_for_html(oracle_text),
            "power": prepare_for_html(power),
            "toughness": prepare_for_html(toughness),
            "loyalty": prepare_for_html(loyalty),
            "flavor_text": f"The blended essence of {', '.join(selected_themes)} flows through this {selected_card_type.lower()}."
        }

        return jsonify({"success": True, "card": card})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)