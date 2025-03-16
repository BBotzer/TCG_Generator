# call with python flask_app.py

import os
from flask import Flask, render_template, request, jsonify
from tokenizers import Tokenizer
import torch
from tcg_generator.theme_options import THEME_OPTIONS, TYPE_LINE_OPTIONS

app = Flask(__name__)

MODEL_DIRECTORY = ""
TOKENIZER_FILE = ""

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

    model = torch.load(MODEL_DIRECTORY)
    model = model.to(device)
    model.eval()

    # Need to load tokenizer from saved folder.
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
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
            do_sample=True,
        )

    # Decode and return the generated sequences
    generated_sequences = []
    for generated_sequence in output_sequences:
        generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        generated_sequences.append(generated_text)

    return generated_sequences

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
        themes_text = f"<THEMES> {" , ".join(selected_themes)}"
        print(themes_text)

        # Run our model here
        if (
            (MODEL_DIRECTORY != "" and os.path.exists(MODEL_DIRECTORY))
            or (TOKENIZER_FILE != "" and os.path.exists(TOKENIZER_FILE))
        ):
            # Need to load the tokenizer from a saved folder as well.
            model_output = generate_text(themes_text)
        else:
            model_output = f"This is a {selected_card_type.lower()} card combining the themes: {themes_text}."

        # Do all the parsing within here.
        # Mock card generation
        # This is where we'll have to parse out our generated card text and slap it in
        card = {
            "name": themes_text,
            "type": selected_card_type,
            "text": model_output,
            "flavor_text": f"The blended essence of {', '.join(selected_themes)} flows through this {selected_card_type.lower()}."
        }

        return jsonify({"success": True, "card": card})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)