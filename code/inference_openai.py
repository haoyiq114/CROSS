import os
import json
import argparse
import base64
from openai import OpenAI
import tqdm

# Initialize OpenAI using the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("The OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=api_key)

# Helper: Encode an image as a base64 string

def encode_image(image_path: str) -> str:
    """Read an image from *image_path* and return its Base64‑encoded string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# Helper: Call the OpenAI API with text and image

def generate_response(client: OpenAI, model_name: str, query: str, base64_image: str) -> str:
    """Send *query* and *base64_image* to the model and return the response text."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during the API call: {e}")
        return "ERROR: The API call failed."


def run_generation(input_filepath: str, output_filepath: str, model_name: str) -> None:
    """Generate model responses for *input_filepath* and save them to *output_filepath*."""

    # Load the input data
    with open(input_filepath, "r") as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} entries from {input_filepath}.")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # If an output file already exists, load its contents and append new results
    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            results = json.load(f)
        print(
            f"Appending to the existing results file {output_filepath} "
            f"which already contains {len(results)} entries."
        )
    else:
        print(f"Creating a new results file at {output_filepath}.")
        results = []

    # Iterate through the dataset and generate responses
    for idx, data in enumerate(tqdm.tqdm(data_list), start=1):
        # Skip items that have already been processed
        if idx <= len(results):
            continue

        query = data.get("query", "")
        image_path = data.get("file_name", "")

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} was not found; skipping this entry.")
            continue

        base64_img = encode_image(image_path)
        result = generate_response(client, model_name, query, base64_img)
        data["model_response"] = result
        results.append(data)

        # Write intermediate results to disk after each iteration
        with open(output_filepath, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model responses for CASA and SafeWorld datasets."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Full model name used for generation.",
    )
    parser.add_argument(
        "--setting",
        default="direct",
        help="Generation setting (default: direct)",
    )
    parser.add_argument(
        "--round",
        required=True,
        help="Evaluation round index (e.g., 1).",
    )

    args = parser.parse_args()
    model_name = args.model_name
    setting = args.setting
    run_id = args.round

    # CASA
    casa_input_filepath = f"../data/casa/{setting}.json"
    casa_output_filepath = (
        f"../generation_results/{setting}/casa/results_{model_name}_r{run_id}.json"
    )
    run_generation(casa_input_filepath, casa_output_filepath, model_name)

    # SafeWorld
    safeworld_input_filepath = f"../data/safeworld/{setting}.json"
    safeworld_output_filepath = (
        f"../generation_results/{setting}/safeworld/results_{model_name}_r{run_id}.json"
    )
    run_generation(safeworld_input_filepath, safeworld_output_filepath, model_name)
