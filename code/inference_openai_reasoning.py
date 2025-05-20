import os
import json
import argparse
import base64
from openai import OpenAI
import tqdm

REASONING_EFFORT_LEVELS = {"low", "medium", "high"}

# Initialize OpenAI from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("The OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=api_key)


def encode_image(image_path: str) -> str:
    """Return *image_path* encoded as a Base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def generate_response(client: OpenAI, model_name: str, reasoning_effort: str, query: str, base64_image: str) -> str:
    """
    Send *query* together with *base64_image* to the model and
    return the generated response text.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            reasoning_effort=reasoning_effort,
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


def run_generation(input_filepath: str, output_filepath: str, model_name: str, reasoning_effort: str) -> None:
    """Generate model responses from *input_filepath* and write them to *output_filepath*."""

    # Load the input data
    with open(input_filepath, "r") as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} entries from {input_filepath}.")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Resume from an existing results file if present
    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            results = json.load(f)
        print(
            f"Appending to existing results in {output_filepath}; "
            f"found {len(results)} completed entries."
        )
    else:
        print(f"Creating a new results file at {output_filepath}.")
        results = []

    # Iterate through the dataset and generate responses
    for idx, data in enumerate(tqdm.tqdm(data_list), start=1):
        # Skip items already processed
        if idx <= len(results):
            continue

        query = data.get("query", "")
        image_path = data.get("file_name", "")

        if not os.path.exists(image_path):
            print(f"Warning: image {image_path} not found; skipping this entry.")
            continue

        base64_img = encode_image(image_path)
        result = generate_response(client, model_name, reasoning_effort, query, base64_img)
        data["model_response"] = result
        results.append(data)

        # Persist intermediate results after each iteration
        with open(output_filepath, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model responses for the CASA and SafeWorld datasets."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Full model name used for generation (e.g., gpt-4o).",
    )
    parser.add_argument(
        "--setting",
        default="direct",
        help="Generation setting (default: direct).",
    )
    parser.add_argument(
        "--reasoning_effort",
        required=True,
        choices=REASONING_EFFORT_LEVELS,
        help="Reasoning effort level: low, medium, or high.",
    )
    parser.add_argument(
        "--round",
        required=True,
        help="Evaluation‑round index (e.g., 1).",
    )

    args = parser.parse_args()
    model_name = args.model_name
    setting = args.setting
    reasoning_effort = args.reasoning_effort
    run_id = args.round

    # CASA
    casa_input_filepath = f"../data/casa/{setting}.json"
    casa_output_filepath = (
        f"../generation_results/{setting}/casa/results_{model_name}_r{run_id}.json"
    )
    run_generation(casa_input_filepath, casa_output_filepath, model_name, reasoning_effort)

    # SafeWorld
    safeworld_input_filepath = f"../data/safeworld/{setting}.json"
    safeworld_output_filepath = (
        f"../generation_results/{setting}/safeworld/results_{model_name}_r{run_id}.json"
    )
    run_generation(safeworld_input_filepath, safeworld_output_filepath, model_name, reasoning_effort)
