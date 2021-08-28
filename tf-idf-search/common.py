import json


def load_json(json_path):
    """Load wiki data from json file.

    Returns as a list of dictionaries, one per wiki article.
    """
    with open(json_path, "r") as f:
        return json.load(f)


def write_json(object, json_path):
    with open(json_path, "w") as f:
        json.dump(object, f)