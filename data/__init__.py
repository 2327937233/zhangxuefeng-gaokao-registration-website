import json
import os

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(filename: str) -> list | dict:
    path = os.path.join(_DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


majors = load_json('majors.json')
schools = load_json('schools.json')
quotes = load_json('quotes.json')
expression_dna = load_json('expression_dna.json')
