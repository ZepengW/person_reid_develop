import json

def format_dict(d):
    return json.dumps(d, indent=4, ensure_ascii=False)