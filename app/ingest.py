import pandas as pd
from pathlib import Path

def load_notes(path):
    p = Path(path)
    if p.is_dir():
        files = list(p.glob('*.txt'))
    else:
        files = [p]
    rows = []
    for f in files:
        text = f.read_text(encoding='utf-8')
        rows.append({'id': f.stem, 'text': text})
    return pd.DataFrame(rows)
