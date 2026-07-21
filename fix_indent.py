import json

notebook_path = r"u:\assesment\notebooks\DeepGlobe_PartialCE.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if line.strip().startswith('"DATA_ROOT = r\\"u:\\\\assesment'):
                new_source.append('DATA_ROOT = r"u:\\assesment\\DeepGlobe Land Cover Classification Dataset"  # <-- point at real DeepGlobe folder when USE_SYNTHETIC_DATA=False\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
