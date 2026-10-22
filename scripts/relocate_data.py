"""Rewrite the absolute data paths in the shared manifest to this checkout's location.

The manifest records image and mask paths under the original training host's root
(/mnt/data/soccer_gif_reid/). After unzipping the shared data archive at the repository
root, run this once:

    python scripts/relocate_data.py

The manifest's SHA-256 changes, so run configs written afterwards record a different
manifest_sha256 than the published runs; the rows themselves are unchanged.
"""
from pathlib import Path

OLD = '/mnt/data/soccer_gif_reid/'
root = Path(__file__).resolve().parents[1]
new = str(root) + '/'
data = root / 'data/soccernet_full_v2'

for path in [data / 'manifest.jsonl', data / 'backup_prerepartition/masked_queries.jsonl']:
    if not path.exists():
        print('missing', path)
        continue
    text = path.read_text()
    count = text.count(OLD)
    if new != OLD and count:
        path.write_text(text.replace(OLD, new))
    print(f'{path.relative_to(root)}: {count} paths -> {new}')
