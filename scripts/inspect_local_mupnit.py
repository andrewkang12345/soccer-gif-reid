"""Safely extract the newly supplied ReID archive and record its actual layout."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/MuPNIT-ReID.tar.gz"
DEST = ROOT / "data/MuPNIT-ReID"


def main():
    if DEST.exists():
        raise FileExistsError("Inspect existing extracted data instead of overwriting it")
    counts, sizes = Counter(), Counter()
    examples, identities = defaultdict(list), defaultdict(set)
    metadata = []
    begin = datetime.now(timezone.utc).isoformat()
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "MuPNIT-ReID":
                raise ValueError(f"Unexpected member path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unexpected link/special member: {member.name}")
            archive.extract(member, path=ROOT / "data", filter="data")
            if not member.isfile():
                continue
            section = path.parts[1] if len(path.parts) > 2 else "root_files"
            counts[section] += 1; sizes[section] += member.size
            if len(examples[section]) < 8:
                examples[section].append(member.name)
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                if path.parent.name.startswith("player_"):
                    pid = str(int(path.parent.name.split("_")[-1]))
                elif path.name.split("_")[0].isdigit():
                    pid = str(int(path.name.split("_")[0]))
                else:
                    pid = "unparsed:" + path.parent.name
                identities[section].add(pid)
            elif path.suffix.lower() in {".json", ".txt", ".md", ".csv"}:
                metadata.append(member.name)
            if sum(counts.values()) % 25000 == 0:
                print(json.dumps({"extracted_files": sum(counts.values()), "sections": counts}), flush=True)
    report = {"started_at": begin, "completed_at": datetime.now(timezone.utc).isoformat(),
              "archive": str(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
              "extracted_root": str(DEST), "files_by_section": dict(counts),
              "bytes_by_section": dict(sizes), "examples": dict(examples),
              "identities_by_section": {k: sorted(v) for k, v in identities.items()}, "metadata_files": metadata,
              "training_started": False, "note": "Layout/extraction audit; reference quality, split leakage and semantic identity mapping still require validation."}
    (ROOT / "reports/local_dataset_inventory.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"complete": True, "files": counts, "identities": {k: len(v) for k, v in identities.items()}, "metadata": metadata}), flush=True)


if __name__ == "__main__":
    main()
