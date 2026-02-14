from pathlib import Path

def label_from_filename(path: str | Path) -> str:
    """Oxford-IIIT Pet filenames look like: 'american_bulldog_34.jpg'
    Breed name may contain underscores; numeric id is after the last underscore.
    """
    name = Path(path).stem  # no extension
    if "_" not in name:
        return name
    return name.rsplit("_", 1)[0]
