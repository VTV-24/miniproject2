import json
from pathlib import Path

KERNEL_SPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

LANG_INFO = {
    "name": "python",
    "version": "3.10"
}

nb_dir = Path("notebooks")

for nb in nb_dir.glob("*.ipynb"):
    with open(nb, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.setdefault("metadata", {})
    data["metadata"]["kernelspec"] = KERNEL_SPEC
    data["metadata"]["language_info"] = LANG_INFO

    with open(nb, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Fixed:", nb)

print("Done.")
