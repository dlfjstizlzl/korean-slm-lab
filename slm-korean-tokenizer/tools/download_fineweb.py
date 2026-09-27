from datasets import load_dataset
from pathlib import Path
import unicodedata

TARGET_CHARS = 50_000_000

output = Path("data/raw/fineweb_ko_50m.txt")
output.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "HuggingFaceFW/fineweb-2",
    "kor_Hang",
    split="train",
    streaming=True,
)

chars = 0
docs = 0

with output.open("w", encoding="utf-8") as f:
    for row in ds:
        text = row["text"]

        if not text:
            continue

        text = unicodedata.normalize("NFC", text).strip()

        if len(text) < 100:
            continue

        f.write(text.replace("\x00", "") + "\n")

        chars += len(text)
        docs += 1

        if chars >= TARGET_CHARS:
            break

print(f"documents = {docs:,}")
print(f"characters = {chars:,}")
print(f"saved = {output}")
