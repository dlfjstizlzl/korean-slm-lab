from pathlib import Path
from datasets import load_dataset
import unicodedata
import json

ROOT = Path("data/external")
JSON_DIR = ROOT / "json"
SQL_DIR = ROOT / "sql"

JSON_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)


def normalize(text):
    if not isinstance(text, str):
        return ""

    return unicodedata.normalize("NFC", text).replace("\x00", "").strip()


def save_stream(dataset, output, target_chars, field="content"):
    chars = 0
    rows = 0
    seen = set()

    with output.open("w", encoding="utf-8") as f:
        for row in dataset:
            text = normalize(row.get(field, ""))

            if not text:
                continue

            # exact dedup
            key = hash(text)

            if key in seen:
                continue

            seen.add(key)

            f.write(text)
            f.write("\n")

            chars += len(text)
            rows += 1

            if rows % 1000 == 0:
                print(
                    output.name,
                    f"{rows:,} rows",
                    f"{chars:,} chars",
                )

            if chars >= target_chars:
                break

    print(
        f"[DONE] {output}: "
        f"{rows:,} rows / {chars:,} chars"
    )


# --------------------------------------------------
# 1. StarCoderData JSON
# --------------------------------------------------

json_ds = load_dataset(
    "bigcode/starcoderdata",
    data_dir="json",
    split="train",
    streaming=True,
)

save_stream(
    json_ds,
    JSON_DIR / "starcoder_json.txt",
    target_chars=3_000_000,
    field="content",
)


# --------------------------------------------------
# 2. The Stack Smol SQL
# --------------------------------------------------

sql_ds = load_dataset(
    "bigcode/the-stack-smol",
    data_dir="data/sql",
    split="train",
)

save_stream(
    sql_ds,
    SQL_DIR / "stack_sql.txt",
    target_chars=3_000_000,
    field="content",
)


# --------------------------------------------------
# 3. Spider
# --------------------------------------------------

spider = load_dataset(
    "xlangai/spider",
    "spider",
)

with (SQL_DIR / "spider_sql.txt").open(
    "w",
    encoding="utf-8",
) as f:

    seen = set()

    for split_name in ["train", "validation"]:
        for row in spider[split_name]:

            sql = normalize(row["query"])

            if sql and sql not in seen:
                f.write(sql + "\n")
                seen.add(sql)

print(
    "[DONE] Spider:",
    len(seen),
    "unique SQL queries"
)