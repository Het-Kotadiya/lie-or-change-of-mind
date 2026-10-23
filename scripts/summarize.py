# summarize.py: count each kind of agent in every results file
import glob, json, os
from collections import Counter

KINDS = [
    ("premeditated_deception", "real_lie"),
    ("fully_honest", "honest"),
    ("impulsive_deviation", "changed_mind"),
    ("intended_deceptive_complied", "unclear"),
    ("unknown", "errors"),
]

for path in sorted(
    glob.glob("./data/meta-llama_llama-4-maverick_endogenous_sprint.json")
):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    c = Counter(
        a["typology"]
        for r in data["rounds"]
        for t in r["trials"]
        for a in t["agents"].values()
    )
    game = path.replace("\\", "/").split("/")[-3]
    run = os.path.basename(path).replace("_endogenous", "").replace(".json", "")
    counts = "  ".join(f"{label}={c.get(k, 0)}" for k, label in KINDS)
    print(f"{game:12} {run:38} N={sum(c.values()):<4} {counts}")
