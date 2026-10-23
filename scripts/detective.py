import json, requests, os, re, random, time
from dotenv import load_dotenv

# loading the env file, override=True uses the key from env file in case if the if the system environment also has an key present
load_dotenv(override=True)

KEY = os.getenv("SARVAM_API_KEY", "") # fetching the key from the env file.
print(f"API Key found?: {bool(KEY)}")

# Reading the detetctive prompt from the text file.
with open("./detective_prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

# API call to the detective LLM (SARVAM-105B)
def ask_detective(evidence):
    filled = prompt.replace("{EVIDENCE}", evidence)
    r = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={"api-subscription-key": KEY},
        json={
            "model": "sarvam-105b",
            "temperature": 0,
            "max_tokens": 8000,
            "reasoning_effort": None, # reasoning disabled
            "messages": [{"role": "user", "content": filled}],
        },
        timeout=120,
    )

    r.raise_for_status()
    choice = r.json()["choices"][0]
    return choice["message"]["content"], choice["finish_reason"]

# Extracting the verdict score from the detective output
def get_verdict(reply):
    if not reply:
        return None
    
    # detective LLM was explicitly asked to put the verdict score inside <verdict> SCORE </verdict>
    PATTERN = r"<verdict>\s*(\d+(?:\.\d+)?)\s*</verdict>"

    matches = re.findall(PATTERN, reply)

    if not matches:
        return None

    return float(matches[-1])

# Loading the test data set prepared from the experiment, it consist of 30 liar agents, 15 honest agents, and 15 mind changer agents.
with open('./data/test_set.json', 'r', encoding='utf-8') as f:
    test_set = json.load(f)

calls = [(item, view) for item in test_set for view in ['A', 'B', 'C', 'D']]

random.seed(42)
random.shuffle(calls)


done = set()

if os.path.exists('./data/judge_verdict.jsonl'):
    with open('./judge_verdict.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            done.add((row['id'], row['view'])) in done

count = 1
for item, view in calls:
    if (item["id"], view) in done:
        continue
    
    evidence = item['views'][view]
    try:
        reply, reason = ask_detective(evidence)
    except Exception as e:
        print("Error:", item['id'], view, e)
        time.sleep(10)
        continue
    score = get_verdict(reply)

    row = {
        'id': item['id'],
        'kind': item['kind'],
        'label': item['label'],
        'view': view,
        'score': score,
        'finish_reason': reason,
        'reply': reply 
    }

    with open('./data/judge_verdict.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f"Completed {count}/{len(calls)}: {item['id']} view {view} = verdict {score}")
    count = count + 1
