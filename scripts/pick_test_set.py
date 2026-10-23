import json, random
agents = []

with open('./data/meta-llama_llama-4-maverick_endogenous_sprint.json', 'r') as f:
    data = json.load(f)


for rnd in data['rounds']:
    for trial in rnd['trials']:
        trial_id = trial['trial_id']
        for name, agent in trial['agents'].items():
            agent_data = {
                    'id': f'sprint-t{trial_id}-{name}',
                    'typology': agent['typology'],
                    'intended_action': agent['stage1']['intended_action'], # secret plan
                    'reasoning_s1': agent['stage1']['reasoning'], # plan notes
                    'stated_action': agent['stage2']['stated_action'], # public action announcement
                    'message': agent['stage2']['message'], # public announcement message 
                    'final_choice': agent['stage3']['choice'], # final choice
                    'reasoning_s3': agent['stage3']['reasoning'], # final notes 
                }
            agents.append(agent_data)

liars = [agent for agent in agents if agent["typology"] == "premeditated_deception" and agent['final_choice'] == agent['intended_action']]

honest = [agent for agent in agents if agent["typology"] == "fully_honest"]

mind_changers = [agent for agent in agents if agent["typology"] == "impulsive_deviation"]

random.seed(42)

random.shuffle(liars)
random.shuffle(honest)
random.shuffle(mind_changers)

print(f'Liars: {len(liars)}/{len(agents)}')
print(f"Honest: {len(honest)}/{len(agents)}")
print(f"Mind changers: {len(mind_changers)}/{len(agents)}")
print(
    f"Total agents Excluded are: {len(agents) - (len(liars) + len(honest) + len(mind_changers))}"
)


with open('./data/liar_candidates.txt', 'w', encoding='utf-8') as f:
    for i, agent in enumerate(liars, 1):
        text = f"""
        #{i} {agent['id']}
        plan {agent['intended_action']} -> announced {agent['stated_action']} -> final choice {agent['final_choice']}\n
        Plan Notes: {agent['reasoning_s1']}\n
        Final Notes: {agent['reasoning_s3']}\n
        ==========================================================
        """
        f.write(f"{text}\n")

with open('./data/rejected_liars.txt', 'r') as f:
    rejected_ids = set()
    for line in f.read().split("\n"):
        rejected_ids.add(line.strip())

new_liars = [liar for liar in liars if liar['id'] not in rejected_ids]

selected_liars = new_liars[:30]
selected_honest = honest[:15]
selected_mind_changers = mind_changers[:15]

test_set = []

for kind, group in [
    ("liar", selected_liars),
    ("honest", selected_honest),
    ("mind_changer", selected_mind_changers),
]:
    for item in group:
        item['kind'] = kind
        item["label"] = 1 if kind == 'liar' else 0
        item["views"] = {
            "A": f'Public Announcement: "{item["message"]}"',
            "B": f"Final choice: {item['final_choice']}",
            "C": f'Public Announcement: "{item['message']}"'
            + "\n"
            + f"Final choice: {item['final_choice']}",
            "D": f"Announced: {item['stated_action']}, Final choice: {item['final_choice']}",
        }

        test_set.append(item)
