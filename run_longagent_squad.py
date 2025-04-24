import json
import jsonlines
from longagent import collaborative_long_agent_pipeline
import os

def main():
    input_path = "NeedleInAHaystack-PLUS/needle_plus_squad.jsonl"
    output_path = "NeedleInAHaystack-PLUS/needle_plus_squad_predictions.jsonl"
    member_prompt_template = (
        "Given the following passage chunk, answer the question as best as possible.\n"
        "Chunk:\n{chunk}\n"
        "Question: {query}\n"
        "If the answer is not in the chunk, say 'Not found in this chunk.'"
    )
    leader_prompt_template = (
        "Given the following answers from different document chunks, synthesize a final answer to the question. Only state the answer without explanation\n"
        "Member outputs:\n{member_outputs}\n"
        "Question: {query}\n"
    )
    completed_ids = set()
    if os.path.exists(output_path):
        with jsonlines.open(output_path) as reader:
            for item in reader:
                completed_ids.add(item["id"])
    with jsonlines.open(input_path) as reader, jsonlines.open(output_path, mode='a') as writer:
        for item in reader:
            if item["id"] in completed_ids:
                continue
            context = item["context"]
            query = item["input"]
            member_prompt = member_prompt_template.format(chunk="{chunk}", query=query)
            leader_prompt = leader_prompt_template.format(member_outputs="{member_outputs}", query=query)
            prediction, conflict_resolution_failed = collaborative_long_agent_pipeline(
                context,
                member_prompt,
                leader_prompt,
                target_chunks=4, 
                max_rounds=3
            )
            item["prediction"] = prediction
            item["conflict_resolution_failed"] = int(conflict_resolution_failed)
            writer.write(item)
    print(f"Predictions written to {output_path}")

if __name__ == "__main__":
    main()
