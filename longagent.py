import numpy as np
import re

def split_into_chunks(doc: str, target_chunks: int) -> list:
    # Split by passage headers like "Passage 0:", "Passage 1:", etc.
    passages = re.split(r'Passage \d+:', doc)
    # Remove empty strings and strip whitespace
    passages = [p.strip() for p in passages if p.strip()]
    print("Num of Passages: ", len(passages), "Target num Chunks:", target_chunks)
    chunks = []
    for passage in passages:
        tokens = passage.split()
        print("Num of tokens in passage: ", len(tokens))
        # if target chunks < num of passages, each chunk is a passage
        max_tokens_per_chunk = len(tokens) // (int(np.ceil(target_chunks/len(passages))))
        print(f"Max tokens per chunk: {max_tokens_per_chunk}")
        # Further split each passage into sub-chunks of max_tokens_per_chunk
        for i in range(0, len(tokens), max_tokens_per_chunk):
            chunk = ' '.join(tokens[i:i+max_tokens_per_chunk])
            if chunk:
                chunks.append(chunk)
    return chunks

from gemini_model import gemini_predict
from typing import List, Dict, Any
import re
from difflib import SequenceMatcher

STOPWORDS = set('the a an and or but if while with for to of in on at by from as is are was were be been being has have had do does did not no nor so than too very can will just'.split())

def remove_stopwords(text):
    return ' '.join([w for w in re.findall(r'\w+', text.lower()) if w not in STOPWORDS])

def normalized_edit_distance(a, b):
    return 1 - SequenceMatcher(None, a, b).ratio()

def simple_conflict_detector(answers, threshold=0.4):
    filtered = [remove_stopwords(ans) for ans in answers if ans.strip()]
    if len(filtered) <= 1:
        return False
    for i in range(len(filtered)):
        for j in range(i+1, len(filtered)):
            if normalized_edit_distance(filtered[i], filtered[j]) > threshold:
                return True
    return False

def call_gemini(prompt: str) -> str:
    """
    Wrapper for Gemini model call. Returns the model's response as a string.
    """
    return gemini_predict(prompt)

def member_agent(chunk: str, prompt_template: str) -> str:
    """
    Processes a single chunk using the provided prompt template and Gemini model.
    Args:
        chunk (str): The text chunk to process.
        prompt_template (str): The prompt template, should contain a placeholder for the chunk (e.g., {chunk}).
    Returns:
        str: The Gemini model's response for this chunk.
    """
    prompt = prompt_template.format(chunk=chunk)
    return call_gemini(prompt)

def leader_agent(member_outputs: list, leader_prompt_template: str) -> str:
    """
    Aggregates Member agent outputs and synthesizes a final answer using the Gemini model.
    Args:
        member_outputs (list): List of strings, each the output from a Member agent.
        leader_prompt_template (str): Prompt template for the Leader, should contain a placeholder for the member outputs (e.g., {member_outputs}).
    Returns:
        str: The Gemini model's synthesized answer.
    """
    joined_outputs = "\n".join(member_outputs)
    prompt = leader_prompt_template.format(member_outputs=joined_outputs)
    return call_gemini(prompt)

def collaborative_long_agent_pipeline(
    doc: str,
    member_prompt_template: str,
    leader_prompt_template: str,
    target_chunks: int = 4,
    max_rounds: int = 5
) -> (str, bool):
    """
    Implements the collaborative reasoning loop and conflict resolution as described in detailed_logic.md.
    Returns:
        final_answer (str): The synthesized answer.
        conflict_resolution_failed (bool): True if consensus was not reached within max_rounds.
    """
    # Step 1: Chunking
    chunks = split_into_chunks(doc, target_chunks)
    member_chunks = {i: chunk for i, chunk in enumerate(chunks)}
    # print(member_chunks)
    member_outputs = {}
    dialogue_history = []
    state = "NEW_STATE"
    round_num = 0
    conflict_resolution_failed = False

    while state != "ANSWER" and round_num < max_rounds:
        print(f"\n--- Reasoning Round {round_num+1} ---")
        member_outputs = {}
        for i, chunk in member_chunks.items():
            prompt = member_prompt_template.format(chunk=chunk)
            answer = call_gemini(prompt)
            member_outputs[i] = answer
        dialogue_history.append({"round": round_num, "member_outputs": member_outputs.copy()})
        answers = list(member_outputs.values())
        # Use simple conflict detection
        if any(a.strip() == "" for a in answers):
            print("Some members returned empty answers. Entering NEW_STATE.")
            state = "NEW_STATE"
        elif not simple_conflict_detector(answers):
            print("No conflict detected. All members agree.")
            print("Member outputs:", answers)
            state = "ANSWER"
            break
        else:
            print(f"Conflict detected in round {round_num+1}. Entering CONFLICT state.")
            print("Member outputs:", answers)
            state = "CONFLICT"
        
        # Step 4: Conflict resolution (inter-member communication)
        if state == "CONFLICT":
            # Find clusters of conflicting answers
            answer_to_ids = {}
            for idx, ans in member_outputs.items():
                key = ans.strip().lower()
                answer_to_ids.setdefault(key, []).append(idx)
            # Only handle pairwise conflict for simplicity
            if len(answer_to_ids) == 2:
                ids1, ids2 = list(answer_to_ids.values())
                # Merge chunks for each cluster and re-query
                merged1 = " ".join([member_chunks[i] for i in ids1 + ids2])
                merged2 = merged1  # Both clusters get the same merged evidence
                for i in ids1 + ids2:
                    prompt = member_prompt_template.format(chunk=merged1)
                    member_outputs[i] = call_gemini(prompt)
                # Update member_chunks to reflect merged evidence
                for i in ids1 + ids2:
                    member_chunks[i] = merged1
                # Continue loop
            else:
                # If more than 2 clusters, merge all
                merged = " ".join([member_chunks[i] for i in member_chunks])
                for i in member_chunks:
                    prompt = member_prompt_template.format(chunk=merged)
                    member_outputs[i] = call_gemini(prompt)
                    member_chunks[i] = merged
        round_num += 1

    if state != "ANSWER":
        print(f"Conflict was not resolved after {max_rounds} rounds.")
        conflict_resolution_failed = True

    # Step 5: Synthesize final answer
    final_outputs = list(member_outputs.values())
    prompt = leader_prompt_template.format(member_outputs="\n".join(final_outputs))
    final_answer = call_gemini(prompt)
    return final_answer, conflict_resolution_failed

def test_collaborative_long_agent_pipeline():
    """
    Test the collaborative_long_agent_pipeline using the first example from needle_plus_hotpotqa.jsonl.
    """
    # Extracted from the first line of the file
    context = '''Passage 0:
Noelle Scaggs
Noelle Scaggs (born October 8, 1979) is an American musician and singer-songwriter from Los Angeles. For ten years she served as front-woman for soul band The Nirvana, and has also collaborated as a composer or vocalist with artists such as The Black Eyed Peas, Dilated Peoples, Quantic, Mayer Hawthorne, Defari, and Damian Marley.
Passage 0:
Fitz and The Tantrums
Fitz and The Tantrums (FATT) is an American indie pop and neo soul band from Los Angeles that formed in 2008. The band consists of Michael Fitzpatrick (lead vocals), Noelle Scaggs (co-lead vocals and percussion), James King (saxophone, flute, keyboard, percussion and guitar), Joseph Karnes (bass guitar), Jeremy Ruzumna (keyboards) and John Wicks (drums and percussion). Their debut studio album, "Pickin' Up the Pieces", was released in August 2010 on indie label Dangerbird Records and received critical acclaim. It reached No. 1 on the "Billboard" Heatseekers chart. The band signed to their current label Elektra Records in early 2013 and went on to release their sophomore LP, "More Than Just a Dream," on May 7, 2013. Their self-titled third album was released on June 10, 2016.
Passage 1:
Shaker communities
After the Shakers arrived in the United States in 1774, they established numerous communities in the late-18th century through the entire 19th century. The first villages organized in Upstate New York and the New England states, and, through Shaker missionary efforts, Shaker communities appeared in the Midwestern states. Communities of Shakers were governed by area bishoprics and within the communities individuals were grouped into "family" units and worked together to manage daily activities. By 1836 eighteen major, long-term societies were founded, comprising some sixty families, and many smaller, short-lived communities were established over the course of the 19th century, including two failed ventures into the Southeastern United States and an urban community in Philadelphia, Pennsylvania. The Shakers peaked in population by the early 1850s. With the turmoil of the American Civil War and subsequent Industrial Revolution, Shakerism went into severe decline, and as the number of living Shakers diminished, Shaker villages ceased to exist. Some of their buildings and sites have become museums, and many are historic districts under the National Register of Historic Places. The only active community is Sabbathday Lake Shaker Village in Maine.
Passage 2:
Larry Elin
Larry Elin is an associate professor in the Television, Radio, Film department at the S.I Newhouse School of Public Communications at Syracuse University. He teaches media business, interactive media, and animation and special effects. He started his career, however, as an animator at Mathematical Applications Group, Inc., in Elmsford, NY, in 1973, one of the first 3-D computer animation companies. By 1980, Elin had become head of production, and hired Chris Wedge, who later founded Blue Sky Studios, among others. Elin and Wedge were the key animators on MAGI's work on the feature film "Tron", which included the Lightcycle, Recognizer, and Tank sequences. Elin later became executive producer at Kroyer Films, which produced the animation for .
Passage 3:
Scott Special
The Scott Special, also known as the Coyote Special, the Death Valley Coyote or the Death Valley Scotty Special, was a one-time, record-breaking (and the best-known) passenger train operated by the Atchison, Topeka and Santa Fe Railway (Santa Fe) from'''
    query = "For which band, was the female member of Fitz and The Tantrums, the front woman for ten years ?"
    gold_answer = "The Nirvana"

    # Prompt templates
    member_prompt_template = (
        "Given the following passage chunk, answer the question as best as possible.\n"
        "Chunk:\n{chunk}\n"
        "Question: " + query + "\n"
        "If the answer is not in the chunk, say 'Not found in this chunk.'"
    )
    leader_prompt_template = (
        "Given the following answers from different document chunks, synthesize a final answer to the question.\n"
        "Member outputs:\n{member_outputs}\n"
        "Question: " + query + "\n"
        "If there is a clear, correct answer, state it. If not, explain why."
    )

    # Run the collaborative pipeline
    final_answer, conflict_resolution_failed = collaborative_long_agent_pipeline(
        context,
        member_prompt_template,
        leader_prompt_template,
        max_rounds=3
    )
    print("Gold answer:", gold_answer)
    print("Final answer:", final_answer)
    print("Conflict resolution failed:", conflict_resolution_failed)

def test_conflict_resolution():
    """
    Test the collaborative_long_agent_pipeline with a synthetic example that triggers conflict resolution.
    """
    context = (
        "Passage 1: Alice was the CEO of TechCorp for ten years. "
        "Passage 2: Bob was the CEO of TechCorp for ten years."
    )
    query = "Who was the CEO of TechCorp for ten years?"
    gold_answer = "Conflict: Both Alice and Bob were claimed to be the CEO for ten years."

    member_prompt_template = (
        "Given the following passage chunk, answer the question as best as possible.\n"
        "Chunk:\n{chunk}\n"
        "Question: " + query + "\n"
        "If the answer is not in the chunk, say 'Not found in this chunk.'"
    )
    leader_prompt_template = (
        "Given the following answers from different document chunks, synthesize a final answer to the question.\n"
        "Member outputs:\n{member_outputs}\n"
        "Question: " + query + "\n"
        "If there is a clear, correct answer, state it. If not, explain why."
    )

    final_answer, conflict_resolution_failed = collaborative_long_agent_pipeline(
        context,
        member_prompt_template,
        leader_prompt_template,
        max_rounds=3
    )
    print("Gold answer:", gold_answer)
    print("Final answer:", final_answer)
    print("Conflict resolution failed:", conflict_resolution_failed)
