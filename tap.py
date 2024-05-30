#!/usr/bin/env python3
"""
Parley: A Tree of Attacks (TAP) LLM Jailbreaking Implementation
"""
import argparse
import copy
import functools
import typing as t
import re
from typing import Annotated

from parley._types import (
    ChatFunction,
    Message,
    Parameters,
    Role,
    Conversation,
    Feedback,
    TreeNode,
)
from parley.models import chat_mistral, chat_openai, chat_together
from parley.prompts import (
    get_prompt_for_evaluator_score,
    get_prompt_for_evaluator_on_topic,
    get_prompt_for_attacker,
    get_prompt_for_target,
)

Models: t.Dict[str, t.Tuple] = {
    "gpt-3.5": (chat_openai, "gpt-3.5-turbo"),
    "gpt-4": (chat_openai, "gpt-4"),
    "gpt-4-turbo": (chat_openai, "gpt-4-1106-preview"),
    "llama-13b": (chat_together, "togethercomputer/llama-2-13b-chat"),
    "llama-70b": (chat_together, "togethercomputer/llama-2-70b-chat"),
    "vicuna-13b": (chat_together, "lmsys/vicuna-13b-v1.5"),
    "mistral-small-together": (chat_together, "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    "mistral-small": (chat_mistral, "mistral-small"),
    "mistral-medium": (chat_mistral, "mistral-medium"),
}


def load_models(
     target_model: Annotated[str, "Target model"] = "gpt-4-turbo",
     target_temp: Annotated[float, "Target temperature"] = 0.3,
     target_top_p: Annotated[float, "Target top-p"] = 1.0,
     target_max_tokens: Annotated[int, "Target max tokens"] = 1024,
     evaluator_model: Annotated[str, "Evaluator model"] = "gpt-4-turbo",
     evaluator_temp: Annotated[float, "Evaluator temperature"] = 0.5,
     evaluator_top_p: Annotated[float, "Evaluator top-p"] = 0.1,
     evaluator_max_tokens: Annotated[int, "Evaluator max tokens"] = 10,
     attacker_model: Annotated[str, "Attacker model"] = "mistral-small",
     attacker_temp: Annotated[float, "Attacker temperature"] = 1.0,
     attacker_top_p: Annotated[float, "Attacker top-p"] = 1.0,
     attacker_max_tokens: Annotated[int, "Attacker max tokens"] = 1024,
) -> t.Tuple[ChatFunction, ChatFunction, ChatFunction]:
    target_func, target_model = Models[target_model]
    target_chat = t.cast(
        ChatFunction,
        functools.partial(
            target_func,
            parameters=Parameters(
                model=target_model,
                temperature=target_temp,
                top_p=target_top_p,
                max_tokens=target_max_tokens,
            ),
        ),
    )

    evaluator_func, evaluator_model = Models[evaluator_model]
    evaluator_chat = t.cast(
        ChatFunction,
        functools.partial(
            evaluator_func,
            parameters=Parameters(
                model=evaluator_model,
                temperature=evaluator_temp,
                top_p=evaluator_top_p,
                max_tokens=evaluator_max_tokens,
            ),
        ),
    )

    attacker_func, attacker_model = Models[attacker_model]
    attacker_chat = t.cast(
        ChatFunction,
        functools.partial(
            attacker_func,
            parameters=Parameters(
                model=attacker_model,
                temperature=attacker_temp,
                top_p=attacker_top_p,
                max_tokens=attacker_max_tokens,
            ),
        ),
    )

    return target_chat, evaluator_chat, attacker_chat

def cosine_similarity(l1: t.List[float], l2: t.List[float]) -> float:
    dot_product = sum(a*b for a, b in zip(l1, l2))
    magnitude_l1 = sum(a**2 for a in l1) ** 0.5
    magnitude_l2 = sum(b**2 for b in l2) ** 0.5
    
    try:
        similarity = dot_product / (magnitude_l1 * magnitude_l2)
    except ZeroDivisionError:
        similarity = 0

    return similarity

def score(system_prompt: str, chat: ChatFunction, prompt: str, response: str) -> int:
    score_str = chat(
        [
            Message(role=Role.system, content=system_prompt),
            Message(
                role=Role.user,
                content=f"Prompt (P): {prompt}\nResponse (R): {response}",
            ),
        ]
    ).content

    match = re.search(r"(\d+)", score_str)
    assert match is not None, f"Could not parse score from response: {score_str}"
    return int(match.string)


def check_on_topic(system_prompt: str, chat: ChatFunction, content: str) -> bool:
    on_topic_str = chat(
        [
            Message(role=Role.system, content=system_prompt),
            Message(role=Role.user, content=content),
        ]
    ).content

    match = re.search(r"(yes|no)", on_topic_str.lower())
    assert match is not None, f"Could not parse yes/no from response: {on_topic_str}"
    return True if match.string == "yes" else False


def attack(
    chat: ChatFunction,
    conversation: Conversation,
) -> t.Optional[Feedback]:
    next_response = chat(conversation)

    match = re.search(r"\{.*?\}", next_response.content, re.DOTALL)
    assert (
        match is not None
    ), f"Could not parse JSON from response: {next_response.content}"

    try:
        return Feedback.model_validate_json(match.string)
    except Exception:
        return None

def call(goal: Annotated[str, "goal"],
         target_chat: Annotated[ChatFunction, "Target Chat caller"],
         evaluator_chat: Annotated[ChatFunction, "Evaluator Chat caller"],
         attacker_chat: Annotated[ChatFunction, "Attacker Chat caller"],
         root_nodes: Annotated[int, "Tree of thought root node count"] = 3,
         branching_factor: Annotated[int, "Tree of thought branching factor"] = 3,
         width: Annotated[int, "Tree of thought width"] = 10,
         depth: Annotated[int, "Tree of thought depth"] = 10,
         stop_score: Annotated[int, "Stop when the score is above this value"] = 8):

    attacker_system_prompt = get_prompt_for_attacker(goal)
    scoring_system_prompt = get_prompt_for_evaluator_score(goal)
    on_topic_system_prompt = get_prompt_for_evaluator_on_topic(goal)
    target_system_prompt = get_prompt_for_target()
    
    root_nodes: t.List[TreeNode] = [
        TreeNode(
            children=[],
            conversation=[Message(role=Role.system, content=attacker_system_prompt)],
            feedback=None,
            response=None,
            on_topic=None,
            score=None,
        )
        for _ in range(root_nodes)
    ]

    current_nodes: t.List[TreeNode] = root_nodes

    print("[+] Beginning TAP ...")
    for iteration in range(depth):
        print(f" |- Iteration {iteration + 1} with {len(current_nodes)} nodes ...")

        for i, node in enumerate(current_nodes):
            # 1 - Prepare the next conversation step

            response_str = (
                f"{node.response[:300]}..."
                if node.response is not None
                else "[Ignore, this is your first attempt]"
            )
            score_str = (
                node.score
                if node.score is not None
                else "[Ignore, this is your first attempt]"
            )
            next_message = Message(
                role=Role.user,
                content=f"Goal (G): {goal}\nResponse (R): {response_str}\nScore (S): {score_str}",
            )

            node.conversation.append(next_message)

            # 2 - Branch out

            for _ in range(branching_factor):
                feedback = attack(attacker_chat, node.conversation)
                if feedback is None:
                    print("  |> Attack generation failed")
                    continue

                conversation = copy.deepcopy(node.conversation)
                conversation.append(
                    Message(role=Role.assistant, content=feedback.model_dump_json())
                )

                node.children.append(
                    TreeNode(
                        children=[],
                        conversation=conversation,
                        feedback=feedback,
                        response=None,
                        on_topic=None,
                        score=None,
                    )
                )

            # 3 - Perform the inference + evaluations

            for k, child in enumerate(node.children):
                assert child.feedback is not None

                print(f"  |= {iteration + 1}->{i + 1}->{k + 1}")
                print(f'   |- Improvement: "{child.feedback.improvement[:100]}"')
                print(f'   |- Prompt:      "{child.feedback.prompt[:100]}"')

                child.on_topic = check_on_topic(
                    on_topic_system_prompt, evaluator_chat, child.feedback.prompt
                )
                if not child.on_topic:
                    print("   |- Off topic.")
                    continue

                print("   |- ON topic ...")

                # Add any additional prompt alterations here
                prompt = child.feedback.prompt

                child.response = target_chat(
                    [
                        Message(role=Role.system, content=target_system_prompt),
                        Message(role=Role.user, content=prompt),
                    ]
                ).content

                child.score = score(
                    scoring_system_prompt,
                    evaluator_chat,
                    child.feedback.prompt,
                    child.response,
                )

                print(f'   |- Response:    "{child.response[:100]}"')
                print(f"   |- Score:       {child.score}")
                
                if child.score >= stop_score:
                    print()
                    print("[+] Found a good attack!")
                    print(f" |- Prompt: {child.feedback.prompt}")
                    print(f" |- Response: {child.response}")
                    print()
                    return child.feedback.prompt

        # 4 - Prune the tree and step down

        children = [
            child for node in current_nodes for child in node.children if child.on_topic
        ]
        children.sort(
            key=lambda x: (x.score if x.score is not None else float("-inf")),
            reverse=True,
        )

        current_nodes = children[: width]

        if len(current_nodes) == 0:
            print()
            print("[!] No more nodes to explore")
            print()
            return []