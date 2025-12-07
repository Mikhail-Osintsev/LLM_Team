from typing import TypedDict, List, Tuple, Literal, Dict, Any, Optional

import os
import json
from pprint import pprint

from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI

from app.backend.core.generator import generate_answer_from_passages
from app.backend.core.mcp_tools import (
    list_tools_for_llm,
    call_tool_from_llm,
)


class RAGState(TypedDict, total=False):
    question: str
    top_k: int
    passages: List[Tuple[str, float]]
    answer: str
    decision: Literal["tool", "answer"]
    tool_name: str
    tool_args: Dict[str, Any]
    tool_calls: int
    messages: List[Dict[str, str]]


LLM = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.environ.get("MISTRAL_API_KEY"),
)

TOOLS = list_tools_for_llm()
TOOLS_SCHEMA_STR = json.dumps(TOOLS, ensure_ascii=False, indent=2)
MAX_TOOL_CALLS = 2


def _parse_json_from_model(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            middle = parts[1].lstrip()
            if middle.lower().startswith("json"):
                middle = middle[4:].lstrip()
            raw = middle
    return json.loads(raw)


def planner_node(state: RAGState) -> Dict[str, Any]:
    question = state["question"]
    messages: List[Dict[str, str]] = state.get("messages", [])

    if not messages:
        system_prompt = """Ты ассистент, который профессионально отвечает на вопросы по книге.
У тебя есть инструменты, описанные JSON-схемой ниже.
Ты ДОЛЖЕН использовать эту схему, когда формируешь tool_args.

ОБЯЗАТЕЛЬНО: на самом первом шаге ты всегда сначала вызываешь инструмент (например, retrieve),
чтобы получить фрагменты книги. Никогда не отвечай сразу на первом шаге.

Доступные инструменты (JSON-схема):
{tools_schema}


Если для корректного ответа нужны данные из книги, обязательно используй инструмент.
Если данных достаточно, можешь сразу ответить.

Ответ возвращай строго в формате JSON без лишнего текста.

1) Если нужно вызвать инструмент:
{{
  "decision": "tool",
  "tool_name": "<имя_инструмента>",
  "tool_args": {{ ... }}
}}

2) Если можешь ответить сразу:
{{
  "decision": "answer",
}}
""".format(tools_schema=TOOLS_SCHEMA_STR)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вопрос пользователя: {question!r}"},
        ]
    else:
        followup_user = {
            "role": "user",
            "content": """Ты видишь выше историю диалога и результаты вызовов инструментов.
Теперь реши, нужно ли ещё раз вызвать какой-либо инструмент, 
или можно дать финальный ответ пользователю.

Отвечай строго в формате JSON в одном объекте, без ``` и лишнего текста.

1) Вызов инструмента:
{
  "decision": "tool",
  "tool_name": "<имя_инструмента>",
  "tool_args": { ... }
}

2) Финальный ответ:
{
  "decision": "answer",
}
""",
        }
        messages = messages + [followup_user]

    resp = LLM.invoke(messages)
    content = getattr(resp, "content", resp)

    if isinstance(content, str):
        raw = content
    else:
        try:
            raw = content[0].get("text", "")
        except Exception:
            raw = str(content)

    data = _parse_json_from_model(raw)

    messages = messages + [{"role": "assistant", "content": raw}]

    decision = data.get("decision", "tool")

    if decision == "answer":
        return {
            "decision": "answer",
            "messages": messages,
        }

    tool_name = data.get("tool_name", "retrieve")
    tool_args = data.get("tool_args") or {}
    tool_args["query"] = question
    tool_args["top_k"] = state.get("top_k", 4)

    return {
        "decision": "tool",
        "tool_name": tool_name,
        "tool_args": tool_args,
        "messages": messages,
    }


def tools_node(state: RAGState) -> Dict[str, Any]:
    tool_name = state.get("tool_name", "retrieve")
    tool_args = state.get("tool_args", {}) or {}
    current_calls = state.get("tool_calls", 0)
    messages = state.get("messages", [])

    try:
        result = call_tool_from_llm(tool_name, tool_args)
    except KeyError:
        new_messages = messages + [
            {
                "role": "assistant",
                "content": f"Инструмент {tool_name!r} не найден. "
                           f"Пробую продолжить без него.",
            }
        ]
        return {
            "tool_calls": current_calls + 1,
            "messages": new_messages,
            "passages": [],
        }

    passages = result.get("passages")

    if passages:
        # Поддержка нового формата с метаданными (text, score, metadata)
        passage_texts = []
        for passage_data in passages:
            if len(passage_data) == 3:
                text, score, metadata = passage_data
                book_name = metadata.get("book_name", "")
                page_number = metadata.get("page_number", 0)
                if book_name and page_number:
                    passage_texts.append(f"[score={score:.3f}, {book_name}, стр. {page_number}]\n{text}")
                else:
                    passage_texts.append(f"[score={score:.3f}]\n{text}")
            else:
                # Старый формат (text, score)
                text, score = passage_data
                passage_texts.append(f"[score={score:.3f}]\n{text}")

        full_text = "\n\n---\n\n".join(passage_texts)
        tool_msg = (
            f"Результат инструмента {tool_name!r} с аргументами {tool_args!r}:\n"
            f"{full_text}"
        )
    else:
        tool_msg = f"Инструмент {tool_name!r} вернул результат: {result!r}"

    new_messages = messages + [
        {"role": "assistant", "content": tool_msg}
    ]

    if "passages" not in result:
        result["passages"] = []

    result["tool_calls"] = current_calls + 1
    result["messages"] = new_messages

    return result


def generate_node(state: RAGState) -> Dict[str, Any]:
    question = state["question"]
    passages = state.get("passages", [])

    answer = generate_answer_from_passages(
        llm=LLM,
        question=question,
        passages=passages,
    )

    return {
        "answer": answer,
    }


def route_after_plan(state: RAGState) -> str:
    calls = state.get("tool_calls", 0)
    decision = state.get("decision", "tool")

    if calls == 0:
        return "tools"

    if decision == "answer":
        return "generate"

    if calls >= MAX_TOOL_CALLS:
        return "generate"

    return "tools"


_graph = StateGraph(RAGState)

_graph.add_node("plan", planner_node)
_graph.add_node("tools", tools_node)
_graph.add_node("generate", generate_node)

_graph.add_edge(START, "plan")

_graph.add_conditional_edges(
    "plan",
    route_after_plan,
    {
        "tools": "tools",
        "generate": "generate",
    },
)

_graph.add_edge("tools", "plan")
_graph.add_edge("generate", END)

rag_graph = _graph.compile()


def run_rag(question: str, top_k: int = 4) -> RAGState:
    initial_state: RAGState = {
        "question": question,
        "top_k": top_k,
        "tool_calls": 0,
    }

    final_state: Optional[RAGState] = None

    for state in rag_graph.stream(initial_state, stream_mode="values"):
        print("=== STATE ===")
        pprint(state)
        final_state = state

    if final_state is None:
        raise RuntimeError("Graph did not produce any state")

    return final_state