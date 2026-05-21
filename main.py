"""
StudyBuddy v2 — RAG-powered agentic study assistant.

Upgrades over v1:
- Real RAG pipeline (AST chunking + dual retrieval + RRF fusion)
- LLM-grounded quiz and summarization (no more fake truncation)
- System prompt with persona
- .env for API key management
- Persistent vector store (ChromaDB)
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from litellm import completion

# Load API keys from .env — never hardcode keys again
load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "groq/llama-3.1-8b-instant")

SYSTEM_PROMPT = """You are StudyBuddy, an intelligent AI study assistant.

You help students by:
- Answering questions grounded in their uploaded study material (rag_answer)
- Generating quizzes based on actual course content (generate_quiz)
- Summarizing topics from indexed notes (summarize_topic)
- Managing a study task list (add_task, get_tasks)
- Ingesting new study files on demand (ingest_material)
- Showing what material is already indexed (list_sources)

Always prefer using rag_answer for factual questions if material has been indexed.
Be encouraging, concise, and cite sources when available.
"""

# ─── Core LLM call ────────────────────────────────────────────────────────

def generate_response(messages, tools=None):
    return completion(
        model=DEFAULT_MODEL,
        messages=messages,
        tools=tools,
        max_tokens=1024
    )


# ─── Goal class (unchanged) ───────────────────────────────────────────────

class Goal:
    def __init__(self, priority, name, description):
        self.priority = priority
        self.name = name
        self.description = description


# ─── Simple tools (unchanged from v1) ────────────────────────────────────

TASKS = []

def add_task(task: str) -> str:
    TASKS.append(task)
    return f"Task added: '{task}'. You now have {len(TASKS)} task(s)."

def get_tasks() -> str:
    if not TASKS:
        return "No tasks yet. Add one with 'Add task: ...'."
    return "\n".join(f"{i+1}. {t}" for i, t in enumerate(TASKS))

def terminate(message: str) -> str:
    return message


# ─── Action / Registry (unchanged architecture, extended tools) ───────────

class Action:
    def __init__(self, name, function, description, parameters, terminal=False):
        self.name = name
        self.function = function
        self.description = description
        self.parameters = parameters
        self.terminal = terminal


class ActionRegistry:
    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action

    def get_tools_schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": a.name,
                    "description": a.description,
                    "parameters": a.parameters
                }
            }
            for a in self.actions.values()
        ]


# ─── Agent (upgraded with system prompt + better memory) ─────────────────

class Agent:
    def __init__(self, goals, action_registry):
        self.goals = goals
        self.registry = action_registry
        # System prompt as first message — fixes the missing persona bug
        self.memory: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def run(self, user_input: str, max_iterations: int = 10):
        self.memory.append({"role": "user", "content": user_input})

        for iteration in range(max_iterations):
            response = generate_response(
                messages=self.memory,
                tools=self.registry.get_tools_schema()
            )
            message = response.choices[0].message

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                print(f"\n[Tool] {tool_name}({', '.join(f'{k}={repr(v)}' for k, v in args.items())})")

                action = self.registry.actions.get(tool_name)
                if not action:
                    print(f"[Error] Unknown tool: {tool_name}")
                    break

                result = action.function(**args) if args else action.function()
                print(f"[Result] {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}")

                # Append assistant tool call + tool result to memory
                self.memory.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": message.tool_calls
                })
                self.memory.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

                if action.terminal:
                    print(f"\nStudyBuddy: {result}")
                    break

            else:
                content = message.content or ""
                print(f"\nStudyBuddy: {content}")
                self.memory.append({"role": "assistant", "content": content})
                break


# ─── Registry builder ─────────────────────────────────────────────────────

def build_registry() -> ActionRegistry:
    from tools.rag_tools import (
        ingest_material, rag_answer, generate_quiz,
        summarize_topic, list_sources
    )

    registry = ActionRegistry()

    registry.register(Action(
        name="ingest_material",
        function=ingest_material,
        description="Ingest a file or folder into the study knowledge base. Supports .py, .pdf, .txt, .md files.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file or directory to index"}
            },
            "required": ["path"]
        }
    ))

    registry.register(Action(
        name="rag_answer",
        function=rag_answer,
        description="Answer a question using the indexed study material with source citations.",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The student's question"}
            },
            "required": ["question"]
        }
    ))

    registry.register(Action(
        name="generate_quiz",
        function=generate_quiz,
        description="Generate quiz questions on a topic, grounded in indexed study material.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to quiz on"},
                "num_questions": {"type": "integer", "description": "Number of questions (default: 5)"}
            },
            "required": ["topic"]
        }
    ))

    registry.register(Action(
        name="summarize_topic",
        function=summarize_topic,
        description="Summarize indexed study material on a given topic.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to summarize"}
            },
            "required": ["topic"]
        }
    ))

    registry.register(Action(
        name="list_sources",
        function=list_sources,
        description="List all files currently indexed in the knowledge base.",
        parameters={"type": "object", "properties": {}}
    ))

    registry.register(Action(
        name="add_task",
        function=add_task,
        description="Add a study task to the task list.",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task description"}
            },
            "required": ["task"]
        }
    ))

    registry.register(Action(
        name="get_tasks",
        function=get_tasks,
        description="Get all study tasks.",
        parameters={"type": "object", "properties": {}}
    ))

    registry.register(Action(
        name="terminate",
        function=terminate,
        description="End the study session.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        },
        terminal=True
    ))

    return registry


# ─── Entry point ──────────────────────────────────────────────────────────

def main():
    goals = [
        Goal(1, "Answer from material", "Use RAG to answer questions from indexed study files"),
        Goal(2, "Generate content", "Create quizzes and summaries grounded in real material"),
        Goal(3, "Manage tasks", "Track what the student needs to study"),
    ]

    registry = build_registry()
    agent = Agent(goals, registry)

    print("=" * 50)
    print("StudyBuddy v2 — RAG-powered study assistant")
    print("=" * 50)
    print("Try:")
    print("  Ingest my notes: notes/")
    print("  What is the CIA triad?")
    print("  Generate a 5-question quiz on neural networks")
    print("  Summarize backpropagation")
    print("  Show my indexed sources")
    print("  Add task: Review attention mechanism")
    print("  exit")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("StudyBuddy: Good luck with your studies!")
            break

        agent.run(user_input)


if __name__ == "__main__":
    main()
