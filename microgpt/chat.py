"""The narrow Chat Completions contract used by our OpenCode integration."""
import json
import re
import uuid
from typing import Any, Literal

from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel, Field
from .protocol import reasoning_instruction


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Literal["auto", "none"] = "auto"
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.6, ge=0, le=2)
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=0)
    n: Literal[1] = 1
    reasoning_effort: Literal['none', 'low', 'medium', 'high'] | None = None


def normalize(request):
    """Keep roles, assistant calls, and tool-result IDs across every harness turn."""
    messages = []
    pending = set()
    used = set()
    for original in request.messages:
        message = dict(original)
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError("Unsupported message role")
        if role == "developer":
            message["role"] = "system"
        if message.get('reasoning_content') is not None:
            if role != 'assistant' or not isinstance(message['reasoning_content'], str):
                raise ValueError('reasoning_content must be assistant text')
        content = message.get("content")
        if isinstance(content, list):
            if any(not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str) for part in content):
                raise ValueError("This server accepts text only")
            message["content"] = "\n".join(part["text"] for part in content)
        elif content is not None and not isinstance(content, str):
            raise ValueError("Message content must be text")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("Tool result has no matching pending assistant call")
            pending.remove(call_id)
        elif pending:
            raise ValueError("All tool results must arrive before the next message")
        if message.get("tool_calls"):
            if role != "assistant":
                raise ValueError("Only assistant messages may contain tool_calls")
            normalized_calls = []
            for original_call in message["tool_calls"]:
                call = dict(original_call)
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in used or call.get("type") != "function":
                    raise ValueError("Tool calls need unique IDs and type=function")
                function = dict(call["function"])
                args = function.get("arguments", {})
                function["arguments"] = json.loads(args) if isinstance(args, str) else args
                if not isinstance(function["arguments"], dict):
                    raise ValueError("Tool arguments must be a JSON object")
                call["function"] = function
                normalized_calls.append(call)
                pending.add(call_id)
                used.add(call_id)
            message["tool_calls"] = normalized_calls
        messages.append(message)
    if pending:
        raise ValueError("Missing tool results")
    if request.reasoning_effort is not None:
        messages.insert(0, {'role': 'system', 'content': reasoning_instruction(request.reasoning_effort)})
    tools = request.tools if request.tool_choice == "auto" else []
    names = set()
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("Only function tools are supported")
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Tool names must be unique nonempty strings")
        names.add(name)
        Draft202012Validator.check_schema(function.get("parameters", {"type": "object"}))
    return messages, tools


def parse_reply(text, tools, finish_reason):
    """Separate model-generated thinking before considering executable tool calls."""
    thought = None
    if text.lstrip().startswith('<think>'):
        thought, closed, text = text.lstrip()[len('<think>'):].partition('</think>')
        if not closed:
            if finish_reason != 'length':
                raise ValueError('Model ended inside an unclosed thinking section; no tool calls dispatched')
            return {'role': 'assistant', 'content': None, 'reasoning_content': thought.strip()}, finish_reason
        text = text.lstrip()
    if '<think' in text or '</think>' in text:
        raise ValueError('Malformed thinking section; no tool calls dispatched')
    message, reason = parse_actions(text, tools, finish_reason)
    if thought is not None:
        message['reasoning_content'] = thought.strip()
    return message, reason


def repeated_tool_cycle(messages, limit=3, max_period=3):
    """Detect identical consecutive actions AND observations without a new user turn.

    Also catches read/edit/read/edit cycles with unchanged observations. A changed
    result or a new user turn breaks the repeated cycle.
    """
    signatures = []
    end = len(messages)
    while end >= 2 and len(signatures) < limit * max_period:
        assistant, result = messages[end - 2:end]
        calls = assistant.get('tool_calls', [])
        if assistant.get('role') != 'assistant' or result.get('role') != 'tool' or len(calls) != 1:
            break
        signatures.append(json.dumps([calls[0]['function'], result.get('content')], sort_keys=True))
        end -= 2
    return any(len(signatures) >= limit * period and signatures[:limit * period] == signatures[:period] * limit
               for period in range(1, max_period + 1))


def parse_actions(text, tools, finish_reason):
    """Never execute output. Only return complete, offered, schema-valid calls."""
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL))
    if not matches:
        return {"role": "assistant", "content": text}, finish_reason
    functions = {tool["function"]["name"]: tool["function"] for tool in tools}
    calls = []
    for match in matches:
        try:
            payload = json.loads(match.group(1))
            name, arguments = payload["name"], payload["arguments"]
            if name not in functions or not isinstance(arguments, dict):
                raise ValueError("Unknown tool or invalid argument object")
            Draft202012Validator(functions[name].get("parameters", {})).validate(arguments)
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise ValueError(f"Model emitted an invalid tool call: {error}") from error
        calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function",
                      "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    # A truncated call must never be dispatched, including after an earlier complete call.
    if "<tool_call>" in content or "</tool_call>" in content or finish_reason == "length":
        return {"role": "assistant", "content": text}, finish_reason
    return {"role": "assistant", "content": content or None, "tool_calls": calls}, "tool_calls"


def sse_events(response, include_usage=False):
    """Buffered SSE: calls are validated before any action is exposed to the client."""
    base = {key: response[key] for key in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"
    message = response["choices"][0]["message"]
    def event(delta, reason=None):
        return "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}) + "\n\n"
    yield event({"role": "assistant"})
    if message.get('reasoning_content'):
        yield event({'reasoning_content': message['reasoning_content']})
    if message.get("content"):
        yield event({"content": message["content"]})
    for index, call in enumerate(message.get("tool_calls", [])):
        yield event({"tool_calls": [{"index": index, **call}]})
    yield event({}, response["choices"][0]["finish_reason"])
    if include_usage:
        yield "data: " + json.dumps({**base, "choices": [], "usage": response["usage"]}) + "\n\n"
    yield "data: [DONE]\n\n"
