"""One explicit text protocol shared by scratch SFT and serving.

Encode each segment separately so BPE merges cannot move a loss-mask boundary.
EOS terminates messages; role headers and tool markup are ordinary text.
"""
import hashlib
import json

VERSION = 'microgpt-chat-v1'


def reasoning_instruction(effort):
    if effort == 'none':
        return 'Answer directly without a thinking section. Use tools when needed.'
    detail = {'low': 'Keep the worked reasoning brief.', 'medium': 'Work through the necessary steps and check the result.',
              'high': 'Break the problem into steps, consider edge cases, and verify the result before answering.'}[effort]
    return ('Use a <think>...</think> section for your worked reasoning, followed by the final answer or tool call. '
            + detail + ' Only tool calls outside the thinking section are actions.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tokenizer_identity(tokenizer):
    if tokenizer.encoding:
        return digest({'encoding': tokenizer.encoding.name, 'vocab': tokenizer.vocab_size})
    if tokenizer.backend:
        return digest(tokenizer.backend.to_str())
    return digest({'encoding': 'utf8-bytes-v1'})


def body(message):
    if message['role'] != 'assistant':
        return json.dumps(message, ensure_ascii=False, sort_keys=True)
    text = message.get('content') or ''
    if message.get('reasoning_content'):
        text = '<think>\n' + message['reasoning_content'].strip() + '\n</think>\n\n' + text
    for call in message.get('tool_calls', []):
        function = call['function']
        arguments = function['arguments']
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        text += '\n<tool_call>' + json.dumps({'name': function['name'], 'arguments': arguments}, ensure_ascii=False) + '</tool_call>'
    return text.lstrip('\n')


def prompt_ids(tokenizer, messages, tools):
    ids = tokenizer.encode('MicroGPT conversation.\n')
    if tools:
        ids += tokenizer.encode('Tools: ' + json.dumps(tools, ensure_ascii=False, sort_keys=True) + '\n')
    for message in messages:
        ids += tokenizer.encode(f"{message['role']}:\n")
        ids += tokenizer.encode(body(message)) + [tokenizer.eos]
    return ids + tokenizer.encode('assistant:\n')
