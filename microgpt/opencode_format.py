"""OpenCode 1.18.34 fixture formatting shared by training and evaluation.

The real harness still executes tools. These helpers only format original
simulated demonstrations; no model answers or grader results are injected.
"""
from pathlib import PurePosixPath

AGENT_PROMPT = 'You fix Python code. Read solution.py, edit it to satisfy the request, then call check to test. Preserve earlier requirements in TASK_STATE.md when present. Line numbers in read output are annotations, not file content. Correct failures. Give a brief final answer.'


def cli_user_content(text):
    """Observed OpenCode 1.18.34 CLI text: quote wrapping, not JSON encoding.

    Embedded quotes are escaped; newlines and existing backslashes are preserved.
    This helper is for generated CLI lessons, not normalization of user requests.
    """
    return '"' + text.replace('"', '\\"') + '"'


def state_text(items):
    notes = '\n'.join(f"- {t['name']}(value) must return value * {t['factor']} + {t['offset']}." for t in items)
    return '# Required behavior\n' + notes + '\nPreserve and test every requirement above.\n'


def task_prompt(task, path='solution.py', state_path='TASK_STATE.md'):
    return f"In {path} add {task['name']}(value), returning value * {task['factor']} + {task['offset']}. Keep all earlier functions correct. Read {state_path} for the requirements, read the code, edit it, and call check. Retry transient tool failures."


def system_prompt(root, agent_prompt=AGENT_PROMPT):
    return (agent_prompt + '\nYou are powered by the model named microgpt. The exact model ID is microgpt/microgpt\n'
            'Here is some useful information about the environment you are running in:\n<env>\n'
            f'  Working directory: {root}\n  Workspace root folder: {root}\n'
            "  Is directory a git repo: yes\n  Platform: linux\n  Today's date: Sun Oct 04 2026\n</env>")


def read_output(path, content, root):
    display = str(PurePosixPath(root) / path)
    lines = content.splitlines()
    numbered = '\n'.join(f'{i}: {line}' for i, line in enumerate(lines, 1))
    return f'<path>{display}</path>\n<type>file</type>\n<content>\n{numbered}\n\n(End of file - total {len(lines)} lines)\n</content>'
