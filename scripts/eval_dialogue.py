"""Save reproducible multilingual conversation probes; these are not fluency scores."""
import argparse
import json
from pathlib import Path
import urllib.request

PROBES = [
    {"id": "pt", "messages": [{"role": "user", "content": "Responda em português, em duas frases: por que o céu parece azul?"}]},
    {"id": "es", "messages": [{"role": "user", "content": "Responde en español: sugiere dos actividades tranquilas para un domingo lluvioso."}]},
    {"id": "english", "messages": [{"role": "user", "content": "I am new to cooking. Suggest a simple breakfast with oats, in three short steps."}]},
    {"id": "context", "messages": [{"role": "user", "content": "Meu gato se chama Pipoca e tem três anos."},
       {"role": "assistant", "content": "Entendi: seu gato se chama Pipoca e tem três anos."},
       {"role": "user", "content": "Qual é o nome do meu gato? Responda apenas o nome."}]},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists")
    results = []
    for probe in PROBES:
        payload = {"model": "microgpt", "messages": probe["messages"], "temperature": 0, "max_tokens": 256}
        request = urllib.request.Request(args.url + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        results.append({**probe, "response": result["choices"][0], "usage": result["usage"]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"label": args.label, "probes": results}, ensure_ascii=False, indent=2) + "\n")
    for result in results:
        print(result["id"], result["response"]["message"]["content"])


if __name__ == "__main__":
    main()
