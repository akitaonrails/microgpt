"""Local Chat Completions server for OpenCode. Tools execute in the harness only."""
import argparse
import hmac
import os
from pathlib import Path
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from jsonschema.exceptions import SchemaError
import torch

from .chat import ChatRequest, normalize, parse_reply, sse_events
from .inference import NativeBackend


def error(message, status=400):
    return JSONResponse({"error": {"message": str(message), "type": "invalid_request_error" if status < 500 else "server_error"}}, status_code=status)


def create_app(backend, model_name="microgpt", api_key=None, max_output=2048):
    app = FastAPI(title="MicroGPT local server")
    lock = threading.Lock()

    @app.middleware("http")
    async def authentication(request: Request, call_next):
        if api_key and not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + api_key):
            return error("Invalid API key", 401)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exception):
        return error(str(exception))

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "context": backend.context, "source": backend.source}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "created": 0, "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def complete(request: ChatRequest):
        if request.model != model_name:
            return error("Unknown model", 404)
        count = request.max_completion_tokens or request.max_tokens or min(512, max_output)
        if count > max_output:
            return error(f"Requested output exceeds server maximum {max_output}")
        stop = [request.stop] if isinstance(request.stop, str) else request.stop or []
        if any(not isinstance(s, str) or not s for s in stop):
            return error("Stop sequences must be nonempty strings")
        try:
            messages, tools = normalize(request)
            ids = backend.encode(messages, tools)
            if len(ids) + count > backend.context:
                return error(f"Prompt ({len(ids)}) + output ({count}) exceeds context {backend.context}; compact history or reduce output")
        except (ValueError, TypeError, KeyError, SchemaError) as exception:
            return error(exception)
        # One cache, one trust domain. Fail fast rather than queue unbounded GPU work.
        if not lock.acquire(blocking=False):
            return error("Model busy; retry after the current request", 429)
        try:
            result = backend.complete(ids, count, request.temperature, stop, request.seed)
        except ValueError as exception:
            return error(exception)
        except torch.cuda.OutOfMemoryError:
            return error("GPU out of memory; reduce context/output or close other GPU workloads", 503)
        finally:
            lock.release()
        try:
            message, reason = parse_reply(result.text, tools, result.finish_reason)
        except ValueError as exception:
            return error(exception, 502)
        response = {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
                    "created": int(time.time()), "model": model_name,
                    "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                    "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                              "total_tokens": result.prompt_tokens + result.completion_tokens,
                              "prompt_tokens_details": {"cached_tokens": result.cached_tokens}}}
        if request.stream:
            return StreamingResponse(sse_events(response, bool((request.stream_options or {}).get("include_usage"))),
                                     media_type="text/event-stream")
        return response

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="microgpt")
    parser.add_argument("--max-output", type=int, default=4096)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--attention", choices=["sdpa", "flash"], default="sdpa")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    key = os.environ.get("MICROGPT_API_KEY")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not key:
        parser.error("Set MICROGPT_API_KEY before binding a network interface")
    if args.max_output < 1 or args.prefill_chunk < 1:
        parser.error("Token limits must be positive")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(4)
    backend = NativeBackend(args.checkpoint, device, args.attention,
                            not args.no_prefix_cache, args.prefill_chunk)
    print(backend.source, flush=True)
    import uvicorn
    uvicorn.run(create_app(backend, args.model_name, key, args.max_output), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
