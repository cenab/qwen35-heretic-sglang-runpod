import os
import subprocess
import time

import requests
import runpod


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "30000"))
BASE_URL = f"http://{HOST}:{PORT}"


def env_value(*names):
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return None


def start_sglang():
    model = env_value("MODEL_NAME", "MODEL_PATH")
    if not model:
        raise RuntimeError("Set MODEL_NAME or MODEL_PATH to the Hugging Face model id.")

    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--model-path",
        model,
    ]

    options = {
        "TOKENIZER_PATH": "--tokenizer-path",
        "TOKENIZER_MODE": "--tokenizer-mode",
        "LOAD_FORMAT": "--load-format",
        "DTYPE": "--dtype",
        "CONTEXT_LENGTH": "--context-length",
        "QUANTIZATION": "--quantization",
        "SERVED_MODEL_NAME": "--served-model-name",
        "CHAT_TEMPLATE": "--chat-template",
        "MEM_FRACTION_STATIC": "--mem-fraction-static",
        "MAX_RUNNING_REQUESTS": "--max-running-requests",
        "MAX_TOTAL_TOKENS": "--max-total-tokens",
        "CHUNKED_PREFILL_SIZE": "--chunked-prefill-size",
        "MAX_PREFILL_TOKENS": "--max-prefill-tokens",
        "SCHEDULE_POLICY": "--schedule-policy",
        "SCHEDULE_CONSERVATIVENESS": "--schedule-conservativeness",
        "TENSOR_PARALLEL_SIZE": "--tp-size",
        "STREAM_INTERVAL": "--stream-interval",
        "RANDOM_SEED": "--random-seed",
        "LOG_LEVEL": "--log-level",
        "LOG_LEVEL_HTTP": "--log-level-http",
        "API_KEY": "--api-key",
        "FILE_STORAGE_PATH": "--file-storage-path",
        "DATA_PARALLEL_SIZE": "--data-parallel-size",
        "LOAD_BALANCE_METHOD": "--load-balance-method",
        "ATTENTION_BACKEND": "--attention-backend",
        "SAMPLING_BACKEND": "--sampling-backend",
        "TOOL_CALL_PARSER": "--tool-call-parser",
        "REASONING_PARSER": "--reasoning-parser",
        "SPECULATIVE_ALGO": "--speculative-algo",
        "SPECULATIVE_NUM_STEPS": "--speculative-num-steps",
        "SPECULATIVE_EAGLE_TOPK": "--speculative-eagle-topk",
        "SPECULATIVE_NUM_DRAFT_TOKENS": "--speculative-num-draft-tokens",
    }

    flags = [
        "SKIP_TOKENIZER_INIT",
        "TRUST_REMOTE_CODE",
        "LOG_REQUESTS",
        "SHOW_TIME_COST",
        "DISABLE_RADIX_CACHE",
        "DISABLE_CUDA_GRAPH",
        "DISABLE_OUTLINES_DISK_CACHE",
        "ENABLE_TORCH_COMPILE",
        "ENABLE_P2P_CHECK",
        "ENABLE_FLASHINFER_MLA",
        "TRITON_ATTENTION_REDUCE_IN_FP32",
    ]

    for env_name, cli_name in options.items():
        value = os.getenv(env_name)
        if value is not None and value != "":
            command.extend([cli_name, value])

    for env_name in flags:
        if os.getenv(env_name, "").lower() in ("1", "true", "yes"):
            command.append(f"--{env_name.lower().replace('_', '-')}")

    printable = " ".join(command)
    print(f"Starting SGLang: {printable}", flush=True)
    return subprocess.Popen(command)


def wait_for_sglang():
    timeout = int(os.getenv("SERVER_READY_TIMEOUT", "3600"))
    interval = int(os.getenv("SERVER_READY_INTERVAL", "5"))
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        try:
            response = requests.get(f"{BASE_URL}/v1/models", timeout=5)
            if response.status_code == 200:
                print("SGLang is ready.", flush=True)
                return
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(interval)

    raise TimeoutError(f"SGLang did not become ready in {timeout}s. Last error: {last_error}")


process = start_sglang()
wait_for_sglang()


def model_name():
    return env_value("SERVED_MODEL_NAME", "MODEL_NAME", "MODEL_PATH") or "default"


def stream_response(response):
    for line in response.iter_lines():
        if line:
            yield line.decode("utf-8")


async def handler(job):
    job_input = job.get("input") or {}

    if job_input.get("openai_route"):
        route = job_input["openai_route"]
        payload = job_input.get("openai_input") or {}
        response = requests.post(
            f"{BASE_URL}{route}",
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=bool(payload.get("stream")),
            timeout=None,
        )
        if response.status_code >= 400:
            yield {"error": response.text, "status_code": response.status_code}
            return
        for chunk in stream_response(response):
            yield chunk
        return

    if "messages" in job_input:
        payload = dict(job_input)
        payload.setdefault("model", model_name())
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=bool(payload.get("stream")),
            timeout=None,
        )
        if response.status_code >= 400:
            yield {"error": response.text, "status_code": response.status_code}
            return
        for chunk in stream_response(response):
            yield chunk
        return

    response = requests.post(
        f"{BASE_URL}/generate",
        headers={"Content-Type": "application/json"},
        json=job_input,
        timeout=None,
    )
    if response.status_code >= 400:
        yield {"error": response.text, "status_code": response.status_code}
        return
    yield response.json()


def max_concurrency(default=4):
    return int(os.getenv("MAX_CONCURRENCY", str(default)))


runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": max_concurrency,
        "return_aggregate_stream": True,
    }
)
