### This Script is Vibe Coded ###

import statistics
import threading
import time
from engine.engine import Engine

model_name = "Qwen3-0.6B"  # must be a key in MODEL_TO_TOKENIZER
batch_sizes = [1, 4, 16, 64]
base_prompts = [
    "Explain how a KV cache works.",
    "What is continuous batching?",
    "Describe paged attention in one paragraph.",
    "Why is decode memory-bandwidth bound?",
]

engine = Engine(model_name)
engine.start_background()


def drain(q, times):
    # one timestamp per yielded token; "done" is the end sentinel, not a token
    while q.get() != "done":
        times.append(time.perf_counter())


def run_batch(prompts):
    submits, all_times, threads = [], [], []
    for p in prompts:
        times = []
        submits.append(time.perf_counter())
        q = engine.add(p)
        t = threading.Thread(target=drain, args=(q, times))
        t.start()
        threads.append(t)
        all_times.append(times)
    for t in threads:
        t.join()
    return submits, all_times


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def report(n, submits, all_times):
    ttft = [ts[0] - s for s, ts in zip(submits, all_times)]
    e2e = [ts[-1] - s for s, ts in zip(submits, all_times)]
    tpot = [(ts[-1] - ts[0]) / (len(ts) - 1) for ts in all_times if len(ts) > 1]
    total_tokens = sum(len(ts) for ts in all_times)
    wall = max(ts[-1] for ts in all_times) - min(submits)

    print(f"\n=== batch size {n} ===")
    print(f"tokens:     {total_tokens}  ({total_tokens / n:.0f}/req)")
    for name, xs, scale, unit in [("TTFT", ttft, 1000, "ms"), ("TPOT", tpot, 1000, "ms"), ("E2E", e2e, 1, "s")]:
        print(f"{name}:  mean {statistics.mean(xs)*scale:8.1f}  p50 {pct(xs, 50)*scale:8.1f}  p99 {pct(xs, 99)*scale:8.1f} {unit}")
    print(f"decode/req: {1 / statistics.mean(tpot):.0f} tok/s")
    print(f"Throughput: {total_tokens / wall:.0f} tok/s (aggregate)")


# warmup so the first batch doesn't pay for the engine's idle sleep
run_batch(base_prompts[:1])

for n in batch_sizes:
    prompts = [base_prompts[i % len(base_prompts)] for i in range(n)]
    report(n, *run_batch(prompts))
