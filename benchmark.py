import time
from engine.engine import Engine

model_name = "~/huggingface/Qwen3-0.6B/"

engine = Engine(model_name)
prompt = "Explain how a KV cache works."

start = time.perf_counter()
times = [time.perf_counter() for _ in engine.stream(prompt)]
# the list comprehension records one timestamp per yielded token

ttft = times[0] - start
e2e  = times[-1] - start
tpot = (times[-1] - times[0]) / (len(times) - 1)

print(f"tokens: {len(times)}")
print(f"TTFT:   {ttft*1000:.1f} ms")
print(f"TPOT:   {tpot*1000:.1f} ms  ({1/tpot:.0f} tok/s decode)")
print(f"E2E:    {e2e:.2f} s")
print(f"Throughput: {len(times)/e2e:.0f} tok/s")