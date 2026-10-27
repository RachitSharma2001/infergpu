import time
from engine.engine import Engine

engine = Engine("Qwen3-0.6B", 10)
engine.start_background()
prompt = "Explain how a KV cache works."

q = engine.add(prompt)

while True:
    r = q.get()
    if r == "done":
        break
    print(r)

print("done")

time.sleep(10)