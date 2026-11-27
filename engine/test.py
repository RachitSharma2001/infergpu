import time
from engine import Engine
import threading

engine = Engine("gpt2", batch_size=32, max_response_size=500)

engine.start_background()


def collect(q, lis):
  while True:
      r = q.get()
      if r == "done":
        print(lis)
        break
      lis.append(r)

for _ in range(100):
  prompt = "What is pizza?"
  q = engine.add(prompt)


  l1 = []
  t1 = threading.Thread(target=collect, args=(q, l1))
  t1.start()
  t1.join()