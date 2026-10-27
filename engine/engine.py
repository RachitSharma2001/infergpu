import random
import time
import queue 
from transformers import AutoTokenizer
import torch
import threading 

MODEL_TO_TOKENIZER = {
    "Qwen3-0.6B": AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
}

MODEL_TO_EOS = {
    "Qwen3-0.6B": 151645
}

class Engine:
    def __init__(self, model_name, max_response_size=100):
        self.model_name = model_name
        self.scheduler = Scheduler()
        self.model_runner = ModelRunner()
        self.tokenizer = MODEL_TO_TOKENIZER[model_name]
        self.eos = MODEL_TO_EOS[model_name]
        self.max_response_size = max_response_size

    def start_background(self):
        thread = threading.Thread(
            target = self.manage_steps,
            daemon=True,
        )
        thread.start()

    def add(self, prompt):
        input_token_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids'][0]
        req = Request(prompt, input_token_ids)
        self.scheduler.add_new_request(req)
        return req.queue

    def step(self):
        batch = self.scheduler.construct_batch()
        if len(batch) == 0:
            return 0
        results = self.model_runner.forward_pass(batch)
        for req, res in zip(batch, results):
            req.queue.put(self.tokenizer.batch_decode(res.output_tokens))
            if req.num_run_iterations == self.max_response_size or res.output_tokens[-1] == self.eos:
                req.queue.put("done")
                self.scheduler.remove_request(req)
            req.num_run_iterations += 1
        return len(batch)

    def manage_steps(self):
        while True:
            num_processed = self.step()
            if num_processed == 0:
                time.sleep(0.5)


class Result:
    def __init__(self, output_tokens):
        self.output_tokens = output_tokens

class Request:
    def __init__(self, prompt, input_token_ids):
        self.prompt = prompt 
        self.input_token_ids = input_token_ids
        self.rid = f"{random.randrange(100000)}"
        self.queue = queue.Queue()
        self.num_run_iterations = 0

class Scheduler:
    def __init__(self):
        self.waiting_queue = []

    def add_new_request(self, req):
        self.waiting_queue.append(req)

    def remove_request(self, req):
        self.waiting_queue.remove(req)

    def construct_batch(self):
        return self.waiting_queue

class ModelRunner:
    def forward_pass(self, requests):
        results = []
        for _ in requests:
            results.append(Result([840]))
        return results 