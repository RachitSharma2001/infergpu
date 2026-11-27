import random
import time
import queue
from transformers import AutoTokenizer
import torch
import threading

MODEL_TO_TOKENIZER = {
    "gpt2": AutoTokenizer.from_pretrained("gpt2")
}

MODEL_TO_EOS = {
    "gpt2": 151645
}

import random
import queue

class Result:
    def __init__(self, output_tokens):
        self.output_tokens = output_tokens

class Request:
    def __init__(self, prompt, input_token_ids):
        self.prompt = prompt
        self.total_token_ids = input_token_ids
        self.current_token_ids = input_token_ids
        self.rid = random.randrange(100000)
        self.queue = queue.Queue()
        self.num_run_iterations = 0
        self.read_slots = []
        self.write_slots = []

class Engine:
    def __init__(self, model_name, batch_size = 1, max_response_size=10):
        self.model_name = model_name
        block_manager = BlockManager(1000)
        self.scheduler = Scheduler(batch_size, block_manager)
        self.model_runner = ModelRunner(model_name)
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
        if len(batch.batch) == 0:
            return 0
        results = self.model_runner.forward_pass(batch.batch, batch.is_prefill)
        for req, res in zip(batch.batch, results):
            req.queue.put(self.tokenizer.batch_decode(res.output_tokens))
            if req.num_run_iterations == self.max_response_size or res.output_tokens[-1] == self.eos:
                req.queue.put("done")
                self.scheduler.remove_request(req)
            req.num_run_iterations += 1
        return len(batch.batch)

    def manage_steps(self):
        while True:
            num_processed = self.step()
            if num_processed == 0:
                time.sleep(0.5)


