import random
import queue

class BlockManager:
  def __init__(self, max_slots):
    self.available_slots = [i for i in range(max_slots)]

  def request_slots(self, n):
    if n > len(self.available_slots):
      return None
    available_slots = self.available_slots[:n]
    self.available_slots = self.available_slots[n:]
    return available_slots

  def give_back_slots(self, slots):
    self.available_slots.extend(slots)

# class Request:
#     def __init__(self, prompt, input_token_ids):
#         self.prompt = prompt
#         self.total_token_ids = input_token_ids
#         self.current_token_ids = input_token_ids
#         self.rid = random.randrange(100000)
#         self.queue = queue.Queue()
#         self.num_run_iterations = 0
#         self.read_slots = []
#         self.write_slots = []

class Batch:
    def __init__(self, batch, is_prefill):
      self.batch = batch
      self.is_prefill = is_prefill

class Scheduler:
    def __init__(self, batch_size, block_manager):
        self.batch_size = batch_size
        self.block_manager = block_manager
        self.waiting_queue = []
        self.running_queue = []

    def add_new_request(self, request):
       self.waiting_queue.append(request)

    def remove_request(self, request):
        if request in self.waiting_queue:
            self.waiting_queue.remove(request)
        if request in self.running_queue:
            self.running_queue.remove(request)
        slots = request.read_slots
        slots.extend(request.write_slots)
        self.block_manager.give_back_slots(slots)

    def construct_batch(self):
        batch = []
        for _, req in enumerate(self.waiting_queue):
           if len(req.read_slots) == 0:
              available_slots = self.block_manager.request_slots(len(req.current_token_ids))
              if available_slots is not None:
                 req.write_slots = available_slots
                 batch.append(req)
        if len(batch) > 0:
           for _, req in enumerate(batch):
              self.waiting_queue.remove(req)
              # req.read_slots = req.write_slots
              # req.write_slots = []
              self.running_queue.append(req)
           return Batch(batch, True)

        for _, req in enumerate(self.running_queue):
           available_slots = self.block_manager.request_slots(1)
           if available_slots is not None:
              req.read_slots.extend(req.write_slots)
              req.write_slots = available_slots
              batch.append(req)

        return Batch(batch, False)


