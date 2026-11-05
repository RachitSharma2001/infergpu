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

'''



Changes:
- model runner should pre play the decode graph at various batch sizes 
- model runner needs to get results and store it back in request 

Scheduler:
- add_new_request
- construct_batch
- remove_request

waiting_queue
running_queue


construct_batch:
- First: try constructing for prefill
  check waiting queue for any request in prefill stage 
  for any of those requests
    get block manager to allocate the necessary write slots, if can't, skip

    store those in write_slots for the request 


if there are at least one
  return that as the batch 

- Next: construct for decode
  go through each request in running queue 
    check block maanger if a slot is open 


BlockManager:

'''

class Request:
    def __init__(self, prompt, input_token_ids):
        self.prompt = prompt
        self.token_ids = input_token_ids
        self.rid = random.randrange(100000)
        self.queue = queue.Queue()
        self.num_run_iterations = 0
        self.read_slots = []
        self.write_slots = []

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
        print("slots", slots)
        self.block_manager.give_back_slots(slots)

    def construct_batch(self):
        batch = []
        for _, req in enumerate(self.waiting_queue):
           if len(req.read_slots) == 0:
              available_slots = self.block_manager.request_slots(len(req.token_ids))
              if available_slots is not None:
                 req.write_slots = available_slots
                 batch.append(req)
        if len(batch) > 0:
           for _, req in enumerate(batch):
              self.waiting_queue.remove(req)
              req.read_slots = req.write_slots
              req.write_slots = []
              self.running_queue.append(req)
           return batch 

        for _, req in enumerate(self.running_queue):
           available_slots = self.block_manager.request_slots(1)
           if available_slots is not None:
              req.read_slots.extend(req.write_slots)
              req.write_slots = available_slots
              batch.append(req)

        return batch 
       

scheduler = Scheduler(2, BlockManager(100))

req = Request("a b c", [12, 13, 15])
req2 = Request("d e f", [12, 13, 14])
req3 = Request("w w w", [1, 3, 4])

scheduler.add_new_request(req)
# scheduler.add_new_request(req2)
# scheduler.add_new_request(req3)

print(scheduler.construct_batch())
print(scheduler.construct_batch())
print(scheduler.construct_batch())


scheduler.remove_request(req)




# bm = BlockManager(10)
# print(bm.request_slots(8))
# print(bm.request_slots(1))
# print(bm.request_slots(3))
# bm.give_back_slots([3, 7, 2])

# print(bm.request_slots(3))