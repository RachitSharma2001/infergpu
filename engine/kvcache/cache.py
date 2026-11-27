import torch

class Pool:
    def __init__(self, max_layers, max_slots, num_heads, head_dim):
        kpool = {}
        vpool = {}
        for layer in range(max_layers):
            kpool[layer] = torch.empty((max_slots, num_heads, head_dim), device="cuda")
            vpool[layer] = torch.empty((max_slots, num_heads, head_dim), device="cuda")
        self.kpool_from_layer = kpool
        self.vpool_from_layer = vpool
        self.free_slots = [i for i in range(max_slots)]
        self.num_heads = num_heads
        self.head_dim = head_dim

    def set_vector(self, layer_num, slot_num, k, v):
        self.kpool_from_layer[layer_num][slot_num] = k
        self.vpool_from_layer[layer_num][slot_num] = v


    def free_these_slots(self, slot_nums):
        for _, n in enumerate(slot_nums):
            self.free_slots.append(n)

    def request_slots(self, num_slots):
        slots_to_return = self.free_slots[:num_slots]
        self.free_slots = self.free_slots[num_slots:]
        return slots_to_return

    def compute_metadata(self, request_ids, read_slots, read_offsets, write_slots):
      unique_req_ids, counts = torch.unique_consecutive(request_ids, return_counts=True)
      read_slots_split = torch.split(read_slots, read_offsets.tolist())
      write_slots_split = torch.split(write_slots, counts.tolist())
      tot_slots = []
      req_len = counts + read_offsets
      block_nums = torch.ones((torch.sum(counts), )) + torch.repeat_interleave(read_offsets, repeats=counts)
      for read_slots_piece, write_slots in zip(read_slots_split, write_slots_split):
        tot_slots.extend(read_slots_piece)
        tot_slots.extend(write_slots)

      p = torch.cumsum(block_nums, dim=0) - 1
      new_req_ids = torch.repeat_interleave(unique_req_ids, repeats=req_len)

      return p, new_req_ids, torch.tensor(tot_slots)

    def prepend(self, layer_num, write_slots, tot_slots, k, v):
      self.kpool_from_layer[layer_num][write_slots] = k.reshape(len(write_slots), self.num_heads, self.head_dim)
      self.vpool_from_layer[layer_num][write_slots] = v.reshape(len(write_slots), self.num_heads, self.head_dim)
      return self.kpool_from_layer[layer_num][tot_slots], self.vpool_from_layer[layer_num][tot_slots]
