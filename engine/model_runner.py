from itertools import chain
import torch
import random
import os
from models.gpt2 import GPT2Model
from kvcache.cache import Pool 

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class ModelRunner:
    def __init__(self, model_name):
      # todo use the actual parameters of gpt2
      self.max_size = 1000
      self.kvcache = Pool(12, self.max_size, 12, 64)
      if model_name != "gpt2":
        raise Exception("model not supported")

      with torch.device("cuda"):
        self.gpt2_model = GPT2Model(self.kvcache)



      self.token_ids_gpu = torch.zeros(self.max_size, device="cuda", dtype=int)
      self.request_ids_gpu = torch.zeros(self.max_size, device="cuda", dtype=int)
      self.position_ids_gpu = torch.zeros(self.max_size, device="cuda", dtype=int)
      self.write_slots_gpu = torch.zeros(self.max_size, device="cuda", dtype=int)
      self.p_gpu = torch.zeros(self.max_size, device="cuda")
      self.new_req_ids_gpu = torch.zeros(self.max_size, device="cuda", dtype=int)
      self.tot_slots_gpu = torch.zeros(self.max_size, device="cuda", dtype=int)
      self.logits_gpu = torch.zeros((self.max_size, 50257), device="cuda")

      # self.cuda_graph = torch.cuda.CUDAGraph()
      self.max_response_tokens = 15
      self.cuda_graph = None
      self.num_replays = 0


      # warmup
      _a = torch.randn(8, 8, device="cuda")
      _ = _a @ _a
      torch.cuda.synchronize()


    def forward_pass(self, requests, is_prefill):
      token_ids = torch.cat(
          [torch.tensor(req.current_token_ids) for req in requests],
          dim=0,
      )

      read_slots = torch.tensor(list(chain.from_iterable([req.read_slots for req in requests])))
      read_offsets = torch.tensor([len(req.read_slots) for req in requests])
      write_slots = torch.tensor(list(chain.from_iterable([req.write_slots for req in requests])))
      request_ids = torch.cat(
          [
              torch.full(
                  (len(req.current_token_ids),),
                  req.rid,
                  device="cpu",
                  dtype=torch.long,
              )
              for i, req in enumerate(requests)
          ],
          dim=0,
      )
      if is_prefill:
        position_ids = torch.cat(
            [
                torch.arange(len(req.total_token_ids), device="cpu")
                for req in requests
            ],
            dim=0,
        )
      else:
        position_ids = torch.cat(
            [
                torch.tensor([len(req.total_token_ids)-1])
                for req in requests
            ],
            dim=0,
        )

      p, new_req_ids, tot_slots = self.kvcache.compute_metadata(request_ids, read_slots, read_offsets, write_slots)
      # print("new request ids, tot slots: ", new_req_ids, tot_slots)
      self.token_ids_gpu[:len(token_ids)].copy_(token_ids, non_blocking=True)
      self.request_ids_gpu[:len(request_ids)].copy_(request_ids, non_blocking=True)
      self.position_ids_gpu[:len(position_ids)].copy_(position_ids, non_blocking=True)
      self.write_slots_gpu[:len(write_slots)].copy_(write_slots, non_blocking=True)
      self.p_gpu[:len(p)].copy_(p, non_blocking=True)
      self.new_req_ids_gpu[:len(new_req_ids)].copy_(new_req_ids, non_blocking=True)
      self.tot_slots_gpu[:len(tot_slots)].copy_(tot_slots, non_blocking=True)

      torch.cuda.synchronize()

      with torch.device("cuda"):
        if is_prefill is False:
          if self.cuda_graph is None:
              self.cuda_graph = torch.cuda.CUDAGraph()
              print("capturing")
              with torch.cuda.graph(self.cuda_graph):
                self.logits_gpu = self.gpt2_model.forward(
                    self.token_ids_gpu[:len(token_ids)],
                    self.request_ids_gpu[:len(request_ids)],
                    self.position_ids_gpu[:len(position_ids)],
                    self.write_slots_gpu[:len(write_slots)],
                    self.p_gpu[:len(p)],
                    self.new_req_ids_gpu,
                    self.tot_slots_gpu,
                )
          # print(self.token_ids_gpu[:len(token_ids)])
          # print(self.request_ids_gpu[:len(request_ids)])
          # print(self.position_ids_gpu[:len(position_ids)])
          # print(self.write_slots_gpu[:len(write_slots)])
          # print(self.p_gpu[:len(p)])
          # print(self.new_req_ids_gpu[:len(new_req_ids)])
          # print(self.tot_slots_gpu[:len(tot_slots)])
          self.cuda_graph.replay()
          torch.cuda.synchronize()
          # self.cuda_graph  = None
        else:
          self.cuda_graph = None
          self.logits_gpu = self.gpt2_model.forward(
              self.token_ids_gpu[:len(token_ids)],
              self.request_ids_gpu[:len(request_ids)],
              self.position_ids_gpu[:len(position_ids)],
              self.write_slots_gpu[:len(write_slots)],
              self.p_gpu[:len(p)],
              self.new_req_ids_gpu[:len(new_req_ids)],
              self.tot_slots_gpu[:len(tot_slots)],
          )


      logits_cpu = self.logits_gpu.to("cpu")
      probs = torch.softmax(logits_cpu, dim=-1)
      next_tokens = torch.argmax(probs, dim=-1)

      offset = 0
      results = []
      # print("----------------------")
      for req in requests:
        request_len = len(req.current_token_ids)
        next_token = next_tokens[offset + request_len - 1]
        # print(next_token)
        req.total_token_ids = torch.cat((req.total_token_ids, next_token.view(1)))
        req.current_token_ids = next_token.view(1)
        results.append(Result(next_token.view(1)))
        offset += request_len
      # print("----------------------")
      return results

