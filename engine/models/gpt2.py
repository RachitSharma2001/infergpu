import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import math

class GPT2Model:

    def __init__(self, kpool, model_name="gpt2"):
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        # initialize the token embedding layer, which is [vocab_size] x [hidden dimension size]
        self.token_embedding_layer = nn.Embedding(
            hf_model.config.vocab_size,
            hf_model.config.n_embd
        )
        # initialize the position embedding layer, which is [max # of tokens possible to generate] x [hidden dimension size]
        self.position_embedding_layer = nn.Embedding(
            hf_model.config.n_positions,
            hf_model.config.n_embd,
        )
        # prevent py torch from computing gradients for this operation; this is inference, not training
        with torch.no_grad():
            self.token_embedding_layer.weight.copy_(hf_model.transformer.wte.weight)
            self.position_embedding_layer.weight.copy_(hf_model.transformer.wpe.weight)

        self.transformer_blocks = [TransformerBlock(hf_model, i, kpool) for i in range(hf_model.config.n_layer)]

        self.lm_head = nn.Linear(hf_model.config.n_embd, hf_model.config.vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(hf_model.lm_head.weight)

        self.final_layer_norm = nn.LayerNorm(hf_model.config.n_embd)
        with torch.no_grad():
            self.final_layer_norm.weight.copy_(hf_model.transformer.ln_f.weight)
            self.final_layer_norm.bias.copy_(hf_model.transformer.ln_f.bias)


    # performs forward pass for continuous batching. token_ids, request_ids,
    # and position_ids are all flat 1D tensors of length total_tokens -- the
    # tokens of every in-flight request packed back-to-back, with no batch
    # dimension. request_ids says which request each packed token belongs
    # to; position_ids is that token's position within its own request.
    def forward(self, token_ids, request_ids, position_ids, write_slots, p, new_req_ids, tot_slots, return_intermediates=False):
        # e.g. token_ids = [3, 121, 9]. This produces one row per token id:
        # [vector 3 in token embedding layer], [vector 121 ...], [vector 9 ...]
        token_embedding = self.token_embedding_layer(token_ids)
        # grabs the appropriate position vector for each packed token
        positional_embedding = self.position_embedding_layer(position_ids)
        # adds token and positional embedding results
        x = token_embedding + positional_embedding
        embeddings = x

        hidden_states = []
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, request_ids, write_slots, p, new_req_ids, tot_slots)
            hidden_states.append(x)

        final_hidden_states = self.final_layer_norm(x)
        logits = self.lm_head(final_hidden_states)

        if return_intermediates:
            return {
                "embeddings": embeddings,
                "hidden_states": hidden_states,
                "final_hidden_states": final_hidden_states,
                "logits": logits,
            }
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, hf_model, block_index_number, kpool):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hf_model.config.n_embd)
        self.layer_norm_2 = nn.LayerNorm(hf_model.config.n_embd)
        with torch.no_grad():
            self.layer_norm_1.weight.copy_(hf_model.transformer.h[block_index_number].ln_1.weight)
            self.layer_norm_1.bias.copy_(hf_model.transformer.h[block_index_number].ln_1.bias)
        with torch.no_grad():
            self.layer_norm_2.weight.copy_(hf_model.transformer.h[block_index_number].ln_2.weight)
            self.layer_norm_2.bias.copy_(hf_model.transformer.h[block_index_number].ln_2.bias)
        self.attention = Attention(hf_model.config, hf_model.transformer.h[block_index_number].attn, block_index_number, kpool)
        self.mlp = MLP(hf_model.config, hf_model.transformer.h[block_index_number].mlp)

    def forward(self, x, request_ids, write_slots, p, new_req_ids, tot_slots):
        x = x + self.attention(self.layer_norm_1(x), request_ids, write_slots, p, new_req_ids, tot_slots)
        x = x + self.mlp(self.layer_norm_2(x))
        return x

#### This Attention Class is vibe coded ####
class Attention(nn.Module):
    def __init__(self, config, hf_attention, block_index_number, kpool):
        super().__init__()

        self.kpool = kpool

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # GPT-2 uses one linear layer to produce Q, K, and V
        self.c_attn = nn.Linear(
            self.n_embd,
            3 * self.n_embd
        )

        # Projects concatenated attention heads back to hidden dimension
        self.c_proj = nn.Linear(
            self.n_embd,
            self.n_embd
        )

        # Load pretrained Hugging Face weights
        # HF's GPT-2 uses Conv1D internally, which stores its weight as
        # [in_features, out_features] -- the transpose of nn.Linear's
        # [out_features, in_features] -- so it must be transposed on copy.
        with torch.no_grad():
            self.c_attn.weight.copy_(hf_attention.c_attn.weight.t())
            self.c_attn.bias.copy_(hf_attention.c_attn.bias)

            self.c_proj.weight.copy_(hf_attention.c_proj.weight.t())
            self.c_proj.bias.copy_(hf_attention.c_proj.bias)

        self.block_index_number = block_index_number

    def forward(self, x, request_ids, write_slots, p, new_req_ids, tot_slots):
        # x is packed: [total_tokens, C], no batch dimension. total_tokens
        # is every in-flight request's tokens concatenated back-to-back.
        total_tokens, _ = x.shape

        # [T, C] -> [T, 3C]
        qkv = self.c_attn(x)

        # [T, 3C] -> three [T, C] tensors
        q, k, v = qkv.split(self.n_embd, dim=-1)

        k, v = self.kpool.prepend(self.block_index_number, write_slots, tot_slots, k, v)

        key_ind = torch.arange(len(k))

        if self.block_index_number == 0:
          print("saved: ", len(k), len(v), len(p))

        # [T, C] -> [num_heads, T, head_dim]
        q = q.view(
            total_tokens,
            self.n_head,
            self.head_dim
        ).transpose(0, 1)

        k = k.view(
            len(k),
            self.n_head,
            self.head_dim
        ).transpose(0, 1)

        v = v.view(
            len(v),
            self.n_head,
            self.head_dim
        ).transpose(0, 1)

        # Q @ K^T
        # [H, T, D] @ [H, D, T]
        # -> [H, T, T]
        attention_scores = q @ k.transpose(-2, -1)

        # Scale by sqrt(head_dim)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        causal_mask = key_ind <= p.reshape(len(p), 1)
        same_request_mask = request_ids.reshape(len(request_ids), 1) == new_req_ids


        # # Causal mask: token i cannot attend to tokens after i
        # causal_mask = torch.tril(
        #     torch.ones(
        #         total_tokens,
        #         total_tokens,
        #         device=x.device,
        #         dtype=torch.bool
        #     )
        # )

        # # Block-diagonal mask: token i can only attend to tokens from the
        # # same request. Packing is order-preserving per request, so
        # # combined with the causal mask above this reproduces exactly what
        # # each request would see if it were run through attention alone.
        # same_request_mask = request_ids.unsqueeze(0) == request_ids.unsqueeze(1)

        attention_mask = causal_mask & same_request_mask

        attention_scores = attention_scores.masked_fill(
            ~attention_mask,
            float("-inf")
        )

        # Convert scores into probabilities
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1
        )

        # Weighted sum of values
        # [H, T, T] @ [H, T, D]
        # -> [H, T, D]
        y = attention_probs @ v

        # Combine attention heads
        # [H, T, D] -> [T, H, D] -> [T, C]
        y = y.transpose(0, 1).contiguous()
        y = y.view(total_tokens, self.n_embd)

        # Final output projection
        y = self.c_proj(y)

        return y

#### This MLP Class is vibe coded ####
class MLP(nn.Module):
    def __init__(self, config, hf_mlp):
        super().__init__()

        self.fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd
        )

        self.proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd
        )

        # Load pretrained Hugging Face weights (see note on Conv1D transpose
        # in Attention.__init__)
        with torch.no_grad():
            self.fc.weight.copy_(hf_mlp.c_fc.weight.t())
            self.fc.bias.copy_(hf_mlp.c_fc.bias)

            self.proj.weight.copy_(hf_mlp.c_proj.weight.t())
            self.proj.bias.copy_(hf_mlp.c_proj.bias)

    def forward(self, x):
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.proj(x)

        return x
