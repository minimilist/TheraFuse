import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaModel
import bitsandbytes as bnb
from transformers.utils import WEIGHTS_NAME
from transformers.utils.hub import cached_file
from transformers.modeling_outputs import CausalLMOutputWithPast

class RobustCrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, llama_hidden, gnn_embeds, attention_mask=None):
        # gnn_embeds.to(dtype=torch.float16)
        # print(f"gnn_embeds dtype: {gnn_embeds.dtype}")
        # Prepare query, key, value
        query = llama_hidden.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        key = gnn_embeds.transpose(0, 1)  # (num_utterances, batch_size, hidden_size)
        value = gnn_embeds.transpose(0, 1)  # (num_utterances, batch_size, hidden_size)

        # print(f"query dtype: {query.dtype}")
        # print(f"key dtype: {key.dtype}")
        # print(f"attention weights dtype: {self.attention.in_proj_weight.dtype}")
        
        # # Check if attention_mask is provided
        # if attention_mask is not None:
        #     print(f"attention_mask dtype: {attention_mask.dtype}")
        # Multi-head attention
        attn_output, _ = self.attention(query=query, key=key, value=value, attn_mask=None)
        attn_output = attn_output.transpose(0, 1)  # Back to (batch_size, seq_len, hidden_size)

        # Residual connection and layer normalization
        out = self.layer_norm1(llama_hidden + attn_output)

        # Feed-forward layer
        ff_out = self.ff(out)
        out = self.layer_norm2(out + ff_out)
        
        return out.to(dtype=llama_hidden.dtype)

class RobustIntermediateFusionLLaMABlock(nn.Module):
    def __init__(self, llama_layer, hidden_size):
        super().__init__()
        self.llama_layer = llama_layer
        self.fusion = RobustCrossAttentionFusion(hidden_size)
        
    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        position_ids=None, 
        gnn_embeds=None, 
        **kwargs
    ):
        # Ensure position_ids exists
        if position_ids is None:
            position_ids = torch.arange(
                0, hidden_states.size(1), 
                dtype=torch.long, 
                device=hidden_states.device
            ).unsqueeze(0).repeat(hidden_states.size(0), 1)
        
        # Handle case with no GNN embeddings
        if gnn_embeds is None:
            return self.llama_layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                **kwargs
            )
        
        # Process through LLaMA layer first
        llama_output = self.llama_layer(
            hidden_states, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            **kwargs
        )
        
        # Extract hidden states
        if isinstance(llama_output, tuple):
            hidden_states = llama_output[0]
            outputs = llama_output[1:]
        else:
            hidden_states = llama_output
            outputs = None
        
        # Apply cross-attention fusion
        fused_states = self.fusion(hidden_states, gnn_embeds, attention_mask)
        
        # Return with original outputs if present
        if outputs is not None:
            return (fused_states,) + outputs
        return fused_states
    
class CustomLlamaModel(LlamaModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        gnn_embeds=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position= None,
    ):
        # Adapted from LlamaModel.forward()

        # Set default values for return_dict and use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Check that either input_ids or inputs_embeds is provided
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Prepare inputs_embeds if not provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Initialize past_key_values if not provided
        if use_cache and past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_key = past_key_values[0]  # This is a tuple (key_tensor, value_tensor)
            key_tensor = past_key[0]       # key_tensor
            past_seq_length = key_tensor.size(2)  # The third dimension is seq_length
        else:
            past_seq_length = 0

        # Compute attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device)

        if use_cache and past_key_values is None:
            past_key_values = [None for _ in range(len(self.layers))]

        if cache_position is None:
            past_seen_tokens = past_seq_length
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        # Create causal mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # Position ids
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embed positions
        hidden_states = inputs_embeds

        # Layer norm
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # Initialize variables for outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # Iterate over layers
        for idx, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Check if the layer is a fusion layer
            if isinstance(layer_module, RobustIntermediateFusionLLaMABlock):
                # Pass gnn_embeds to the fusion layers
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    gnn_embeds=gnn_embeds,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            else:
                # Regular layers that don't use gnn_embeds
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_past_key_value = layer_outputs[1]
                past_key_values[idx] = next_past_key_value

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)

        if use_cache:
            outputs += (past_key_values,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_self_attentions,)

        if return_dict:
            return {
                "last_hidden_state": outputs[0],
                "past_key_values": outputs[1] if use_cache else None,
                "hidden_states": outputs[2] if output_hidden_states else None,
                "attentions": outputs[3] if output_attentions else None,
            }
        else:
            return outputs

class LLaMAWithIntermediateFusion(LlamaForCausalLM):
    def __init__(self, model_id, fusion_layers=None, quantization_config=None, device_map=None):
        base_model = LlamaForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float32,  # Use float16 to save memory
        )

        # Initialize base class with configuration
        super().__init__(base_model.config)

        # Initialize your custom model
        self.model = CustomLlamaModel(base_model.config)

        # Load weights into self.model and self.lm_head
        self.model.load_state_dict(base_model.model.state_dict(), strict=False)
        self.lm_head.load_state_dict(base_model.lm_head.state_dict(), strict=False)
        del base_model
        self.model.to(device='cuda:0')
        self.lm_head.to(device='cuda:0')
        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        

        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers

        # Default fusion layers
        fusion_layers = fusion_layers or list(range(0, num_layers, 3))

        # Replace specified layers with fusion-enabled versions
        for idx in fusion_layers:
            original_layer = self.model.layers[idx]
            fusion_layer = RobustIntermediateFusionLLaMABlock(
                llama_layer=original_layer, hidden_size=hidden_size
            ).to(original_layer.self_attn.q_proj.weight.device)
            # Unfreeze fusion-related parameters
            for name, param in fusion_layer.named_parameters():
                if 'fusion' in name:
                    param.requires_grad = True

            self.model.layers[idx] = fusion_layer

        # print('Model now looks like:')
        # print(self.model)
            # Additional initialization code can go here if needed

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # Prepare inputs for generation, including gnn_embeds
        # We'll assume that gnn_embeds are passed in kwargs

        # Retrieve gnn_embeds from kwargs
        gnn_embeds = kwargs.get('gnn_embeds', None)
        # node_to_graph = kwargs.get('node_to_graph', None)  # If needed

        # Prepare model inputs
        input_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': False,  # Enable caching for faster generation
            'gnn_embeds': gnn_embeds,
            'output_attentions': False,
            'output_hidden_states': False,
        }
        return input_dict

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        gnn_embeds=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Forward pass through the custom LlamaModel
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            gnn_embeds=gnn_embeds,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs['last_hidden_state'] if return_dict else transformer_outputs[0]

        # Compute logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.get('past_key_values'),
            hidden_states=transformer_outputs.get('hidden_states'),
            attentions=transformer_outputs.get('attentions')
        )
    
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())
