import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import HeteroData
from llama_fusion import LLaMAWithIntermediateFusion

class ConversationGraphModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_layers):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
        
        # Output layer
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, edge_type):
        # print(f"x dtype: {x.dtype}")
        for idx, conv in enumerate(self.convs):
            # for name, param in conv.named_parameters():
            #     print(f"{name} dtype: {param.dtype}")
            x = conv(x, edge_index, edge_type)
            if idx < self.num_layers - 1:
                x = self.relu(x)
                x = self.dropout(x)
        return x  # Updated node embeddings
    
class ConversationSummarizationModel(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.fused_model = LLaMAWithIntermediateFusion(
            model_id=model_id,
            fusion_layers=[27, 29, 31],
            device_map='auto',  
        )  
        self.hidden_size = self.fused_model.config.hidden_size
        self.graph_model = ConversationGraphModel(in_channels=self.hidden_size, 
                                                  hidden_channels=self.hidden_size, 
                                                  out_channels=self.hidden_size, 
                                                  num_relations=17, 
                                                  num_layers=3
        )  

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        x=None,
        edge_index=None,
        edge_type=None,
        node_to_graph=None,
    ):
        # Process the graph data with the GNN
        # x: Node features (utterance embeddings)
        # edge_index: Edge indices
        # edge_type: Edge types
        node_embeddings = self.graph_model(x, edge_index, edge_type)  # (num_nodes, hidden_size) 
        batch_size = input_ids.size(0)
        hidden_size = node_embeddings.size(1)

        gnn_embeds_per_sample = []
        max_num_nodes = 0

        # First, collect node embeddings per sample and find the maximum number of nodes
        for i in range(batch_size):
            node_mask = (node_to_graph == i)
            sample_node_embeddings = node_embeddings[node_mask]  # Shape: (num_nodes_i, hidden_size)
            num_nodes_i = sample_node_embeddings.size(0)
            if num_nodes_i > max_num_nodes:
                max_num_nodes = num_nodes_i
            gnn_embeds_per_sample.append(sample_node_embeddings)

        # Then, pad node embeddings per sample to have the same length
        padded_gnn_embeds = []

        for emb in gnn_embeds_per_sample:
            num_nodes = emb.size(0)
            padding_size = max_num_nodes - num_nodes
            if padding_size > 0:
                pad = torch.zeros(padding_size, hidden_size, device=emb.device, dtype=emb.dtype)
                emb_padded = torch.cat([emb, pad], dim=0)
            else:
                emb_padded = emb
            padded_gnn_embeds.append(emb_padded)

        # Stack to form tensors
        gnn_embeds = torch.stack(padded_gnn_embeds, dim=0)
        # gnn_embeds.to(dtype=torch.float16)
        # print(f"gnn_embeds dtype : {gnn_embeds.dtype}")
        # Pass inputs to fused_model
        outputs = self.fused_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            gnn_embeds=gnn_embeds,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True
        )
        
        return outputs

    def generate(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            x=None,
            edge_index=None,
            edge_type=None,
            node_to_graph=None,
            model_max_length=None
    ):
        node_embeddings = self.graph_model(x, edge_index, edge_type)  # (num_nodes, hidden_size) 
        batch_size = input_ids.size(0)
        hidden_size = node_embeddings.size(1)

        gnn_embeds_per_sample = []
        max_num_nodes = 0

        # First, collect node embeddings per sample and find the maximum number of nodes
        for i in range(batch_size):
            node_mask = (node_to_graph == i)
            sample_node_embeddings = node_embeddings[node_mask]  # Shape: (num_nodes_i, hidden_size)
            num_nodes_i = sample_node_embeddings.size(0)
            if num_nodes_i > max_num_nodes:
                max_num_nodes = num_nodes_i
            gnn_embeds_per_sample.append(sample_node_embeddings)

        # Then, pad node embeddings per sample to have the same length
        padded_gnn_embeds = []

        for emb in gnn_embeds_per_sample:
            num_nodes = emb.size(0)
            padding_size = max_num_nodes - num_nodes
            if padding_size > 0:
                pad = torch.zeros(padding_size, hidden_size, device=emb.device, dtype=emb.dtype)
                emb_padded = torch.cat([emb, pad], dim=0)
            else:
                emb_padded = emb
            padded_gnn_embeds.append(emb_padded)

        # Stack to form tensors
        gnn_embeds = torch.stack(padded_gnn_embeds, dim=0)
        gnn_embeds.to(dtype=torch.float16)

        generated_ids = self.fused_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=model_max_length,
                num_beams=5,
                early_stopping=True,
                gnn_embeds=gnn_embeds,
            )
        
        return generated_ids
