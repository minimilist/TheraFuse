from torch_geometric.data import Data, Batch
from graph_data import DialogueDataset, DialogueProcessor, load_dataset
from torch.utils.data import Dataset, DataLoader
import torch


class ConversationDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_seq_length=512):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.hidden_size = tokenizer.model_max_length  # Adjust this based on your model

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]

        # Tokenize prompt and summary
        prompt = data_item['prompt'] + " ### RESPONSE:"
        summary = data_item['summary']

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        summary_ids = self.tokenizer.encode(summary, add_special_tokens=False)

        # Truncate if necessary
        total_length = len(prompt_ids) + len(summary_ids) + 1  # Plus EOS token
        if total_length > self.max_seq_length:
            excess = total_length - self.max_seq_length
            # Truncate prompt_ids if needed
            if len(prompt_ids) > excess:
                prompt_ids = prompt_ids[excess:]
            else:
                # If prompt_ids are not enough, trim from summary_ids
                summary_ids = summary_ids[excess - len(prompt_ids):]
                prompt_ids = []

        # Create input_ids (prompt + summary + eos)
        input_ids = prompt_ids

        # Labels are -100 for prompt tokens, then summary_ids + eos token
        labels = summary_ids + [self.tokenizer.eos_token_id]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Edge information for graph
        utterance_embeddings = data_item['utterance_embeddings']
        # print(f"utterance_embeddings: {utterance_embeddings}")  # Tensor of shape (num_utterances, hidden_size)
        links_ids = data_item['links_ids']  # List of [source, target]
        link_type = data_item['link_type']  # List of edge types

        if len(links_ids) == 0:
            # Handle case with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor(links_ids, dtype=torch.long).t().contiguous()  # Shape [2, num_edges]
            edge_type = torch.tensor(link_type, dtype=torch.long)  # Shape [num_edges]

        # Create Data object (from torch_geometric)
        graph_data = Data(
            x=utterance_embeddings,  # Node features
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=utterance_embeddings.size(0),
        )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'graph_data': graph_data
        }
    

def collate_fn(batch):
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    graph_data_list = [item['graph_data'] for item in batch]

    # Pad sequences to the maximum length in the batch
    max_seq_len = max([input_ids.size(0) for input_ids in input_ids_list])

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
        seq_len = input_ids.size(0)
        padding_len = max_seq_len - seq_len
        if padding_len > 0:
            padded_input_ids.append(torch.cat([input_ids, torch.zeros(padding_len, dtype=torch.long)]))
            padded_attention_mask.append(torch.cat([attention_mask, torch.zeros(padding_len, dtype=torch.long)]))
            padded_labels.append(torch.cat([labels, torch.full((padding_len,), -100, dtype=torch.long)]))
        else:
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_labels.append(labels)

    input_ids = torch.stack(padded_input_ids, dim=0)
    attention_mask = torch.stack(padded_attention_mask, dim=0)
    labels = torch.stack(padded_labels, dim=0)

    # Batch the graph data using torch_geometric's Batch
    batched_graph = Batch.from_data_list(graph_data_list)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'x': batched_graph.x,
        'edge_index': batched_graph.edge_index,
        'edge_type': batched_graph.edge_type,
        'node_to_graph': batched_graph.batch
    }