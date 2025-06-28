"""
Conversation Summarization Model Training Script

This script trains a conversation summarization model that combines LLaMA with graph neural networks
for enhanced understanding of dialogue structure and relationships.
"""

import os
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
from torch_geometric.nn import RGCNConv
from torch_geometric.data import HeteroData, Data, Batch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    LlamaConfig, 
    LlamaModel, 
    LlamaForCausalLM,
    LlamaTokenizer, 
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# Local imports
from llama_fusion import LLaMAWithIntermediateFusion
from fusion_model import ConversationSummarizationModel, ConversationGraphModel
from graph_data import DialogueDataset, DialogueProcessor, load_dataset


class ConversationDataset(Dataset):
    """
    Dataset class for conversation summarization with graph structure.
    
    This dataset handles tokenization of conversations and preparation of graph data
    for training the conversation summarization model.
    """
    
    def __init__(
        self, 
        data_list: List[Dict[str, Any]], 
        tokenizer: AutoTokenizer, 
        max_seq_length: int = 7000
    ):
        """
        Initialize the conversation dataset.
        
        Args:
            data_list: List of conversation data items
            tokenizer: Tokenizer for text processing
            max_seq_length: Maximum sequence length for truncation
        """
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.hidden_size = tokenizer.model_max_length

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing tokenized inputs and graph data
        """
        data_item = self.data_list[idx]

        # Prepare text data
        prompt = data_item['prompt'] + " ### RESPONSE:"
        summary = data_item['summary']

        # Tokenize sequences
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        summary_ids = self.tokenizer.encode(summary, add_special_tokens=False)

        # Handle sequence length constraints
        prompt_ids, summary_ids = self._truncate_sequences(prompt_ids, summary_ids)

        # Create input sequences
        input_ids = prompt_ids + summary_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + summary_ids + [self.tokenizer.eos_token_id]

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Prepare graph data
        graph_data = self._prepare_graph_data(data_item)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'graph_data': graph_data
        }

    def _truncate_sequences(
        self, 
        prompt_ids: List[int], 
        summary_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Truncate sequences to fit within maximum length constraints.
        
        Args:
            prompt_ids: Tokenized prompt sequence
            summary_ids: Tokenized summary sequence
            
        Returns:
            Tuple of truncated prompt and summary sequences
        """
        total_length = len(prompt_ids) + len(summary_ids) + 1  # Plus EOS token
        
        if total_length > self.max_seq_length:
            excess = total_length - self.max_seq_length
            
            if len(prompt_ids) > excess:
                prompt_ids = prompt_ids[excess:]
            else:
                summary_ids = summary_ids[excess - len(prompt_ids):]
                prompt_ids = []
                
        return prompt_ids, summary_ids

    def _prepare_graph_data(self, data_item: Dict[str, Any]) -> Data:
        """
        Prepare graph data from the conversation item.
        
        Args:
            data_item: Dictionary containing conversation data
            
        Returns:
            PyTorch Geometric Data object
        """
        utterance_embeddings = data_item['utterance_embeddings']
        links_ids = data_item['links_ids']
        link_type = data_item['link_type']

        if len(links_ids) == 0:
            # Handle case with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor(links_ids, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(link_type, dtype=torch.long)

        return Data(
            x=utterance_embeddings,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=utterance_embeddings.size(0),
        )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching conversation data.
    
    Args:
        batch: List of conversation data items
        
    Returns:
        Batched data dictionary
    """
    # Extract components
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    graph_data_list = [item['graph_data'] for item in batch]

    # Pad sequences to maximum length in batch
    max_seq_len = max(input_ids.size(0) for input_ids in input_ids_list)

    # Pad each sequence
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
        seq_len = input_ids.size(0)
        padding_len = max_seq_len - seq_len
        
        if padding_len > 0:
            padded_input_ids.append(
                torch.cat([input_ids, torch.zeros(padding_len, dtype=torch.long)])
            )
            padded_attention_mask.append(
                torch.cat([attention_mask, torch.zeros(padding_len, dtype=torch.long)])
            )
            padded_labels.append(
                torch.cat([labels, torch.full((padding_len,), -100, dtype=torch.long)])
            )
        else:
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_labels.append(labels)

    # Stack into tensors
    input_ids = torch.stack(padded_input_ids, dim=0)
    attention_mask = torch.stack(padded_attention_mask, dim=0)
    labels = torch.stack(padded_labels, dim=0)

    # Batch graph data
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


class ConversationTrainer:
    """
    Trainer class for the conversation summarization model.
    """
    
    def __init__(
        self,
        model_path: str,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda:0',
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0,
        max_seq_length: int = 7000
    ):
        """
        Initialize the trainer.
        
        Args:
            model_path: Path to the base model
            checkpoint_path: Path to model checkpoint (optional)
            device: Device to use for training
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            max_seq_length: Maximum sequence length
        """
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.max_seq_length = max_seq_length
        
        # Initialize model and tokenizer
        self.model = self._initialize_model(model_path, checkpoint_path)
        self.tokenizer = self._initialize_tokenizer(model_path)
        self.optimizer = None
        self.scheduler = None
        
    def _initialize_model(
        self, 
        model_path: str, 
        checkpoint_path: Optional[str] = None
    ) -> ConversationSummarizationModel:
        """Initialize the conversation summarization model."""
        model = ConversationSummarizationModel(model_path)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
        model.to(self.device)
        return model
        
    def _initialize_tokenizer(self, model_path: str) -> AutoTokenizer:
        """Initialize the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
        
    def _setup_optimization(self, total_steps: int, warmup_steps: int = 3):
        """Setup optimizer and learning rate scheduler."""
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(trainable_parameters, lr=self.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"Number of trainable parameters: "
              f"{self.model.fused_model.trainable_parameters()}/{self.model.fused_model.total_parameters()}")
    
    def load_optimizer_state(self, checkpoint_path: str):
        """Load optimizer state from checkpoint."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer state to device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            enumerate(dataloader), 
            desc=f"Epoch {epoch+1}", 
            total=len(dataloader)
        )
        
        for step, batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                x=batch['x'],
                edge_index=batch['edge_index'],
                edge_type=batch['edge_type'],
                node_to_graph=batch['node_to_graph']
            )
            
            loss = outputs['loss'] / self.gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation and optimization step
            if ((step + 1) % self.gradient_accumulation_steps == 0) or ((step + 1) == len(dataloader)):
                clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        
        return epoch_loss / len(dataloader)
    
    def save_checkpoint(self, epoch: int, loss: float, save_path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, save_path)
        
        print(f"Model saved to {save_path}")


def validate_data(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and filter the dataset.
    
    Args:
        data_list: List of conversation data items
        
    Returns:
        Filtered list of valid data items
    """
    valid_data = []
    
    for i, data_item in enumerate(data_list):
        try:
            # Basic validation - check if item is accessible
            _ = data_item['prompt']
            _ = data_item['summary']
            valid_data.append(data_item)
        except (KeyError, TypeError, IndexError) as e:
            print(f"Data item {i} not accessible: {e}")
            continue
    
    return valid_data


def main():
    """Main training function."""
    # Configuration
    CONFIG = {
        'model_path': '/home/models/Meta-Llama-3.1-8B-Instruct',
        'checkpoint_path': 'fusion_models_aiims_3/conversation_summarization_model_epoch_6.pt',
        'data_path': 'data/aiims_train',
        'output_dir': 'fusion_models_aiims_3',
        'device': 'cuda:0',
        'batch_size': 1,
        'num_epochs': 20,
        'max_seq_length': 7000,
        'learning_rate': 1e-5,
        'gradient_accumulation_steps': 2,
        'max_grad_norm': 1.0,
        'save_every_n_epochs': 2
    }
    
    # Load and validate data
    print("Loading dataset...")
    data_list = load_dataset(CONFIG['data_path'])
    data_list = validate_data(data_list)
    print(f"Loaded {len(data_list)} valid data items")
    
    # Print sample data
    if len(data_list) > 0:
        print("\nSample data items:")
        for i in [0, min(5, len(data_list)-1), min(100, len(data_list)-1)]:
            if i < len(data_list):
                print(f"Data item {i}: {type(data_list[i])}")
    
    # Initialize trainer
    trainer = ConversationTrainer(
        model_path=CONFIG['model_path'],
        checkpoint_path=CONFIG['checkpoint_path'],
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        max_grad_norm=CONFIG['max_grad_norm'],
        max_seq_length=CONFIG['max_seq_length']
    )
    
    # Create dataset and dataloader
    dataset = ConversationDataset(
        data_list, 
        trainer.tokenizer, 
        max_seq_length=CONFIG['max_seq_length']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Setup optimization
    total_steps = len(dataloader) * CONFIG['num_epochs'] // CONFIG['gradient_accumulation_steps']
    trainer._setup_optimization(total_steps)
    trainer.load_optimizer_state(CONFIG['checkpoint_path'])
    
    # Training loop
    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    
    for epoch in range(CONFIG['num_epochs']):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % CONFIG['save_every_n_epochs'] == 0:
            save_path = os.path.join(
                CONFIG['output_dir'], 
                f"conversation_summarization_model_epoch_{epoch+1}.pt"
            )
            trainer.save_checkpoint(epoch + 1, avg_loss, save_path)
    
    print("Training completed!")


if __name__ == "__main__":
    main()