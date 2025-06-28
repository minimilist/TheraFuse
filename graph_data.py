import torch
from transformers import LlamaTokenizer, LlamaModel, AutoTokenizer
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tqdm

@dataclass
class DialogueDataset:
    utterance_embeddings: Dict[str, Dict[int, torch.Tensor]]
    summaries: Dict[str, str]
    prompts: Dict[str, str]
    links: Dict[str, List[List[int]]]
    link_types: Dict[str, List[int]]
    
    def __len__(self):
        return len(self.utterance_embeddings)
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, str, str, List[List[int]], List[int]]:
        dialogue_id = list(self.utterance_embeddings.keys())[idx]
        return {
            'utterance_embeddings': self.utterance_embeddings[dialogue_id],
            'summary': self.summaries[dialogue_id],
            'prompt': self.prompts[dialogue_id],
            'links_ids': self.links[dialogue_id],
            'link_type': self.link_types[dialogue_id]
        }

class DialogueProcessor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = LlamaModel.from_pretrained(model_path, device_map='auto')
        self.model.eval()
        
    def get_embeddings(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            text = self.tokenizer.bos_token + text
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            # print(hidden_states)
            bos_embedding = hidden_states[:, 0, :][0]
            # print(bos_embedding)
            return bos_embedding.squeeze(0)
            
    def process_dialogue(self, dialogue: str) -> Dict[int, torch.Tensor]:
        utterances = dialogue.split(';')
        utterance_embeddings = []
        
        for idx, utterance in enumerate(utterances):
            if ':' not in utterance:
                continue
            embeddings = self.get_embeddings(utterance.split(':')[1])
            utterance_embeddings.append(embeddings.cpu().detach())
                
        return utterance_embeddings
        
    def create_dataset(self, data: List[Dict[str, Any]]) -> DialogueDataset:
        utterance_embeddings = {}
        summaries = {}
        prompts = {}
        links = {}
        link_types = {}
        
        for item in tqdm.tqdm(data):
            dialogue_id = item['id']
            utterance_embeddings[dialogue_id] = self.process_dialogue(item['dialogue'])
            summaries[dialogue_id] = item['summary']
            prompts[dialogue_id] = f"Prepare psychotherapy notes for this converstation with fields like 'Patient Particulars', 'Clinical Identifiers', 'Referral Information', 'Therapist Information', 'Past Session Information', 'Presenting Complaints (Symptoms)', 'History', 'Crisis Markers', 'Current Mental Status Examination', 'Psychotherapy Type', 'Psychotherapy Technique', 'Assessments', 'Issues discussed in Current Session', 'Reflections by the therapist', 'Clinical Diagnosis by Reviewer', 'Action Plan', 'Next session details': {item['dialogue']}"
            links[dialogue_id] = item['links']
            link_types[dialogue_id] = item['link_types']
            # print(utterance_embeddings[dialogue_id])

        return DialogueDataset(
            utterance_embeddings=utterance_embeddings,
            summaries=summaries,
            prompts=prompts,
            links=links,
            link_types=link_types
        )
        
    def save_dataset(self, dataset: DialogueDataset, output_path: str):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        embeddings_dict = {
            dialogue_id: [emb.numpy().tolist() for emb in utterances]
            for dialogue_id, utterances in dataset.utterance_embeddings.items()
        }
        
        metadata = {
            'summaries': dataset.summaries,
            'prompts': dataset.prompts,
            'links': dataset.links,
            'link_types': dataset.link_types
        }
        
        torch.save(embeddings_dict, output_path / 'utterance_embeddings.pt')
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

def load_dataset(data_path: str) -> DialogueDataset:
    data_path = Path(data_path)
    
    # Load embeddings
    embeddings_dict = torch.load(data_path / 'utterance_embeddings.pt')
    utterance_embeddings = {}
    for dialogue_id, utterances in embeddings_dict.items():
        if len(utterances) == 0:
            print(f"{dialogue_id} has 0 utterances")
        else:
            utterance_embeddings[dialogue_id] = torch.stack([torch.tensor(emb) for emb in utterances])
    # utterance_embeddings = {
    #     dialogue_id: torch.stack([torch.tensor(emb) for emb in utterances])
    #     for dialogue_id, utterances in embeddings_dict.items()
    # }
    
    # Load metadata
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return DialogueDataset(
        utterance_embeddings=utterance_embeddings,
        summaries=metadata['summaries'],
        prompts=metadata['prompts'],
        links=metadata['links'],
        link_types=metadata['link_types']
    )

# PyTorch DataLoader example
class DialogueDataLoader(torch.utils.data.Dataset):
    def __init__(self, dataset: DialogueDataset):
        self.dataset = dataset
        self.dialogue_ids = list(dataset.summaries.keys())
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

# Example usage for training
def get_training_dataloader(data_path: str, batch_size: int = 32):
    dataset = load_dataset(data_path)
    dataloader = torch.utils.data.DataLoader(
        DialogueDataLoader(dataset),
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader

if __name__ == "__main__":
    dialogue_processor = DialogueProcessor('/home/models/Meta-Llama-3.1-8B-Instruct')
    with open('data/graph_test_aiims.json', 'r') as file:
        data = json.load(file)
    Dataset = dialogue_processor.create_dataset(data)
    print(Dataset)
    print("dataset created successfully!")
    dialogue_processor.save_dataset(Dataset, 'data/aiims_test')
    print('Dataset saved!')