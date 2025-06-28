"""
Conversation Summarization Model Evaluation Script

This script evaluates a trained conversation summarization model by generating
responses and comparing them with reference responses. It includes support for
various evaluation metrics and comprehensive result logging.

Author: [Your Name]
Date: [Current Date]
License: [Your License]
"""

import os
import json
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
import nltk
from tqdm import tqdm
from torch_geometric.nn import RGCNConv
from torch_geometric.data import HeteroData, Data, Batch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Local imports
from llama_fusion import LLaMAWithIntermediateFusion
from fusion_model import ConversationSummarizationModel, ConversationGraphModel
from graph_data import DialogueDataset, DialogueProcessor, load_dataset
from data_utils import ConversationDataset, collate_fn

# Environment setup
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ConversationEvaluator:
    """
    Evaluator class for conversation summarization models.
    
    This class handles model loading, dataset preparation, inference,
    and evaluation metric calculations.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = 'cuda:0',
        max_seq_length: int = 7000,
        batch_size: int = 1
    ):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            tokenizer_path: Path to the tokenizer
            device: Device to use for evaluation
            max_seq_length: Maximum sequence length for generation
            batch_size: Batch size for evaluation
        """
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        
        # Initialize model and tokenizer
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
        # Initialize evaluation metrics
        self.smoothie = SmoothingFunction().method4
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
    
    def _load_model(self) -> ConversationSummarizationModel:
        """
        Load the conversation summarization model from checkpoint.
        
        Returns:
            Loaded and initialized model
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        print(f"Loading model from {self.checkpoint_path}")
        
        # Initialize model
        model = ConversationSummarizationModel(self.tokenizer_path)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to evaluation mode
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """
        Load and configure the tokenizer.
        
        Returns:
            Configured tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
    
    def _prepare_dataset(self, test_data_path: str) -> DataLoader:
        """
        Prepare the test dataset and dataloader.
        
        Args:
            test_data_path: Path to the test data
            
        Returns:
            Configured DataLoader for evaluation
        """
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found: {test_data_path}")
        
        print(f"Loading test dataset from {test_data_path}")
        
        # Load test data
        test_data_list = load_dataset(test_data_path)
        print(f"Loaded {len(test_data_list)} test samples")
        
        if len(test_data_list) > 0:
            print(f"Sample data structure: {type(test_data_list[0])}")
        
        # Create dataset and dataloader
        test_dataset = ConversationDataset(
            test_data_list, 
            self.tokenizer, 
            max_seq_length=self.max_seq_length
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        return test_dataloader
    
    def _generate_response(self, batch: Dict[str, torch.Tensor]) -> str:
        """
        Generate response for a single batch.
        
        Args:
            batch: Input batch data
            
        Returns:
            Generated response text
        """
        try:
            # Move batch data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            x = batch['x'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_type = batch['edge_type'].to(self.device)
            node_to_graph = batch['node_to_graph'].to(self.device)
            
            # Generate response
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                node_to_graph=node_to_graph,
                model_max_length=self.max_seq_length
            )
            
            # Decode generated text
            generated_text = self.tokenizer.batch_decode(
                generated_ids.detach().cpu().numpy(), 
                skip_special_tokens=True
            )[0]
            
            # Extract response part
            generated_response = generated_text.strip().split("### RESPONSE:")[-1].strip()
            return generated_response
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"ERROR: {str(e)}"
    
    def _extract_reference_and_prompt(self, batch: Dict[str, torch.Tensor]) -> Tuple[str, str]:
        """
        Extract reference response and prompt from batch.
        
        Args:
            batch: Input batch data
            
        Returns:
            Tuple of (prompt, reference_response)
        """
        # Decode reference summaries
        reference_summaries = self.tokenizer.batch_decode(
            batch['labels'].cpu().numpy(),
            skip_special_tokens=True
        )
        
        # Decode prompts
        prompts = self.tokenizer.batch_decode(
            batch['input_ids'].cpu().numpy(),
            skip_special_tokens=True
        )
        
        return prompts[0] if prompts else "", reference_summaries[0] if reference_summaries else ""
    
    def _calculate_metrics(self, generated_responses: List[str], reference_responses: List[str]) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            generated_responses: List of generated responses
            reference_responses: List of reference responses
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = {}
        
        # Prepare data for BLEU calculation
        references = [[ref.split()] for ref in reference_responses if ref.strip()]
        hypotheses = [gen.split() for gen in generated_responses if gen.strip()]
        
        if references and hypotheses and len(references) == len(hypotheses):
            # Calculate BLEU scores
            try:
                bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=self.smoothie)
                bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothie)
                bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothie)
                
                metrics['bleu_1'] = bleu_1
                metrics['bleu_2'] = bleu_2
                metrics['bleu_4'] = bleu_4
            except Exception as e:
                print(f"Error calculating BLEU scores: {e}")
                metrics['bleu_1'] = 0.0
                metrics['bleu_2'] = 0.0
                metrics['bleu_4'] = 0.0
            
            # Calculate ROUGE scores
            try:
                rouge_1_scores = []
                rouge_2_scores = []
                rouge_l_scores = []
                
                for gen, ref in zip(generated_responses, reference_responses):
                    if gen.strip() and ref.strip():
                        scores = self.rouge_scorer.score(ref, gen)
                        rouge_1_scores.append(scores['rouge1'].fmeasure)
                        rouge_2_scores.append(scores['rouge2'].fmeasure)
                        rouge_l_scores.append(scores['rougeL'].fmeasure)
                
                metrics['rouge_1'] = sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0.0
                metrics['rouge_2'] = sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0.0
                metrics['rouge_l'] = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
                
            except Exception as e:
                print(f"Error calculating ROUGE scores: {e}")
                metrics['rouge_1'] = 0.0
                metrics['rouge_2'] = 0.0
                metrics['rouge_l'] = 0.0
        else:
            print("Warning: Unable to calculate metrics due to data mismatch")
            metrics = {metric: 0.0 for metric in ['bleu_1', 'bleu_2', 'bleu_4', 'rouge_1', 'rouge_2', 'rouge_l']}
        
        return metrics
    
    def evaluate(
        self, 
        test_data_path: str, 
        output_path: str = "evaluation_results.json",
        skip_samples: int = 0,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data_path: Path to test data
            output_path: Path to save evaluation results
            skip_samples: Number of samples to skip from the beginning
            max_samples: Maximum number of samples to evaluate (None for all)
            verbose: Whether to print detailed outputs
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        # Prepare dataset
        test_dataloader = self._prepare_dataset(test_data_path)
        
        # Initialize storage for results
        results = []
        generated_responses = []
        reference_responses = []
        sample_count = 0
        
        print(f"Starting evaluation (skipping first {skip_samples} samples)")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
                # Skip samples if requested
                if batch_idx < skip_samples:
                    continue
                
                # Check max samples limit
                if max_samples and sample_count >= max_samples:
                    break
                
                sample_count += 1
                
                # Extract prompt and reference
                prompt, reference_response = self._extract_reference_and_prompt(batch)
                
                # Generate response
                generated_response = self._generate_response(batch)
                
                # Store results
                result_item = {
                    "sample_id": batch_idx,
                    "prompt": prompt,
                    "reference_response": reference_response,
                    "generated_response": generated_response,
                }
                results.append(result_item)
                
                # Store for metric calculation
                if reference_response.strip():  # Only include non-empty references
                    generated_responses.append(generated_response)
                    reference_responses.append(reference_response)
                
                # Verbose output
                if verbose:
                    print(f"\n{'='*100}")
                    print(f"SAMPLE {batch_idx + 1}")
                    print(f"{'='*100}")
                    print(f"PROMPT: {prompt}")
                    print(f"{'-'*100}")
                    print(f"REFERENCE: {reference_response}")
                    print(f"{'-'*100}")
                    print(f"GENERATED: {generated_response}")
                    print(f"{'='*100}")
        
        # Calculate metrics
        print(f"\nCalculating metrics for {len(generated_responses)} valid samples...")
        metrics = self._calculate_metrics(generated_responses, reference_responses)
        
        # Prepare final results
        evaluation_results = {
            "evaluation_info": {
                "checkpoint_path": self.checkpoint_path,
                "test_data_path": test_data_path,
                "total_samples": len(results),
                "valid_samples_for_metrics": len(generated_responses),
                "skip_samples": skip_samples,
                "max_samples": max_samples
            },
            "metrics": metrics,
            "samples": results
        }
        
        # Save results
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {output_path}")
        print(f"\nMetrics Summary:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        return evaluation_results


def main():
    """Main evaluation function."""
    # Configuration
    CONFIG = {
        'checkpoint_path': "fusion_models_aiims/conversation_summarization_model_epoch_10.pt",
        'test_data_path': "data/aiims_test",
        'tokenizer_path': "/home/models/Meta-Llama-3.1-8B-Instruct",
        'output_path': "evaluation_results_fusion.json",
        'device': 'cuda:0',
        'max_seq_length': 7000,
        'batch_size': 1,
        'skip_samples': 24,  # Number of samples to skip from beginning
        'max_samples': None,  # Maximum samples to evaluate (None for all)
        'verbose': True  # Whether to print detailed outputs
    }
    
    print("🚀 Starting Conversation Summarization Model Evaluation")
    print("=" * 60)
    
    # Validate paths
    if not os.path.exists(CONFIG['checkpoint_path']):
        raise FileNotFoundError(f"Checkpoint not found: {CONFIG['checkpoint_path']}")
    
    if not os.path.exists(CONFIG['test_data_path']):
        raise FileNotFoundError(f"Test data not found: {CONFIG['test_data_path']}")
    
    # Initialize evaluator
    evaluator = ConversationEvaluator(
        checkpoint_path=CONFIG['checkpoint_path'],
        tokenizer_path=CONFIG['tokenizer_path'],
        device=CONFIG['device'],
        max_seq_length=CONFIG['max_seq_length'],
        batch_size=CONFIG['batch_size']
    )
    
    # Run evaluation
    try:
        results = evaluator.evaluate(
            test_data_path=CONFIG['test_data_path'],
            output_path=CONFIG['output_path'],
            skip_samples=CONFIG['skip_samples'],
            max_samples=CONFIG['max_samples'],
            verbose=CONFIG['verbose']
        )
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()