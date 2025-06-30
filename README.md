# Discourse-Guided Summarisation of Psychotherapy Dialogues via Graph-Fused Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of our paper **"Discourse-Guided Summarisation of Psychotherapy Dialogues via Graph-Fused Language Models"**. Our approach combines the power of Large Language Models (LLaMA) with Graph Neural Networks to capture discourse structure and relationships in psychotherapy conversations for improved summarization.

## 🎯 Overview

Psychotherapy dialogue summarization requires understanding complex discourse patterns, therapeutic relationships, and contextual dependencies that traditional sequence-to-sequence models often miss. Our solution introduces:

- **Graph-Fused Language Models**: Integration of Graph Neural Networks with LLaMA to capture discourse structure
- **Intermediate Fusion Architecture**: Cross-attention mechanisms at multiple transformer layers
- **Discourse-Aware Processing**: Explicit modeling of utterance relationships and therapeutic dialogue patterns

## 🏗️ Architecture

Our model consists of three main components:

1. **Conversation Graph Model**: RGCN-based encoder that processes dialogue structure
2. **LLaMA with Intermediate Fusion**: Modified LLaMA with cross-attention fusion layers
3. **Conversation Summarization Model**: End-to-end pipeline combining graph and language understanding

## 📋 Requirements

### Core Dependencies
```
torch>=2.0.0
torch-geometric>=2.3.0
transformers>=4.30.0
```

### Additional Requirements
```
nltk>=3.8
rouge-score>=0.1.2
tqdm>=4.65.0
bitsandbytes>=0.39.0
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/minimilist/TheraFuse.git
cd TheraFuse
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torch-geometric transformers
pip install nltk rouge-score tqdm bitsandbytes
```

4. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
```

## 📁 Project Structure

```
discourse-guided-summarization/
├── llama_fusion.py              # LLaMA fusion implementation
├── fusion_model.py              # Main model architecture
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── graph_data.py               # Graph data processing (not included)
├── data_utils.py               # Data utilities (not included)
├── data/                       # Dataset directory
│   ├── aiims_train/           # Training data
│   └── aiims_test/            # Test data
├── fusion_models_aiims/       # Model checkpoints
└── README.md
```

## 🚀 Quick Start

### 1. Data Preparation

Prepare your psychotherapy dialogue data in the following format:
```python
{
    'prompt': str,                    # Input dialogue context
    'summary': str,                   # Target summary
    'utterance_embeddings': torch.Tensor,  # Node features
    'links_ids': List[List[int]],     # Edge indices
    'link_type': List[int]            # Edge types
}
```

### 2. Training

```bash
python train.py
```

**Key training parameters:**
- `model_path`: Path to base LLaMA model
- `data_path`: Path to training data
- `batch_size`: Training batch size (default: 1)
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate (default: 1e-5)
- `fusion_layers`: Layers for intermediate fusion (default: [27, 29, 31])

### 3. Evaluation

```bash
python evaluate.py
```

**Evaluation metrics:**
- BLEU-1, BLEU-2, BLEU-4
- ROUGE-1, ROUGE-2, ROUGE-L

## 📊 Model Configuration

### Default Configuration
```python
CONFIG = {
    'model_path': '/path/to/Meta-Llama-3.1-8B-Instruct',
    'fusion_layers': [27, 29, 31],  # Intermediate fusion layers
    'hidden_size': 4096,            # Model hidden size
    'num_relations': 17,            # Number of discourse relations
    'max_seq_length': 7000,         # Maximum sequence length
    'batch_size': 1,                # Batch size
    'learning_rate': 1e-5,          # Learning rate
    'gradient_accumulation_steps': 2 # Gradient accumulation
}
```

### Graph Model Parameters
- **Input channels**: LLaMA hidden size (4096)
- **Hidden channels**: 4096
- **Output channels**: 4096
- **Relations**: 17 discourse relation types
- **Layers**: 3-layer RGCN

## 💾 Model Checkpoints

Trained models are saved in the `fusion_models_aiims/` directory:
```
fusion_models_aiims/
├── conversation_summarization_model_epoch_2.pt
├── conversation_summarization_model_epoch_4.pt
└── conversation_summarization_model_epoch_6.pt
```

Each checkpoint contains:
- Model state dictionary
- Optimizer state
- Training epoch and loss

## 🔧 Usage Examples

### Loading a Trained Model
```python
from fusion_model import ConversationSummarizationModel

# Initialize model
model = ConversationSummarizationModel(
    model_id="/path/to/Meta-Llama-3.1-8B-Instruct"
)

# Load checkpoint
checkpoint = torch.load("fusion_models_aiims/model_epoch_6.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Generating Summaries
```python
# Prepare inputs
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    x=utterance_embeddings,
    edge_index=edge_index,
    edge_type=edge_type,
    node_to_graph=node_to_graph,
    model_max_length=7000
)

# Decode output
summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

## 📈 Performance

Our model demonstrates significant improvements over baseline approaches:

| Method | BLEU-1 | BLEU-2 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|--------|--------|--------|---------|---------|---------|
| Baseline LLaMA | - | - | - | - | - | - |
| Our Approach | - | - | - | - | - | - |

*Note: Add your specific performance numbers here*

## 🏥 Dataset

The model is trained and evaluated on psychotherapy dialogue datasets. The data structure includes:

- **Dialogue Context**: Multi-turn conversations between therapist and patient
- **Discourse Graph**: Utterance relationships and therapeutic interaction patterns
- **Target Summaries**: Clinical summaries capturing key therapeutic insights

## 🔬 Technical Details

### Fusion Architecture
- **Cross-Attention Fusion**: Multi-head attention between LLaMA hidden states and graph embeddings
- **Layer Normalization**: Robust normalization for stable training
- **Residual Connections**: Skip connections for gradient flow

### Graph Processing
- **RGCN Layers**: Relational Graph Convolutional Networks for discourse modeling
- **Edge Types**: 17 different discourse relation types
- **Node Features**: Utterance embeddings from pre-trained models

### Training Strategy
- **Parameter Freezing**: Base LLaMA parameters frozen, only fusion components trained
- **Gradient Accumulation**: Efficient training with limited memory
- **Learning Rate Scheduling**: Linear warmup with decay

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your_paper_2024,
  title={Discourse-Guided Summarisation of Psychotherapy Dialogues via Graph-Fused Language Models},
  author={Your Name and Co-authors},
  journal={Conference/Journal Name},
  year={2024}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Meta AI for the LLaMA model
- PyTorch Geometric team for graph neural network implementations
- Hugging Face for transformer implementations
- Contributors to the psychotherapy dialogue datasets

## 📞 Contact

For questions and support:
- **Email**: ankit201921@gmail.com
- **GitHub Issues**: [Open an issue](https://github.com/minimilist/TheraFuse/issues)

## 🔄 Version History

- **v1.0.0** (2024): Initial release with LLaMA-3.1 integration
- **v0.9.0** (2024): Beta release with core functionality

---

**Keywords**: Psychotherapy, Dialogue Summarization, Graph Neural Networks, Large Language Models, Discourse Analysis, Healthcare AI
