# Fine-Tuned-BERT-for-Movie-Review-Classification-with-PEFT-and-LoRA

This project involves fine-tuning a BERT-based model for sentiment classification of movie reviews using the IMDB dataset. The model is trained using advanced techniques like Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA).

## Introduction

In this project, we fine-tune a BERT model for the task of binary sentiment classification. The IMDB dataset is used, which consists of movie reviews labeled as positive or negative. We utilize advanced techniques to efficiently adapt the model for this specific task.

Novel Techniques
----------------

### Parameter-Efficient Fine-Tuning (PEFT)

PEFT techniques aim to reduce the number of parameters that need to be fine-tuned by efficiently adapting only a subset of parameters. This approach is advantageous in scenarios with limited computational resources or when dealing with very large models.

### Low-Rank Adaptation (LoRA)

LoRA is a PEFT method that introduces trainable low-rank matrices into the model. These matrices are added to the original weights of the model and are trained during the fine-tuning process. This method helps in efficiently adapting the model without the need to retrain all the weights from scratch.

**LoRA Configuration:**

*   **r (rank):** 4
    
*   **lora\_alpha:** 32
    
*   **lora\_dropout:** 0.01
    
*   **target\_modules:** \['q\_lin'\]
    

The combination of PEFT and LoRA allows for effective fine-tuning of the BERT model with fewer resources and faster convergence.

Model Training
--------------

1.  **Data Preparation:**
    
    *   The IMDB dataset is loaded and a random subset is sampled for training and validation.
        
    *   The dataset is tokenized using the BERT tokenizer.
        
2.  **Model Initialization:**
    
    *   We use the distilbert-base-uncased model for sequence classification.
        
    *   The model's labels are mapped to "Positive" and "Negative".
        
3.  **Training Configuration:**
    
    *   **Learning Rate:** 1e-3
        
    *   **Batch Size:** 4
        
    *   **Number of Epochs:** 1
        
    *   **Evaluation Strategy:** Evaluate at the end of each epoch
        
    *   **Save Strategy:** Save model at the end of each epoch
        
4.  **Training:**
    
    *   The model is fine-tuned using the Trainer API from the transformers library.

## Setup

To run this project, you need to install the following Python packages:

```bash
pip install datasets evaluate transformers[sentencepiece]
pip install accelerate -U
pip install peft
```
