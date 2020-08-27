# Fine-tuning a BERT model for text extraction with the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/)

We are going to fine-tune BERT (from [HuggingFace Transformers](https://github.com/huggingface/transformers)) for the text-extraction task with a dataset of questions and answers. The questions are about a given paragraph (*context*) that contains the answers. The model will be trained to locate the answer in the context by giving the possitions where the answer starts and finishes.

This example is based on [BERT (from HuggingFace Transformers) for Text Extraction](https://keras.io/examples/nlp/text_extraction_with_bert/).

Before running the python scripts, it's necessary do the following exports:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1
```
