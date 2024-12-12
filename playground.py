import torch
import torch.nn as nn 
from bert_finetune import BertForQA
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import random
import pickle
from bert_finetune import InputFeatures

def load_data():
    # Load SQuAD dataset
    dataset = load_dataset("squad")

    # Split into train and validation sets
    train_data = dataset['train']
    val_data = dataset['validation']

    return train_data, val_data

def get_teacher_logits(context, question):
    inputs = tokenizer(context, question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = teacher_model(**inputs)
    return outputs.start_logits, outputs.end_logits


class RNNForQA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNForQA, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_start = nn.Linear(hidden_dim, output_dim)
        self.fc_end = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        start_logits = self.fc_start(rnn_out)
        end_logits = self.fc_end(rnn_out)
        return start_logits, end_logits


def preprocess_data(example):
    context = example["context"]
    question = example["question"]
    answers = example["answers"]

    # Get teacher logits
    start_logits, end_logits = get_teacher_logits(context, question)

    answer_start = answers["answer_start"][0]
    answer_text = answers["text"][0]
    answer_end = answer_start + len(answer_text)
    # Tokenize inputs
    inputs = tokenizer(context, question, truncation=True, padding=True, return_tensors="pt")
    start_positions = torch.tensor(answers["answer_start"])
    end_positions = torch.tensor([answer_end])

    return inputs, start_logits, end_logits, start_positions, end_positions

def encode_qa_sample(question, context, answer, tokenizer):
    """
    Encodes a QA sample (question, context, and answer) using the tokenizer.

    Args:
        question (str): The question string.
        context (str): The context string containing the answer.
        answer (str): The answer string.
        tokenizer (PreTrainedTokenizer): The tokenizer to encode inputs.

    Returns:
        dict: A dictionary containing tokenized inputs and start/end positions.
    """
    # Tokenize context and question
    inputs = tokenizer(
        context,
        question,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    # Find the answer's start and end positions in the context
    answer_start = context.find(answer)
    answer_end = answer_start + len(answer) - 1

    if answer_start == -1:
        raise ValueError("Answer not found in the context!")

    # Convert character positions to token positions
    token_start = tokenizer.encode(context[:answer_start], add_special_tokens=False)
    token_end = tokenizer.encode(context[:answer_end + 1], add_special_tokens=False)

    # Add start and end positions to the inputs
    inputs["start_positions"] = torch.tensor([len(token_start)])
    inputs["end_positions"] = torch.tensor([len(token_start) + len(token_end) - 1])

    return inputs

if __name__ == '__main__':

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    question = "How many people work in Switzerland?"
    context = (
        "Around 3.8 million people work in Switzerland; about 25% of employees belonged to "
        "a trade union in 2004. Switzerland has a more flexible job market than neighbouring "
        "countries and the unemployment rate is very low. The unemployment rate increased "
        "from a low of 1.7% in June 2000 to a peak of 4.4% in December 2009. The unemployment "
        "rate is 3.2% in 2014. Population growth from net immigration is quite high, at 0.52% "
        "of population in 2004. The foreign citizen population was 21.8% in 2004, about the "
        "same as in Australia. GDP per hour worked is the world's 16th highest, at 49.46 "
        "international dollars in 2012."
    )
    answer = "Around 3.8 million"

    encoded_sample = encode_qa_sample(question, context, answer, tokenizer)

    print("Input IDs:", encoded_sample["input_ids"])
    print("Attention Mask:", encoded_sample["attention_mask"])
    print("Start Position:", encoded_sample["start_positions"])
    print("End Position:", encoded_sample["end_positions"])