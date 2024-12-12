# -*- coding: utf-8 -*-
import os, csv, random, torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from transformers import BertModel, BertPreTrainedModel
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast
from transformers import AutoTokenizer
from transformers import AdamW
from transformers import DistilBertForQuestionAnswering
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm, trange
import time
import pickle
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_file = "eval_file.txt"

class InputExample(object):
    def __init__(self, guid, question, context, answers=None):
        self.guid = guid
        self.question = question
        self.context = context
        self.answers = answers


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


class Processor(object):
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'train.txt'), 'train')

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'test.txt'), 'test')

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'), 'dev')

    def _create_examples(self, data_path, set_type):
        examples = []
        with open(data_path, encoding="utf-8") as f:
            entry = {}
            for line in f:
                line = line.strip()
                if not line:  # Empty line marks the end of an entry
                    if entry:
                        # Create InputExample from the entry
                        examples.append(InputExample(
                            guid=f"{set_type}-{entry['id']}",
                            question=entry['question'],
                            context=entry['context'],
                            answers=[ans.strip() for ans in entry['answers'].split(";")]
                        ))
                        entry = {}  # Reset entry for the next block
                elif line.startswith("ID:"):
                    entry['id'] = line[4:]
                elif line.startswith("Title:"):
                    entry['title'] = line[7:]  # Optional, not used but parsed
                elif line.startswith("Question:"):
                    entry['question'] = line[10:]
                elif line.startswith("Context:"):
                    entry['context'] = line[9:]
                elif line.startswith("Answers:"):
                    entry['answers'] = line[8:]

            # Add the last entry if file does not end with a blank line
            if entry:
                examples.append(InputExample(
                    guid=f"{set_type}-{entry['id']}",
                    question=entry['question'],
                    context=entry['context'],
                    answers=[ans.strip() for ans in entry['answers'].split(";")]
                ))
        return examples


def find_context_start_end_index(sequence_ids):
    token_idx = 0
    # Skip special tokens and question tokens
    while sequence_ids[token_idx] != 1:
        token_idx += 1
    context_start_idx = token_idx

    # Iterate until the context ends
    while token_idx < len(sequence_ids) and sequence_ids[token_idx] == 1:
        token_idx += 1
    context_end_idx = token_idx - 1

    return context_start_idx, context_end_idx

def find_token_start_idx(answer_start_char_idx, context_start_idx, context_end_idx, offset_mapping):
    idx = context_start_idx
    while idx <= context_end_idx and offset_mapping[idx][0] <= answer_start_char_idx:
        idx += 1
    return idx - 1

def find_token_end_idx(answer_end_char_idx, context_start_idx, context_end_idx, offset_mapping):
    idx = context_end_idx
    while idx >= context_start_idx and offset_mapping[idx][1] > answer_end_char_idx:
        idx -= 1
    return idx + 1


def convert_examples_to_features(examples, tokenizer, max_seq):
    features = []

    for ex_index, example in enumerate(examples):
        # Tokenize question and context
        inputs = tokenizer(
            example.question,
            example.context,
            truncation="only_second",
            padding="max_length",
            max_length=max_seq,
            stride=32,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )

        start_positions = []
        end_positions = []
        # print(inputs.keys())

        selected_answer = example.answers[0] if example.answers else None
        # print('=='*25)
        # print(f"Selected Answer: {selected_answer}")

        for i, mapping_idx_pairs in enumerate(inputs['offset_mapping']):
            context_idx = inputs['overflow_to_sample_mapping'][i]
            # print(f"Input IDs Chunk {i}: {inputs['input_ids'][i]}")

            # print(f"Offset Mapping: {inputs['offset_mapping']}")
            if selected_answer:
                answer_start_char_idx = example.context.find(selected_answer)
                # print(f"Answer Start Char Index: {answer_start_char_idx}")
                if answer_start_char_idx == -1:
                    continue
                answer_end_char_idx = answer_start_char_idx + len(selected_answer)
                # print(f"Answer End Char Index: {answer_end_char_idx}")

                sequence_ids = inputs.sequence_ids(i)
                # print(f"sequence_ids: {sequence_ids}")
                context_start_idx, context_end_idx = find_context_start_end_index(sequence_ids)
                # print(f"context_start_idx: {context_start_idx}")
                # print(f"context_end_idx: {context_end_idx}")

                context_start_char_index = mapping_idx_pairs[context_start_idx][0]
                # print(f"context_start_char_index: {context_start_char_index}")

                context_end_char_index = mapping_idx_pairs[context_end_idx][1]
                # print(f"context_end_char_index: {context_end_char_index}")

                if (context_start_char_index > answer_start_char_idx) or (context_end_char_index < answer_end_char_idx):
                    start_positions.append(0)
                    end_positions.append(0)
                    continue
                else:
                    idx = context_start_idx
                    while idx <= context_end_idx and mapping_idx_pairs[idx][0] <= answer_start_char_idx:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end_idx
                    while idx >= context_start_idx and mapping_idx_pairs[idx][1] > answer_end_char_idx:
                        idx -= 1
                    end_positions.append(idx + 1)
                # print("start_positions: ", start_positions)
                # print("end_positions: ", end_positions)
            features.append(InputFeatures(
                input_ids=inputs["input_ids"][i],
                input_mask=inputs["attention_mask"][i],
                label_id=(start_positions, end_positions)
            ))

    return features

class BertForQA(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQA, self).__init__(config)
        self.bert = DistilBertForQuestionAnswering(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()



    def forward(self, input_ids, input_mask, start_position=None, end_position=None):
        # Get the output from the pretrained QA model
        outputs = self.bert(input_ids=input_ids, attention_mask=input_mask,
                                  start_positions=start_position, end_positions=end_position)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Return the loss during training or logits during evaluation
        if start_position is not None and end_position is not None:
            # Loss is already computed by DistilBertForQuestionAnswering
            return outputs.loss

        return start_logits, end_logits

def compute_exact_match(start_pred, end_pred, start_true, end_true):
    return int(start_pred == start_true and end_pred == end_true)


def compute_f1_score(start_pred, end_pred, start_true, end_true):
    # Convert spans to sets of tokens.
    with open(eval_file, "a") as file:
        file.write("\nCompute F1 ")
        file.write(f"\nstart_pred : {start_pred}\n")
        file.write(f"end_pred : {end_pred}\n")
    pred_tokens = set(range(start_pred, end_pred + 1))
    with open(eval_file, "a") as file:
        file.write(f"start_true : {start_true}\n")
        file.write(f"end_true : {end_true}\n")
    true_tokens = set(range(start_true, end_true + 1))
    with open(eval_file, "a") as file:
        file.write(f"true_tokens : {true_tokens}\n")

    if len(pred_tokens) == 0 or len(true_tokens) == 0:  # Handle empty predictions
        return int(pred_tokens == true_tokens)

    # Compute F1 score
    overlap = pred_tokens.intersection(true_tokens)
    with open(eval_file, "a") as file:
        file.write(f"overlap : {overlap}\n")
        file.write(f"pred_tokens : {pred_tokens}\n")

    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(true_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)



def evaluate(model, eval_dataloader, device):
    model.eval()
    exact_matches = []
    f1_scores = []
    total_data =0
    avg_em = 0
    avg_f1 =0
    with open(eval_file, "a") as file:
        file.write("Evaluation Results\n")
        file.write("=" * 40 + "\n")

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
        start_positions, end_positions = label_ids[:, 0,0], label_ids[:, 1,0]

        # Get model predictions
        with torch.no_grad():
            start_logits, end_logits = model(input_ids=input_ids, input_mask=input_mask)


        start_preds = torch.argmax(start_logits, dim=-1)
        end_preds = torch.argmax(end_logits, dim=-1)

        for i in range(len(start_preds)):
            pred_start = start_preds[i].item()
            pred_end = end_preds[i].item()
            gold_start = start_positions[i].item()
            gold_end = end_positions[i].item()

            # Skip examples with invalid gold labels
            if gold_start == 0 and gold_end == 0:
                continue

            # Write detailed results to the evaluation file
            with open(eval_file, "a") as file:
                file.write("-" * 20 + "\n")
                file.write(f"Example {total_data + 1}\n")
                file.write(f"Predicted Start: {pred_start}\n")
                file.write(f"Gold Start: {gold_start}\n")
                file.write(f"Predicted End: {pred_end}\n")
                file.write(f"Gold End: {gold_end}\n")

            # Compute Exact Match and F1 Score
            em = compute_exact_match(pred_start, pred_end, gold_start, gold_end)
            f1 = compute_f1_score(pred_start, pred_end, gold_start, gold_end)

            exact_matches.append(em)
            f1_scores.append(f1)
            total_data += 1

        # Print and return aggregated results
    with open(eval_file, "a") as file:
        file.write(f"Total Examples Evaluated {total_data + 1}\n")
        file.write(f"Total Exact Matches: {sum(exact_matches)}\n")
        file.write(f"Total F1 Scores: {sum(f1_scores)}\n")

    avg_em = (sum(exact_matches) / total_data) * 100 if total_data > 0 else 0.0
    avg_f1 = (sum(f1_scores) / total_data) * 100 if total_data > 0 else 0.0

    with open(eval_file, "a") as file:
        file.write(f"Exact Match (EM): {avg_em:.2f}%")
        file.write(f"F1 Score: {avg_f1:.2f}%")

    return {"Exact Match": avg_em, "F1 Score": avg_f1}


def remove_zeros_preserve_dims(tensor):
    max_len = 0
    result = []
    for matrix in tensor:  # Loop over first dimension
        rows = []
        for row in matrix:  # Loop over second dimension
            non_zero_row = row[row != 0]  # Remove zeros from the row
            rows.append(non_zero_row)
            max_len = max(max_len, len(non_zero_row))  # Track maximum row length
        result.append(rows)

    # Pad rows to make them consistent in the third dimension
    padded_tensor = torch.zeros((len(result), len(result[0]), max_len))
    for i, matrix in enumerate(result):
        for j, row in enumerate(matrix):
            padded_tensor[i, j, :len(row)] = row
    return padded_tensor

def main(data_dir=None,
         model_dir=None,
         bert_model='distilbert-base-uncased',
         cache_dir=None,
         max_seq=128,
         batch_size=64,
         num_epochs=20,
         lr=2e-5):
    processor = Processor()
    train_examples = processor.get_train_examples(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(bert_model, do_lower_case=True, use_fast=True)
    model = BertForQA.from_pretrained(bert_model, cache_dir=cache_dir)

    model.to(device)

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.bias', 'LayerNorm.weight'])], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.bias', 'LayerNorm.weight'])], 'weight_decay': 0.01}
    ]
    print('train...')
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    train_features = convert_examples_to_features(train_examples,tokenizer, max_seq)

    # Find max number of answers across all features
    max_answers = max(len(f.label_id[0]) for f in train_features)

    # Pad label_id to have uniform dimensions
    padded_label_ids = []
    for f in train_features:
        start_positions, end_positions = f.label_id

        # Pad with zeros for missing answers
        start_positions += [0] * (max_answers - len(start_positions))
        end_positions += [0] * (max_answers - len(end_positions))

        # Append normalized labels
        padded_label_ids.append([start_positions, end_positions])

    # Convert to PyTorch tensors
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor(padded_label_ids, dtype=torch.long)

    remove_0_label = remove_zeros_preserve_dims(all_label_ids)
    all_label_ids = torch.tensor(remove_0_label, dtype=torch.long)

    file_path = "label_ids.txt"
    with open(file_path, "w") as file:
        file.write("=="*22)
        file.write(f"\nInput len {len(train_features)}\n")
    with open(file_path, "a") as file:
        for label in all_label_ids:
            file.write(f"{label}\n")
        for input in all_input_ids:
            file.write(f"{input}\n")
    print(f"Label IDs have been saved to {file_path}")

    train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    # train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    print("Train dataloader ", train_dataloader)
    model.train()
    with open(eval_file, "w") as file:
        file.write(f"TRAIN LOSS :\n")
    start = time.time()
    for _ in trange(num_epochs, desc='Epoch'):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)

            start_position, end_position = label_ids[:, 0, 0], label_ids[:,1, 0]

            optimizer.zero_grad()
            loss = model(input_ids, input_mask, start_position, end_position)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tr_loss += loss.item()

        with open(eval_file, "a") as file:
            file.write(f"TRAIN LOSS : {tr_loss}\n")

    end = time.time() - start
    with open(eval_file, "a") as file:
        file.write(f"Total Time : {end}\n")
    print('eval...')

    eval_examples = processor.get_train_examples(data_dir)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, max_seq)
 
    eval_features = random.sample(list(eval_features), int(0.0005 * len(eval_features)))
    with open("evalFeatureSubset.pkl", "wb") as file:
         pickle.dump(eval_features, file)
    # Find max number of answers across all features
    max_answers = max(len(f.label_id[0]) for f in eval_features)

    # Pad label_id to have uniform dimensions
    padded_label_ids = []
    for f in eval_features:
        start_positions, end_positions = f.label_id

        # Pad with zeros for missing answers
        start_positions += [0] * (max_answers - len(start_positions))
        end_positions += [0] * (max_answers - len(end_positions))

        # Append normalized labels
        padded_label_ids.append([start_positions, end_positions])

    # Convert to PyTorch tensors
    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor(padded_label_ids, dtype=torch.long)
    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size)

    evaluate(model, eval_dataloader, device)

    torch.save(model.state_dict(), model_dir)

if __name__ == '__main__':
        main(data_dir="QAdata",
         model_dir="data/model/FTQASubset.pth")
