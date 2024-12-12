# coding:utf-8
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset,SequentialSampler
import torch.nn.functional as F
from bert_finetune import BertForQA, InputFeatures,compute_exact_match,compute_f1_score
import pickle
from tqdm import tqdm
import numpy as np
from collections import Counter
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, AutoModel,AutoTokenizer
from transformers import DistilBertForQuestionAnswering
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_file = "student_file1.txt"

def set_seed(seed=51):
    np.random.seed(seed)
    random.seed(seed)

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
# FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
# LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
device = torch.device('cuda' if USE_CUDA else 'cpu')

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=2):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_start = nn.Linear(hidden_dim * 2, 1)
        self.fc_end = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        start_logits = self.fc_start(lstm_out).squeeze(-1)  # [batch_size, seq_len]
        end_logits = self.fc_end(lstm_out).squeeze(-1)      # [batch_size, seq_len]
        return start_logits, end_logits

class RNNDuo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=2, use_pretrained_bert=False):
        super(RNNDuo, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_pretrained_bert = use_pretrained_bert

        if use_pretrained_bert:
            # Use BERT as the embedding layer
            self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
            self.embedding_dim = self.bert.config.hidden_size  # Typically 768 for BERT
        else:
            # Use standard embedding layer
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.embedding_dim = embedding_dim

        # Bidirectional LSTM
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Fully connected layer for start and end logits
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, attention_mask=None):
        if self.use_pretrained_bert:
            # Use BERT embeddings
            with torch.no_grad():
                bert_output = self.bert(input_ids=x, attention_mask=attention_mask)
                embedded = bert_output.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
        else:
            # Use standard embeddings
            embedded = self.dropout(self.embedding(x))

        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)

        # Project LSTM outputs to logits
        logits = self.fc(lstm_out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # Shape: [batch_size, seq_len]
        end_logits = end_logits.squeeze(-1)  # Shape: [batch_size, seq_len]

        return start_logits, end_logits

class Teacher(object):
    def __init__(self, bert_model='distilbert-base-uncased', max_seq=128, model_dir=None):
        self.max_seq = max_seq
        self.model = BertForQA.from_pretrained('distilbert-base-uncased')
        self.model.load_state_dict(torch.load("data/model/FTQASubset.pth", weights_only=True))
        self.model.eval()

    def predict(self, input_ids, input_mask, start_positions=None, end_position=None):
        with torch.no_grad():
            start, end = self.model(input_ids, input_mask)
            # print(" start ", start.shape)
            # print(" end ", end.shape)
            start_logits = F.softmax(start, dim=-1)
            end_logits = F.softmax(end, dim=-1)

        return start_logits, end_logits

def calculate_vocab_size(train_features):
    # Collect all unique token indices
    vocab_counter = Counter()
    for feature in train_features:  # Assuming train_features contains input_ids
        vocab_counter.update(feature.input_ids)
    return max(vocab_counter) + 1

def evaluateStudent(model, eval_dataloader, device):
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

        # print("input_ids: ", input_ids.shape)
        # Get model predictions
        with torch.no_grad():
            start_logits, end_logits = model(input_ids)


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
    print(f"Total Examples Evaluated: {total_data}")
    print(f"Total Exact Matches: {sum(exact_matches)}")
    print(f"Total F1 Scores: {sum(f1_scores)}")

    avg_em = (sum(exact_matches) / total_data) * 100 if total_data > 0 else 0.0
    avg_f1 = (sum(f1_scores) / total_data) * 100 if total_data > 0 else 0.0

    print(f"Exact Match (EM): {avg_em:.2f}%")
    print(f"F1 Score: {avg_f1:.2f}%")

    return {"Exact Match": avg_em, "F1 Score": avg_f1}

def train_student(teacher_model_path="data/model/FTQASubset.pth",
                  student_model_path="data/model/studentCoba.pth",
                  max_len=512,
                  batch_size=64,
                  lr=0.001,
                  epochs=20,
                  alpha=0.5):

    teacher = Teacher(max_seq=max_len,model_dir=teacher_model_path)
    teacher.model.to(device)

    with open("trainFeatureSubset.pkl", "rb") as file:
        train_features = pickle.load(file)

    train_features = train_features[0:10]
    print("len(train_features")
    # Flatten all input_ids to find the maximum index
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Get the vocab siz
    vocab_size = tokenizer.vocab_size
    print(" vocab size 1", vocab_size)

    # Load student model
    student = RNN(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256)
    simple = RNN(vocab_size=vocab_size, embedding_dim=128, hidden_dim=512)
    secondModel = RNNDuo(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256)

    student.to(device)
    simple.to(device)
    secondModel.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    optimizerSim = torch.optim.Adam(simple.parameters(), lr=lr)

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


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor(padded_label_ids, dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)

    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    student.train()
    start = time.time()
    temperature =7
    probabilities = {'teacher_start': [], 'teacher_end': [], 'student_start': [], 'student_end': [],
                     'distill_start': [], 'distill_end': []}

    with open(eval_file, "w") as file:
        file.write(f"Distilled\n")
    for epoch in range(epochs):
        epoch_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            start_position, end_position = label_ids[:, 0, 0], label_ids[:, 1, 0]


            # Get soft labels from the teacher
            with torch.no_grad():
                teacher_start_logits, teacher_end_logits = teacher.predict(input_ids=input_ids, input_mask=input_mask)
                teacher_start_logits = F.softmax(teacher_start_logits / temperature, dim=-1)
                teacher_end_logits = F.softmax(teacher_end_logits / temperature, dim=-1)

            probabilities['teacher_start'].append(teacher_start_logits.cpu().detach().numpy())
            probabilities['teacher_end'].append(teacher_end_logits.cpu().detach().numpy())

            student_start_logits, student_end_logits = student(input_ids)
            student_start_logits = F.log_softmax(student_start_logits / temperature, dim=-1)
            student_end_logits = F.log_softmax(student_end_logits / temperature, dim=-1)

            probabilities['student_start'].append(student_start_logits.cpu().detach().numpy())
            probabilities['student_end'].append(student_end_logits.cpu().detach().numpy())

            distillation_loss_start = nn.KLDivLoss(reduction='batchmean')(
                student_start_logits, teacher_start_logits
            ) * (temperature ** 2)

            distillation_loss_end = nn.KLDivLoss(reduction='batchmean')(
                student_end_logits, teacher_end_logits
            ) * (temperature ** 2)

            # Hard label loss
            hard_label_loss_start = nn.CrossEntropyLoss()(student_start_logits * temperature, start_position)
            hard_label_loss_end = nn.CrossEntropyLoss()(student_end_logits * temperature, end_position)

            # Total loss
            total_loss = alpha * (distillation_loss_start + distillation_loss_end) + (1 - alpha) * (hard_label_loss_start + hard_label_loss_end)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        with open(eval_file, "a") as file:
            file.write(f"TRAIN Loss: {epoch_loss:.4f}\n")
    with open('probabilities.pkl', 'wb') as f:
        pickle.dump(probabilities, f)
        end = time.time() - start
    with open(eval_file, "a") as file:
        file.write(f"Total Time  : {end}\n")

    simple.train()
    startSim = time.time()
    with open(eval_file, "a") as file:
        file.write(f"Simple\n")

    for epoch in range(epochs):
        epoch_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            start_position, end_position = label_ids[:, 0, 0], label_ids[:, 1, 0]

            # print("input_ids",input_ids )
            student_start_logits, student_end_logits = simple(input_ids)

            hard_label_loss_start = nn.CrossEntropyLoss()(student_start_logits, start_position)
            hard_label_loss_end = nn.CrossEntropyLoss()(student_end_logits, end_position)

            loss_sim = (hard_label_loss_start + hard_label_loss_end) / 2
            # Compute loss
            optimizerSim.zero_grad()
            loss_sim.backward()
            optimizerSim.step()

            epoch_loss += loss_sim.item()

        with open(eval_file, "a") as file:
            file.write(f"TRAIN LOSS: {epoch_loss:.4f}\n")

    endSim = time.time() - startSim
    with open(eval_file, "a") as file:
        file.write(f"Total Time Simple Model: {endSim}\n")

    print('evaluation ')

    with open("evalFeatureSubset.pkl", "rb") as file:
        eval_features = pickle.load(file)

    eval_features = random.sample(list(eval_features), int(0.0005 * len(eval_features)))
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

    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor(padded_label_ids, dtype=torch.long)
    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_label_ids)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size)

    with open(eval_file, "a") as file:
        file.write(f"Student Simple\n")
    evaluateStudent(student, eval_dataloader, device)
    with open(eval_file, "a") as file:
        file.write(f"Eval Simple\n")
    evaluateStudent(simple, eval_dataloader, device)
        # Save the student model
    # torch.save(student.state_dict(), student_model_path)

if __name__ == "__main__":
    train_student() 

