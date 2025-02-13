# Simple Experiment

This repository contains the implementation of the project for the Final6120 course. 
The project focuses on exploring **knowledge distillation** from BERT to RNN models for two key tasks: 
**Question Answering (QA)**. The goal is to analyze the effectiveness of transferring knowledge from large pre-trained models to lightweight architectures.

Code modified form : https://github.com/qiangsiwei/bert_distill
---

## Features
- **Question Answering Task**: Implements distillation from BERT to a biLSTM-based student model.
- Supports hyperparameter tuning, including temperature in distillation process

---
##Information of the file 
eval_file.txt: result of evaluation training in teacher model 
student_file.txt:  result of training loss of distilled model 

## Installation

### Prerequisites
- Python 3.7+

### Steps
1. Fine Tuning teacher model: python bert_finetune.py
2. Distilled the student: python bert_distill.py 
