from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from fast_fine_tuna.dataset import MainDataset
from transformers import AdamW
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

class FastFineTuna:

    def __init__(self, model_name, tokenizer_name):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name


    def cross_validate_fit(self, texts, labels, epochs=5):

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        texts = np.array(texts)
        labels = np.array(labels)

        skf = StratifiedKFold(n_splits=2)

        original = []
        predicted = []

        for train_index, test_index in skf.split(texts, labels):
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)


            X_train, X_test = texts[train_index].tolist(), texts[test_index].tolist()
            y_train, y_test = labels[train_index].tolist(), labels[test_index].tolist()

            tokenized_train = tokenizer(X_train, truncation=True, padding=True)
            tokenized_test = tokenizer(X_test, truncation=True, padding=True)

            train_dataset = MainDataset(tokenized_train, y_train)
            test_dataset = MainDataset(tokenized_test, y_test)

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            model.to(device)
            model.train()


            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

            optim = AdamW(model.parameters(), lr=5e-5)

            pbar = tqdm(total=epochs, position=0, leave=True)
            for epoch in range(epochs):
                pbar.update(1)
                for batch in train_loader:
                    optim.zero_grad()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    lab = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=lab)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
            pbar.close()
            model.eval()

            loader = DataLoader(test_dataset, batch_size=16)
            original.extend(y_test)
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    predicted.extend(torch.argmax(outputs["logits"], axis=1).cpu().numpy().tolist())
            del model
        return original, predicted


    def train_and_save(self, texts, labels, path, epochs=5):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        labels = np.array(labels)

        tokenized_train = tokenizer(texts, truncation=True, padding=True)

        train_dataset = MainDataset(tokenized_train, labels)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model.to(device)
        model.train()

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        optim = AdamW(model.parameters(), lr=5e-5)

        for epoch in range(epochs):
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                lab = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=lab)
                loss = outputs[0]
                loss.backward()
                optim.step()

        os.makedirs(path)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)



