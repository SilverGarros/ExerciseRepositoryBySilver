#task-start
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange
import warnings
import logging
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

class TextClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, nhead=4, num_encoder_layers=2, num_classes=2):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text):

        embedded = self.embedding(text)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[0])
        return output

def get_data_loaders():
    data, labels = pickle.load(open('text_classify_training_data.pkl', 'rb'))

    dataset = TensorDataset(data, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader

def train(model, iterator, criterion, optimizer):
    model.train()
    total_loss = 0

    for text, label in iterator:
        optimizer.zero_grad()
        outputs = model(text.transpose(0, 1))
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    total_predicted_positive = 0
    total_actual_positive = 0
    total_true_positive = 0   
    with torch.no_grad():
        for text, label in iterator:
            outputs = model(text.transpose(0, 1))
            loss = criterion(outputs, label)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
###
            total_correct += (predicted == label).sum().item()
            total_examples += label.size(0)
            total_predicted_positive += predicted.sum().item()
            total_actual_positive += label.sum().item()
            total_true_positive += (predicted * label).sum().item()

    accuracy = total_correct / total_examples
    precision = total_true_positive / total_predicted_positive
    recall = total_true_positive / total_actual_positive
    f1 = 2 * precision * recall / (precision + recall)
###
    return total_loss / len(iterator), accuracy, precision, recall, f1

logging.basicConfig(filename='training.log', filemode='a',level=logging.INFO, format='%(message)s')
def run():
    model = TextClassifier()
    train_loader, val_loader = get_data_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    NUM_EPOCHS = 10
    for epoch in trange(NUM_EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion)
# 按照如下格式记录
# Epoch: 1
# Train Loss: 1.507
# Val Loss: 0.814 | Val Accuracy: 50.00% | Precision: 0.438 | Recall: 0.875 | F1: 0.583

        logging.info(f'Epoch: {epoch+1}')
        logging.info(f'Train Loss: {train_loss:.3f}')
        logging.info(f'Val Loss: {val_loss:.3f} | Val Accuracy: {val_accuracy*100:.2f}% | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}')

    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    run()
#task-end