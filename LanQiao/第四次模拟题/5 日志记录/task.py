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
    
    with torch.no_grad():
        for text, label in iterator:
            # print(label)
            outputs = model(text.transpose(0, 1))
            loss = criterion(outputs, label)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            # print(torch.eq(predicted,label.squeeze(dim=-1)))
            # print(torch.eq(predicted,label.squeeze(dim=-1)).float())
            # print(torch.eq(predicted,label.squeeze(dim=-1)).float().mean())
            accuracy = float(torch.eq(predicted,label).float().mean())
            TP = torch.logical_and(predicted ==1,label==1).sum().item()
            FP = torch.logical_and(predicted ==1,label==0).sum().item()
            TN = torch.logical_and(predicted ==0,label==0).sum().item()
            FN = torch.logical_and(predicted ==0,label==1).sum().item()

            precision = TP/(TP+TN)
            recall = TP/(TP+FN)
            f1 = 2 * precision * recall / (precision + recall)

    return total_loss / len(iterator), accuracy, precision, recall, f1

def run():
    model = TextClassifier()
    train_loader, val_loader = get_data_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    NUM_EPOCHS = 10
    for epoch in trange(NUM_EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion)
        log_file_path="training.log"
        with open(log_file_path, "a") as log_file:
            for i in range(3):
                if i == 0:
                    '''print("Epoch: "+str(epoch+1))'''
                    log_file.write("Epoch: "+str(epoch+1)+'\n')
                if i == 1 :
                    '''print("Train Loss: "+str("{:.3f}".format(train_loss)))'''
                    log_file.write("Train Loss: "+str("{:.3f}".format(train_loss))+'\n')
                if i == 2 :
                    '''print("Val Loss: " + str("{:.3f}".format(val_loss))\
                        +" | Val Accuracy: " + str("{:.2f}".format(100*val_accuracy))\
                        +"% | Precision: " + str("{:.3f}".format(precision))\
                        +" | Recall: " + str("{:.3f}".format(recall))\
                        +" | F1: " + str("{:.3f}".format(f1)))'''
                    log_file.write("Val Loss: " + str("{:.3f}".format(val_loss))\
                                +" | Val Accuracy: " + str("{:.3f}".format(100*val_accuracy))\
                                +"% | Precision: " + str("{:.3f}".format(precision))\
                                +" | Recall: " + str("{:.3f}".format(recall))\
                                +" | F1: " + str("{:.3f}".format(f1))+'\n')

    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    run()
#task-end