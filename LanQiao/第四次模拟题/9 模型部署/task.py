import torch

index2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}
model = torch.jit.load('ner.pt')
model.eval()
outputs = model(torch.tensor([[100, 1003, 1009, 106, 102], [2000, 1003, 1009, 1030, 1090]])).detach().numpy()
output_labels = []
for output in outputs:
    output_labels.append([index2label[o] for o in output])
print(outputs, output_labels)
# [[0 3 4 1 2] [0 3 4 0 0]]
# [['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER'], ['O', 'B-LOC', 'I-LOC', 'O', 'O']]
