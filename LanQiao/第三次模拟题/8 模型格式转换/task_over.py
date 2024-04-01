#task-start
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=512, num_classes=2):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):

        embedded = self.embedding(text)
        packed_output, (hidden, cell) = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output


def convert():
    # TODO
    model = TextClassifier()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    dummy_input = torch.ones([256, 1], dtype=torch.long)
    torch.onnx.export(model, dummy_input, "text_classifier.onnx")


def inference(model_path, input):
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Pad or truncate the input sequence to a length of 256
    input = (input + [0]*256)[:256]

    input = np.array(input, dtype=np.int64).reshape(256, 1)
    result = sess.run([output_name], {input_name: input})

    return result[0].tolist()


def main():
    convert()
    result = inference('/home/project/text_classifier.onnx', [101, 304, 993, 108,102])
    print(result)


if __name__ == '__main__':
    main()
#task-end