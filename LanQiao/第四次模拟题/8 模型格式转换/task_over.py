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

    dummy_input = torch.ones([256, 1],dtype=torch.long)
    torch.onnx.export(model,dummy_input,'/home/project/text_classifier.onnx')


def inference(model_path, input):
    # TODO
    ort_session = ort.InferenceSession(model_path)
    input_data = np.array(input,dtype=np.int64)
    # ！！！将输入填充为256位的数据！！
    if len(input_data) < 256:
        input_data = np.pad(input_data, (0, 256 - len(input_data)), 'constant', constant_values=0).reshape(256,1)
    print(input_data.shape)

    ort_inputs={ort_session.get_inputs()[0].name:input_data}
    ort_outs = ort_session.run(None,ort_inputs)
    result = ort_outs[0]
    return result


def main():
    convert()
    result = inference('/home/project/text_classifier.onnx', [101, 304, 993, 108,102])
    print(result)


if __name__ == '__main__':
    main()