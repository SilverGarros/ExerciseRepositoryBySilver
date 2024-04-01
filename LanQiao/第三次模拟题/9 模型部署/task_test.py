#task-start
from flask import Flask, jsonify, request
import torch

app = Flask(__name__)
model = torch.jit.load('ner.pt')
model.eval()
index2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}

def process(inputs):
    # TODO
    results = []
    for sequence in inputs:
        outputs = model(torch.tensor([sequence])).detach().numpy()
        entities = []
        entity_start = None
        entity_label = None
        for i, output in enumerate(outputs[0]):
            label = index2label[output]
            if label.startswith('B-'):
                if entity_start is not None:
                    entities.append({"start": entity_start, "end": i - 1, "label": entity_label})
                entity_start = i
                entity_label = label[2:]
            elif label == 'O':
                if entity_start is not None:
                    entities.append({"start": entity_start, "end": i - 1, "label": entity_label})
                    entity_start = None
                    entity_label = None
        if entity_start is not None:
            entities.append({"start": entity_start, "end": len(sequence) - 1, "label": entity_label})
        results.append(entities)
    return results


@app.route('/ner', methods=['POST'])
def ner():

    data = request.get_json()
    inputs = data['inputs']
    outputs = process(inputs)
    return jsonify(outputs)


if __name__ == '__main__':
    app.run(debug=True)
#task-end