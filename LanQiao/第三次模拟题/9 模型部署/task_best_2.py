#task-start
from flask import Flask, jsonify, request
import torch

app = Flask(__name__)
model = torch.jit.load('ner.pt')
model.eval()
index2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}

def process(inputs):
    results = []
    for sequence in inputs:
        outputs = model(torch.tensor([sequence])).detach().numpy()
        print(outputs)
        entities = []
        entity_start = None
        entity_end = None
        entity_label = None
        last_label = None
        print("outputs[0]:/n")
        print(outputs[0])   
        for i, output in enumerate(outputs[0]):
            label = index2label[output]
            # 首字母的情况：
            if last_label is None:
                if label == 'O'or label.startswith('I-'):
                    pass
                elif label.startswith('B-'):
                    entity_start = i
                    entity_end = i+1
                    entity_label = label[2:]
            # 非手字母的情况
            else:
                if label == 'O':
                    if entity_label != None:
                        if last_label.startswith('B-'):
                            entity_start = None
                            entity_end = None
                            entity_label = None
                        elif last_label.startswith('I-'):
                            if entity_label != None:
                                entities.append({"start": entity_start, "end": entity_end-1 , "label": entity_label})
                    entity_start = None
                    entity_label = None
                elif label.startswith('B-'):
                    if last_label.startswith('B-'):
                        entity_start = i
                        entity_label = label[2:]
                    elif last_label.startswith('I-'):
                        if entity_label != None:
                            entities.append({"start": entity_start, "end": entity_end-1 , "label": entity_label})
                    entity_start = i
                    entity_end = i+1
                    entity_label = label[2:]
                elif label.startswith('I-'):
                    if last_label == None:
                        pass
                    elif last_label.startswith('B-'):
                        if entity_label != label[2:]:
                            entities.append({"start": entity_start, "end": entity_end-1 , "label": entity_label})
                            entity_start = None
                            entity_end = None
                            entity_label = None
                        else :
                            entity_end =entity_end+1
            last_label = label
        results.append(entities)
    # TODO
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