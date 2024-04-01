def process(inputs):
    results = []
    for sequence in inputs:
        outputs = model(torch.tensor([sequence])).detach().numpy()
        print(outputs)
        entities = []
        entity_start = None
        entity_label = None
        for i, output in enumerate(outputs[0]):
            label = index2label[output]
            if label == 'O':
                entity_start = None
                entity_label = None
            elif label.startswith('B-'):
                '''if entity_start is not None:
                        entity_start = None
                        entity_label = None'''
                entity_start = i
                entity_label = label[2:]
            elif label.startswith('I-'):
                if  entity_start is not None:
                    entities.append({"start": entity_start, "end": i , "label": entity_label})
                elif:
                    entity_start = None
                    entity_label = None

        results.append(entities)
    # TODO
    return results