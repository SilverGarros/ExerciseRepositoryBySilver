
index2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}

def process(inputs):
    results = []
    for sequence in inputs:
        # outputs = model(torch.tensor([sequence])).detach().numpy()
        outputs =[[3,0,4,0,3, 4 ,0,3,0,4,0,1,2,0,4,3,1,2,2,2,2,3,4,4,4,4,3,4,1,2]]
        outputs = [[0,3,4,1,2,0,3,4,0,0]]
        print(outputs)
        entities = []
        entity_start = None
        entity_end = None
        entity_label = None
        last_label = None
        print("outputs[0]:")
        print(outputs[0])   
        for i, output in enumerate(outputs[0]):
            label = index2label[output]
            # 首字母的情况：
            if last_label is None:
                if label == 'O'or label.startswith('I-'):
                    pass
                elif label.startswith('B-'):
                    entity_start = i
                    entity_end = i
                    entity_label = label[2:]
            # 非首非末字母的情况
            else:
                if label == 'O':
                    if entity_label != None:
                        if last_label.startswith('I-'):
                            if entity_label != None:
                                entities.append({"start": entity_start, "end": entity_end , "label": entity_label})
                    entity_start = None
                    entity_label = None
                    entity_end = None
                elif label.startswith('B-'):
                    if last_label.startswith('B-'):
                        entity_start = i
                        entity_label = label[2:]
                    elif last_label.startswith('I-'):
                        if entity_label != None:
                            entities.append({"start": entity_start, "end": entity_end , "label": entity_label})
                    entity_start = i
                    entity_end = i
                    entity_label = label[2:]
                elif label.startswith('I-'):
                    if last_label == None:
                        pass
                    elif last_label.startswith('B-'):
                        if entity_label != label[2:]:
                            # entities.append({"start": entity_start, "end": entity_end , "label": entity_label})
                            entity_start = None
                            entity_end = None
                            entity_label = None
                        else :
                            entity_end =i
                    elif last_label.startswith('I-'):
                        if entity_label != label[2:]:
                            print(entities)
                            entities.append({"start": entity_start, "end": entity_end , "label": entity_label})
                            entity_start = None
                            entity_end = None
                            entity_label = None
                        else:entity_end = i
            last_label = label
            # 末尾的请跨
            if i ==len(outputs[0])-1:
                print("末尾情况")
                if label.startswith('I-'):
                    if last_label.startswith('I-'):
                        if entity_label == label[2:]:
                            print(entities)
                            entities.append({"start": entity_start, "end": entity_end , "label": entity_label})

            
            print(entities)
        results.append(entities)

    # TODO
    print(outputs)
    return results


inputs = [[100, 1003, 1009, 106, 102, 1004, 1003]]
outputs = process(inputs)
print(outputs)