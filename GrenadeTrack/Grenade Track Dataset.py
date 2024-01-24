from xml_processing_function import get_label_classes, batch_rename_labels

DataVOC = 'G:/TrainData/Grenade.v1i.voc/'
Annotations = DataVOC + 'Annotations'

labels = get_label_classes(Annotations)
print(labels)

# batch_rename_labels(Annotations, '0', 'grenade')
# labels = get_label_classes(Annotations)
# print(labels)

