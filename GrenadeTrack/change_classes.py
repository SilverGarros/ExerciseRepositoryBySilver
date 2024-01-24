from xml_processing_function import get_label_classes, batch_rename_labels

labels=get_label_classes("G:/TrainData/Grenade_BIGC/imgselect/dataVOC/Annotations")
print(labels)
batch_rename_labels("G:/TrainData/Grenade_BIGC/imgselect/dataVOC/Annotations",'grenade',"Grenade")
labels=get_label_classes("G:/TrainData/Grenade_BIGC/imgselect/dataVOC/Annotations")
print(labels)