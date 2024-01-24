from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model

if __name__ == '__main__':
    pretrained_model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    pretrained_model.train(data='G:/TrainData/Grenade_BIGC/imgselect/dataYOLO/dataYOLOv8/data_config.yaml', epochs=50, imgsz=640)
