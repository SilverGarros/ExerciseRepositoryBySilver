#quantize-start
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

def quantize_model(model_path, quantized_model_path):
    # TODO
    model=tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_quant_model = converter.convert()
    with open(quantized_model_path, 'wb') as f:
        f.write(tflite_quant_model)

def prediction_label(test_sentence, model_path):
    # TODO
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    with open('word_index.json', 'r', encoding='utf-8') as f:
        word_index = json.load(f)


    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    sequences = tokenizer.texts_to_sequences([test_sentence])

    padded = pad_sequences(sequences, maxlen=100)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], padded)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    return np.argmax(prediction)


def main():
    # 量化模型
    quantize_model('/home/project/model.h5', '/home/project/quantized_model.tflite')
    # 测试示例
    test_sentence = "一个 公益广告 ： 爸爸 得 了 老年痴呆  儿子 带 他 去 吃饭  盘子 里面 剩下 两个 饺子  爸爸 直接 用手 抓起 饺子 放进 了 口袋  儿子 愣住 了  爸爸 说  我 儿子 最爱 吃 这个 了  最后 广告 出字 ： 他 忘记 了 一切  但 从未 忘记 爱 你    「 转 」"
    print(prediction_label(test_sentence, '/home/project/quantized_model.tflite'))
    print(0)

if __name__ == "__main__":  
    main()
#quantize-end