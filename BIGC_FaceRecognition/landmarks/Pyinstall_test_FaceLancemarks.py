import pickle

import numpy as np

PATH_model_predictor_68_face_landmark = "./model/face_recognition_resnet_model.dat"
PATH_model_face_recognition_resnet_model = "./model/shape_predictor_68_face_landmarks.dat"


def get_face_landmarks_128vectors_from_img(img_path):
    global face_descriptor
    import cv2
    import dlib
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(img)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(PATH_model_face_recognition_resnet_model)
    face_rec_model = dlib.face_recognition_model_v1(PATH_model_predictor_68_face_landmark)
    dets = detector(img, 1)  # 人脸标定
    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                     face.bottom()))

        shape = shape_predictor(img2, face)  # 提取68个特征点
        for i, pt in enumerate(shape.parts()):
            # print('Part {}: {}'.format(i, pt))
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
            # print(type(pt))
        """
        shape.part(0) 和 shape.part(1) 分别表示人脸关键点的第一个和第二个点的坐标。在提取人脸的68个特征点时，每个特征点都有 x 和 y 坐标。
        shape.part(0) 表示第一个特征点的坐标，通常是人脸的左眼睛的左侧。shape.part(1) 表示第二个特征点的坐标，通常是人脸的左眼睛的右侧.
        """
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

        cv2.namedWindow(img_path + str(index), cv2.WINDOW_AUTOSIZE)
        cv2.imshow(img_path + str(index), img)

        face_descriptor = face_rec_model.compute_face_descriptor(img2, shape)  # 计算人脸的128维的向量
        face_descriptor = np.array(face_descriptor)
        """"""
        all_face_landmarks_vectors = []

        # 128维度向量保存
        vector_info = {
            'name': img_path + str(index),
            'vector': face_descriptor
        }
        all_face_landmarks_vectors.append(vector_info)
        """"""
        # 保存数据结构到文件
        with open('./face_vectors.pkl', 'wb') as f:
            pickle.dump(all_face_landmarks_vectors, f)

    return face_descriptor.reshape(1, 128).shape


if __name__ == "__main__":
    # main()
    print(get_face_landmarks_128vectors_from_img(r"./mabaoguo_test.jpg"))
