import cv2
import os
import random
import numpy as np


def process_images(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)

        # 读取图像
        image = cv2.imread(input_path)

        if image is not None:
            # 保存原始图像
            save_image(output_folder, f"{image_file}_original", image)

            # 随机裁剪图像为640x640
            x, y, w, h = random_crop(image, 640, 640)
            cropped_image = image[y:y + h, x:x + w]
            save_image(output_folder, f"{image_file}_cropped", cropped_image)

            # 旋转图像90度
            rotated_image = rotate_image(cropped_image, 90)
            save_image(output_folder, f"{image_file}_rotated", rotated_image)


def random_crop(image, width, height):
    h, w, _ = image.shape
    x = random.randint(0, max(0, w - width))
    y = random.randint(0, max(0, h - height))
    return x, y, width, height


def rotate_image(image, angle):
    h, w, _ = image.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image


def save_image(output_folder, filename, image):
    output_path = os.path.join(output_folder, f"{filename}.jpg")
    cv2.imwrite(output_path, image)


def resize_and_rotate_images(input_folder, output_folder, size=1280):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)

        # 读取图像
        image = cv2.imread(input_path)

        if image is not None:
            # 调整图像大小为1280x1280
            resized_image = cv2.resize(image, (size, size))

            # 保存原始大小的图像
            # save_image(output_folder, f"{image_file}_original", resized_image)

            # 保存翻转0°的图像
            save_image(output_folder, f"{image_file}_rotated_0", resized_image)
            # 保存翻转90°的图像
            rotated_90_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
            save_image(output_folder, f"{image_file}_rotated_90", rotated_90_image)

            # 保存翻转270°的图像
            rotated_270_image = cv2.rotate(resized_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            save_image(output_folder, f"{image_file}_rotated_270", rotated_270_image)


if __name__ == "__main__":
    # 设置输入和输出文件夹，以及其他参数
    input_folder = "G:\TrainData\Grenade_BIGC\pre\img_batch2"
    output_folder = "G:/TrainData/Grenade_BIGC/img_batch_enhancement"
    # 调用函数
    # process_images(input_folder, output_folder)

    # 调用函数
    resize_and_rotate_images(input_folder, output_folder)
