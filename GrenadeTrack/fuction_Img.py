import os
import shutil
from PIL import Image
import cv2
import numpy as np


def chachong_images(folder_a, folder_b, folder_c, folder_d):
    """
    对于文件夹B中的图像文件，如果在A中存在则移动到C，否则移动到B
    :param folder_a:
    :param folder_b:
    :param folder_c:
    :param folder_d:
    :return: None
    """
    # 创建目标文件夹
    os.makedirs(folder_c, exist_ok=True)
    os.makedirs(folder_d, exist_ok=True)

    # 获取文件夹 B 中的所有图片文件
    images_b = [f for f in os.listdir(folder_b) if os.path.isfile(os.path.join(folder_b, f))]

    # 遍历文件夹 B 中的图片文件
    for image_b in images_b:
        # 构建图片文件的完整路径
        image_b_path = os.path.join(folder_b, image_b)

        # 获取图片文件名
        image_b_name = os.path.splitext(image_b)[0]

        # 构建文件夹 A 中对应的图片文件路径
        image_a_path = os.path.join(folder_a, image_b_name)

        # 判断文件夹 A 中对应的图片文件是否存在
        if os.path.exists(image_a_path):
            # 将图片文件移动到文件夹 C
            shutil.move(image_b_path, os.path.join(folder_c, image_b))
        else:
            # 将图片文件移动到文件夹 D
            shutil.move(image_b_path, os.path.join(folder_d, image_b))

    print("图片文件移动完成！")


def move_images_based_filenamekeyword(source_folder, destination_folder, keywordslist):
    if not os.path.exists(source_folder) or not os.path.isdir(source_folder):
        print(f"Error: {source_folder} is not a valid directory.")
        return

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, _, files in os.walk(source_folder):
        for file_name in files:
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                for keyword in keywordslist:
                    if keyword.lower() in file_name.lower():
                        source_path = os.path.join(root, file_name)
                        destination_path = os.path.join(destination_folder, file_name)

                        try:
                            count = 1
                            while os.path.exists(destination_path):
                                name, ext = os.path.splitext(file_name)
                                new_name = f"{name}_{count}{ext}"
                                destination_path = os.path.join(destination_folder, new_name)
                                count += 1

                            shutil.move(source_path, destination_path)
                            print(f"Copied: {source_path} -> {destination_path}")
                            break  # Move to the next file once copied
                        except Exception as e:
                            print(f"Error moveing {source_path}: {e}")


def copy_images_based_filenamekeyword(source_folder, destination_folder, keywordslist):
    if not os.path.exists(source_folder) or not os.path.isdir(source_folder):
        print(f"Error: {source_folder} is not a valid directory.")
        return

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, _, files in os.walk(source_folder):
        for file_name in files:
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                for keyword in keywordslist:
                    if keyword.lower() in file_name.lower():
                        source_path = os.path.join(root, file_name)
                        destination_path = os.path.join(destination_folder, file_name)

                        try:
                            count = 1
                            while os.path.exists(destination_path):
                                name, ext = os.path.splitext(file_name)
                                new_name = f"{name}_{count}{ext}"
                                destination_path = os.path.join(destination_folder, new_name)
                                count += 1

                            shutil.copy(source_path, destination_path)
                            print(f"Copied: {source_path} -> {destination_path}")
                            break  # Move to the next file once copied
                        except Exception as e:
                            print(f"Error moveing {source_path}: {e}")


def rename_files(folder_path, name):
    # 获取文件夹内所有文件的路径
    file_paths = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    # 将文件路径按字母顺序排序
    file_paths.sort()

    # 重命名图片和XML文件
    for idx, file_path in enumerate(file_paths[::2], start=1):
        # 分离文件名和扩展名
        file_name, file_ext = os.path.splitext(os.path.basename(file_path))
        new_name = name + f"_{str(idx).zfill(3)}"

        # 重命名图片文件
        new_image_name = f"{new_name}{file_ext}"
        new_image_path = os.path.join(folder_path, new_image_name)
        try:
            os.rename(file_path, new_image_path)
        except Exception as e:
            print(f"Error renaming image file: {e}")
            continue  # 继续处理下一个文件

        # 重命名对应的XML文件
        xml_file_path = os.path.join(folder_path, f"{file_name}.xml")
        if os.path.isfile(xml_file_path):
            new_xml_name = f"{new_name}.xml"
            new_xml_path = os.path.join(folder_path, new_xml_name)
            try:
                os.rename(xml_file_path, new_xml_path)
            except Exception as e:
                print(f"Error renaming XML file: {e}")
                # 如果重命名XML文件失败，将图片文件名恢复为原始状态，避免不一致
                os.rename(new_image_path, file_path)


def delete_same_images(folder_A, folder_B):
    common_images = set(os.listdir(folder_A)) & set(os.listdir(folder_B))

    for image in common_images:
        image_path_A = os.path.join(folder_A, image)
        image_path_B = os.path.join(folder_B, image)

        if os.path.isfile(image_path_A) and os.path.isfile(image_path_B):
            os.remove(image_path_A)
            print(f"Deleted {image_path_A}")


def convert_png_to_jpg(source_folder, backup_folder):
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)

    # 获取源文件夹中所有的文件
    file_list = os.listdir(source_folder)

    for filename in file_list:
        # 确保文件是PNG格式
        if filename.lower().endswith('.png'):
            source_path = os.path.join(source_folder, filename)
            backup_path = os.path.join(backup_folder, filename)
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(source_folder, jpg_filename)

            # 备份PNG文件到目标文件夹B
            shutil.move(source_path, backup_path)

            # 转换PNG文件为JPG格式并保存在目标文件夹A
            img = Image.open(backup_path)
            img = img.convert('RGB')
            img.save(jpg_path, 'JPEG')


import cv2
from PIL import Image
import numpy as np


def extract_frames(input_folder, output_folder, frame_rate=1):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有视频文件
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)

        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算抽帧的间隔
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / frame_rate)

        for i in range(0, frame_count, frame_interval):
            # 设置视频的当前帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            # 读取视频帧
            ret, frame = cap.read()

            if ret:
                # 保存帧为图像文件
                frame_filename = os.path.join(output_folder, f"{video_file}_frame_{i}.jpg")
                cv2.imwrite(frame_filename, frame)

        # 关闭视频文件
        cap.release()

# 示例用法：
# video_to_img('your_video.mp4', 'output_images_folder')
