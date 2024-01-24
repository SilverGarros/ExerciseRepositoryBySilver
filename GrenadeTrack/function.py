import os
import random
import shutil

from PIL import Image


def PickImgOfFolder(N, input_folder_path, output_floder_path):
    """
    从input_folder中随机选取N张图片复制到out_floder
    用于训练图像的随机挑选
    :param N: 目标图片数量：需要随机挑选的图片数量
    :param input_folder_path: 输入文件夹路径：需要进行挑选处理的图片文件夹路径
    :param output_floder_path: 随机挑选的图片保存的文件夹路径
    """
    input_folder = input_folder_path
    output_folder = output_floder_path
    target_count = N
    # 获取输入文件夹中的所有图片
    image_inputfloder_list = [file for file in os.listdir(input_folder) if
                              file.endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if len(image_inputfloder_list) < target_count:
        print("num of img in input_floder < target_count")
        os.makedirs(output_folder, exist_ok=True)
        for image_name in image_inputfloder_list:
            image_path = os.path.join(input_folder, image_name)
            target_path = os.path.join(output_folder, image_name)
            shutil.copyfile(image_path, target_path)
        image_outputfloder_list = [file for file in os.listdir(output_folder) if
                                   file.endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        while len(image_outputfloder_list) < target_count:
            # 计算需要复制的图片数量
            copy_count = target_count - len(image_outputfloder_list)
            if copy_count > len(image_outputfloder_list):
                copy_count = len(image_outputfloder_list)
            for image_name in random.choices(image_outputfloder_list, k=copy_count):
                image_path = os.path.join(input_folder, image_name)
                base_name, ext = os.path.splitext(image_name)
                new_image_name = f"{base_name}_copy{ext}"
                target_path = os.path.join(input_folder, new_image_name)
                shutil.copy(image_path, target_path)
            image_outputfloder_list = [file for file in os.listdir(output_folder)
                                       if file.endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            continue
    else:
        os.makedirs(output_folder, exist_ok=True)

        selected_images = random.sample(image_inputfloder_list, target_count)
        for image_name in selected_images:
            image_path = os.path.join(input_folder, image_name)
            target_path = os.path.join(output_floder_path, image_name)
            shutil.copyfile(image_path, target_path)


def get_average_image_size(folder_path):
    """
    got the average size of all the image(jpg/jpeg/png/gif) file in the folder
    获取文件夹内的所有图像的平均尺寸
    :param folder_path:
    :return:(ave_width,ave_height)
    """
    total_width = 0
    total_height = 0
    image_count = 0
    if not os.path.exists(folder_path):
        raise FileNotFoundError("Folder path Cannot Found!")
    else:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and any(
                    file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                try:
                    image = Image.open(file_path)
                    width, height = image.size
                    total_width += width
                    total_height += height
                    image_count += 1
                    image.close()
                except (IOError, SyntaxError) as e:
                    # 处理无法打开或解析的图像文件
                    print(f"Error processing image file: {file_path}")

        if image_count > 0:
            average_width = total_width / image_count
            average_height = total_height / image_count
            print("计算完成")
            return average_width, average_height
        else:
            print("计算出现异常，请优先确定文件夹内存在图像文件")


def PickImgInFolderAccuderingSize(input_folder, output_folder):
    """
    半成品函数，条件自己根据情况进行更改
    :param input_folder:
    :param output_folder:
    """
    # 确保输出文件夹存在，如果不存在则创建它
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有文件
    file_list = os.listdir(input_folder)

    # 遍历文件列表
    for file_name in file_list:
        # 构建文件的完整路径
        file_path = os.path.join(input_folder, file_name)

        # 检查文件是否是图片
        if file_name.endswith(".jpg"):
            # 打开图像文件
            image = Image.open(file_path)

            # 获取图像的宽度和高度
            width, height = image.size
            "下面为图像条件"
            # 检查图像是否满足条件
            if 100 < width < 200 and 100 < height < 200:
                # 复制文件到输出文件夹
                output_path = os.path.join(output_folder, file_name)
                shutil.copyfile(file_path, output_path)

    print("pick over！")

