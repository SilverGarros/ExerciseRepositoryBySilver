import os
import shutil


def find_missing_xml(a_folder, b_folder):
    """
    获取a文件夹中图像中缺少b文件夹中xml文件的图片名称
    :param a_folder:
    :param b_folder:
    :return: 返回缺少xml文件的图片名称
    """
    # 获取a文件夹中的图片文件名
    image_files = os.listdir(os.path.join(a_folder))
    image_files = [os.path.splitext(file)[0] for file in image_files]

    # 获取b文件夹中的XML文件名
    xml_files = os.listdir(os.path.join(b_folder))
    xml_files = [os.path.splitext(file)[0] for file in xml_files]

    # 查找缺少对应图片的XML文件
    missing_xml_files = []
    for xml_file in xml_files:
        if xml_file not in image_files:
            missing_xml_files.append(xml_file)
    if len(missing_xml_files) > 0:
        print("缺少对应图片的XML文件:")
        for xml_file in missing_xml_files:
            print(xml_file)
    else:
        print("所有XML文件都有对应的图片。")
    return missing_xml_files


def got_missing_xml(Img_folder, Xml_folder, MissingXml_folder):
    # 获取a文件夹中的图片文件名
    image_files = os.listdir(Img_folder)
    image_files = [os.path.splitext(file)[0] for file in image_files]

    # 获取b文件夹中的XML文件名
    xml_files = os.listdir(Xml_folder)
    xml_files = [os.path.splitext(file)[0] for file in xml_files]

    # 查找缺少对应XML的图片文件
    missing_xml_files = []
    for xml_file in xml_files:
        if xml_file in image_files:
            missing_xml_files.append(xml_file)
    if len(missing_xml_files) > 0:
        print("缺少对应图片的XML文件:")
        # 将缺失的XML文件复制到C文件夹中
        for xml_file in missing_xml_files:
            source_path = os.path.join(Xml_folder, xml_file + '.xml')
            destination_path = os.path.join(MissingXml_folder, xml_file + '.xml')
            shutil.copyfile(source_path, destination_path)
            print(xml_file + 'a文件夹图像缺少的xml文件已经从B文件夹添加到C文件夹中')
    else:
        # 将缺失的XML文件复制到C文件夹中
        for xml_file in missing_xml_files:
            source_path = os.path.join(Xml_folder, xml_file + '.xml')
            destination_path = os.path.join(MissingXml_folder, xml_file + '.xml')
            shutil.copyfile(source_path, destination_path)
            print(xml_file + 'a文件夹图像缺少的xml文件已经从B文件夹添加到C文件夹中')


def find_missing_img(a_folder, b_folder):
    """
    获取a文件夹中图像中缺少b文件夹中xml文件的图片名称
    :param a_folder:
    :param b_folder:
    :return: 返回缺少xml文件的图片名称
    """
    # 获取a文件夹中的图片文件名
    image_files = os.listdir(a_folder)
    image_files = [os.path.splitext(file)[0] for file in image_files]

    # 获取b文件夹中的XML文件名
    xml_files = os.listdir(b_folder)
    xml_files = [os.path.splitext(file)[0] for file in xml_files]

    # 查找缺少对应图片的XML文件
    missing_image_files = []
    for image_file in image_files:
        if image_file not in xml_files:
            missing_image_files.append(image_file)

    if len(missing_image_files) > 0:
        print("缺少对xml的图片文件:")
        for image_file in missing_image_files:
            print(image_file + ".xml")
    else:
        print("所有xml都有对应的图片文件。")

    return missing_image_files


def move_images_without_xml(folder_A, folder_B):
    for root, _, files in os.walk(folder_A):
        for image_file in files:
            if image_file.lower().endswith('.jpg') or image_file.lower().endswith('.png'):
                image_path = os.path.join(root, image_file)
                xml_file = os.path.join(root, os.path.splitext(image_file)[0] + '.xml')

                if not os.path.exists(xml_file):
                    destination_path = os.path.join(folder_B, image_file)
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    shutil.move(image_path, destination_path)
                    print(f"Moved {image_path} to {destination_path}")


def get_files_in_folder(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


def rename_files(folder_a, folder_b, name_prefix):
    # 获取文件夹a和文件夹b中的所有文件列表
    image_files_original = get_files_in_folder(folder_a)
    xml_files_original = get_files_in_folder(folder_b)

    # 获取文件夹a中的所有图片文件
    image_files = [f for f in image_files_original if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()

    # 获取文件夹b中的所有xml文件
    xml_files = [f for f in xml_files_original if f.lower().endswith('.xml')]
    xml_files.sort()

    if len(image_files) != len(xml_files):
        print("图片文件和XML文件数量不匹配，无法进行重命名。")
        return

    try:
        for index, (image_file, xml_file) in enumerate(zip(image_files, xml_files), 1):
            # 构造新的文件名
            new_name = f"{name_prefix}_{str(index).zfill(5)}"

            # 重命名图片文件
            image_extension = os.path.splitext(image_file)[1]
            new_image_name = f"{new_name}{image_extension}"
            os.rename(os.path.join(folder_a, image_file), os.path.join(folder_a, new_image_name))

            # 重命名xml文件
            xml_extension = os.path.splitext(xml_file)[1]
            new_xml_name = f"{new_name}{xml_extension}"
            os.rename(os.path.join(folder_b, xml_file), os.path.join(folder_b, new_xml_name))

            print(f"重命名: {image_file} -> {new_image_name}, {xml_file} -> {new_xml_name}")

    except Exception as e:
        print("重命名失败，正在回退到原始状态...")
        for original_image_file in image_files_original:
            os.rename(os.path.join(folder_a, original_image_file), os.path.join(folder_a, original_image_file))
        for original_xml_file in xml_files_original:
            os.rename(os.path.join(folder_b, original_xml_file), os.path.join(folder_b, original_xml_file))
        print("已回退到原始状态。")
        raise e  # 抛出异常以通知调用者重命名过程中出现了问题
