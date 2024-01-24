import os
import shutil
import glob
import xml.etree.ElementTree as ET
from PIL import Image


def filter_labels(xml_folder, keep_classes):
    """
    对xml_folder文件夹中的xml标签文件进行批处理，保留列表keep_classes中的类别其他都删除
    :param xml_folder: 想要批量处理的xml文件所在文件夹
    :param keep_classes:list类型，格式['keep_class1_name','keep_class2_name']
    :return:
    """
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_folder, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            objects = root.findall('object')
            for obj in objects:
                name = obj.find('name').text
                if name not in keep_classes:
                    root.remove(obj)
                    print("From" + filename + '.xml中移除了' + name + '类')

            tree.write(xml_path)


def delete_labels(xml_folder, delete_classes):
    """
    对xml_folder文件夹中的xml标签文件进行批处理，删除某类类别
    :param xml_folder: 想要批量处理的xml文件所在文件夹
    :param delete_classes:list类型，格式['keep_class1_name','keep_class2_name']
    :return:
    """
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_folder, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            objects = root.findall('object')
            for obj in objects:
                name = obj.find('name').text
                if name in delete_classes:
                    root.remove(obj)
                    print("From" + filename + '.xml中移除了' + name + '类')

            tree.write(xml_path)


def batch_rename_labels(folder_path, old_label, new_label):
    # 获取文件夹中所有的XML文件
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]

    # 遍历每个XML文件
    for xml_file in xml_files:
        xml_path = os.path.join(folder_path, xml_file)

        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 查找需要更改的标签名并进行替换
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label == old_label:
                obj.find('name').text = new_label
                print(xml_path + ' 中的label ' + old_label + '替换为' + new_label)

        # 保存修改后的XML文件
        tree.write(xml_path)

    print(old_label + '替换为' + new_label + "标签名批量更改完成！")


def get_label_classes(xml_folder):
    """
    返回文件夹内所有xml文件中yolo的标签类别
    :param xml_folder:
    :return:
    """
    label_classes = set()

    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_folder, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            objects = root.findall('object')
            for obj in objects:
                name = obj.find('name').text
                label_classes.add(name)

    return label_classes


def get_label_classesFromXML(xml_path):
    """
    返回文件夹内所有xml文件中yolo的标签类别
    :param xml_folder:
    :return:
    """
    label_classes = set()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('object')
    for obj in objects:
        name = obj.find('name').text
        label_classes.add(name)

    return label_classes


def add_yolo_labels(folder_path):
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        xml_file = os.path.splitext(image_file)[0] + '.xml'
        xml_path = os.path.join(folder_path, xml_file)

        # Open the image
        image = Image.open(image_path)
        width, height = image.size

        # Calculate box coordinates
        box_left = 0
        box_upper = int(height * (1 - 3 / 4))
        box_right = width
        box_lower = height

        # Create the XML root element
        root = ET.Element('annotation')

        # Add filename element
        filename_elem = ET.SubElement(root, 'filename')
        filename_elem.text = image_file

        # Add size element
        size_elem = ET.SubElement(root, 'size')
        width_elem = ET.SubElement(size_elem, 'width')
        width_elem.text = str(width)
        height_elem = ET.SubElement(size_elem, 'height')
        height_elem.text = str(height)

        # Add object element
        object_elem = ET.SubElement(root, 'object')
        name_elem = ET.SubElement(object_elem, 'name')
        name_elem.text = 'kg'
        bndbox_elem = ET.SubElement(object_elem, 'bndbox')
        xmin_elem = ET.SubElement(bndbox_elem, 'xmin')
        xmin_elem.text = str(box_left)
        ymin_elem = ET.SubElement(bndbox_elem, 'ymin')
        ymin_elem.text = str(box_upper)
        xmax_elem = ET.SubElement(bndbox_elem, 'xmax')
        xmax_elem.text = str(box_right)
        ymax_elem = ET.SubElement(bndbox_elem, 'ymax')
        ymax_elem.text = str(box_lower)

        # Create and save the XML file
        tree = ET.ElementTree(root)
        tree.write(xml_path)

        print(f"XML file created: {xml_file}")

    print("Labeling complete.")


def SiftingBasedLabel(folder_xml, folder_Img, folder_image, keywords):
    # 遍历文件夹A中的XML文件
    for root, dirs, files in os.walk(folder_xml):
        for file in files:
            if file.lower().endswith('.xml'):
                # 提取文件名中的关键字
                file_name = os.path.splitext(file)[0]
                file_keywords = set(file_name.split('_'))
                # 检查关键字是否在指定的关键字集合中
                if file_keywords.issubset(keywords):
                    # 构建源文件的路径
                    source_xml = os.path.join(root, file)
                    source_image = os.path.join(folder_Img, file_name + '.jpg')
                    # 构建目标文件的路径
                    destination_xml = os.path.join(folder_image, file)
                    destination_image = os.path.join(folder_image, file_name + '.jpg')
                    # 复制XML文件和对应的图片文件到文件夹C
                    shutil.copy2(source_xml, destination_xml)
                    shutil.copy2(source_image, destination_image)
                    print(destination_xml + "已经复制到" + folder_image + "中")
                    print(destination_image + "已经复制到" + folder_image + "中")


def delete_images_with_label(folder_xml, folder_Img, keyword):
    for root, dirs, files in os.walk(folder_xml):
        for file in files:
            if file.lower().endswith('.xml'):
                xml_file = os.path.join(root, file)

                image_file = os.path.join(folder_Img, os.path.splitext(file)[0] + '.jpg')
                label_names = get_label_classesFromXML(xml_file)
                print(label_names)
                if keyword in label_names:
                    # 删除图像文件和对应的 XML 文件
                    os.remove(xml_file)
                    os.remove(image_file)
                    print(f"Deleted {xml_file}")
                    print(f"Deleted {image_file}")


def delete_images_without_label(folder_xml, folder_Img, keyword):
    for root, dirs, files in os.walk(folder_xml):
        for file in files:
            if file.lower().endswith('.xml'):
                xml_file = os.path.join(root, file)

                image_file = os.path.join(folder_Img, os.path.splitext(file)[0] + '.jpg')
                label_names = get_label_classesFromXML(xml_file)
                print(label_names)
                if keyword not in label_names:
                    # 删除图像文件和对应的 XML 文件
                    os.remove(xml_file)
                    os.remove(image_file)
                    print(f"Deleted {xml_file}")
                    print(f"Deleted {image_file}")


def delete_files_with_nf_class(xml_folder, image_folder, delect_label):
    # 获取所有XML文件路径
    xml_files = glob.glob(os.path.join(xml_folder, '*.xml'))

    for xml_file in xml_files:
        # 解析XML文件
        tree = ET.parse(xml_file)
        root = tree.getroot()

        contains_nf_class = False
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            if delect_label in obj_name:
                contains_nf_class = True
                break

        # 如果包含"nf"类别，则删除XML文件和对应的图片文件
        if contains_nf_class:
            xml_filename = os.path.basename(xml_file)
            image_filename = os.path.splitext(xml_filename)[0] + '.jpg'
            image_path = os.path.join(image_folder, image_filename)

            if os.path.exists(xml_file):
                os.remove(xml_file)
                print(f"Deleted XML file: {xml_file}")

            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted image file: {image_path}")
