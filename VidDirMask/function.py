import os


def get_filename_without_extension(file_path):
    # 使用os.path.basename获取文件名
    filename_with_extension = os.path.basename(file_path)

    # 使用os.path.splitext获取文件名和扩展名的元组
    filename, file_extension = os.path.splitext(filename_with_extension)

    return filename
