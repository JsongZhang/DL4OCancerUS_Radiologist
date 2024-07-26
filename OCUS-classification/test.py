# import torch
# from torch.utils.data import DataLoader
# from dataset import ImageDataset, val_transform, test_transform
# import os
# import timm
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# from tqdm import tqdm
# import shutil
#
#
# model_name = 'resnet50'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Test = True
# model = timm.create_model(model_name, pretrained=False, num_classes=3)
# if Test:
#
#
#     state_dict = torch.load('./0.8775_tt_newest_best_model_resnet50_acc.pth', map_location='cpu')
#     model.load_state_dict(state_dict)
#     model.to(device)
#     print('Model Loaded!!!')
#
#
#
# def test_model(model, data_loader):
#     model.eval()
#     # correct = 0
#     # total = 0
#     all_predictions = []
#     all_labels = []
#
#     # with torch.no_grad():
#     #     for images, labels in data_loader:
#     #         images = images.to(device)
#     #         labels = labels.to(device)
#     #         outputs = model(images)
#     #         _, predicted = torch.max(outputs.data, 1)
#     #         all_predictions.extend(predicted.cpu().numpy())
#     #         all_labels.extend(labels.cpu().numpy())
#     #
#     # # Compute the classification metrics
#     # print(classification_report(all_labels, all_predictions, target_names=data_loader.dataset.classes))
#     #
#     # # Calculate confusion matrix
#     # cm = confusion_matrix(all_labels, all_predictions)
#     # print("\nConfusion Matrix:")
#     # print(cm)
#     #
#     # # Plot confusion matrix
#     # plt.figure(figsize=(10, 7))
#     # sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=data_loader.dataset.classes,
#     #             yticklabels=data_loader.dataset.classes)
#     # plt.xlabel('Predicted')
#     # plt.ylabel('True')
#     # plt.title('Confusion Matrix')
#     # plt.show()
#     model.eval()
#     correct = 0
#     total = 0
#     all_predictions = []
#     all_labels = []
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(tqdm(data_loader, dynamic_ncols=True, ascii=True)):
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             all_predictions.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#             #计算这个批次中每个图像的索引偏移
#             batch_start_idx = batch_idx * data_loader.batch_size
#             for i, pred in enumerate(predicted.cpu().numpy()):
#                 # 计算全局索引位置
#                 img_idx = batch_start_idx + i
#                 # 获取正确的图像路径
#                 img_path = data_loader.dataset.images[img_idx]
#                 pred_class = class_names[pred]
#                 target_dir = os.path.join(target_root_dir, pred_class)
#                 target_path = os.path.join(target_dir, os.path.basename(img_path))
#                 shutil.copy(img_path, target_path)
#                 print(f"Copied {img_path} to {target_path}")
#
#     # 计算全局指标
#     accuracy = 100 * correct / total
#     precision = precision_score(all_labels, all_predictions, average=None)  # 按类别计算精度
#     recall = recall_score(all_labels, all_predictions, average=None)  # 按类别计算召回率
#     f1 = f1_score(all_labels, all_predictions, average=None)  # 按类别计算F1分数
#     conf_matrix = confusion_matrix(all_labels, all_predictions)  # 获取混淆矩阵
#
#     print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
#     print('Precision by class:', precision)
#     print('Recall by class:', recall)
#     print('F1 Score by class:', f1)
#     print('Confusion Matrix:\n', conf_matrix)
#
# test_dataset = ImageDataset(root_dir='/data1/zhangjiansong/py_project/Hosptial_other_project/Fujian_2/Doc_WangYanli/data/OvarianCancerUS/External_test_0619', transform=test_transform)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#
# # 指定分类后的图像存储路径
# target_root_dir = './out_0619_Gen2text_images'
# class_names = test_dataset.classes  # 使用 dataset 的 classes 属性
# # 创建类别目录
# for class_name in class_names:
#     os.makedirs(os.path.join(target_root_dir, class_name), exist_ok=True)
#
# test_model(model, test_loader)


import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset, val_transform, test_transform
import os
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import shutil

model_name = 'resnet50'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
Test = True
model = timm.create_model(model_name, pretrained=False, num_classes=3)
if Test:
    state_dict = torch.load('./0.8775_tt_newest_best_model_resnet50_acc.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    print('Model Loaded!!!')

# def test_model(model, data_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     all_predictions = []
#     all_labels = []
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(tqdm(data_loader, dynamic_ncols=True, ascii=True)):
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             all_predictions.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#             # 计算这个批次中每个图像的索引偏移
#             batch_start_idx = batch_idx * data_loader.batch_size
#             for i, pred in enumerate(predicted.cpu().numpy()):
#                 # 计算全局索引位置
#                 img_idx = batch_start_idx + i
#                 # 获取正确的图像路径
#                 img_path = data_loader.dataset.images[img_idx]
#                 pred_class = class_names[pred]
#                 target_dir = os.path.join(target_root_dir, pred_class)
#                 target_path = os.path.join(target_dir, os.path.basename(img_path))
#                 try:
#                     shutil.copy(img_path, target_path)
#                     print(f"Copied {img_path} to {target_path}")
#                 except Exception as e:
#                     print(f"Failed to copy {img_path} to {target_path}: {e}")
#
#     # 计算全局指标
#     accuracy = 100 * correct / total
#     precision = precision_score(all_labels, all_predictions, average=None)  # 按类别计算精度
#     recall = recall_score(all_labels, all_predictions, average=None)  # 按类别计算召回率
#     f1 = f1_score(all_labels, all_predictions, average=None)  # 按类别计算F1分数
#     conf_matrix = confusion_matrix(all_labels, all_predictions)  # 获取混淆矩阵
#
#     print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
#     print('Precision by class:', precision)
#     print('Recall by class:', recall)
#     print('F1 Score by class:', f1)
#     print('Confusion Matrix:\n', conf_matrix)


# def test_model(model, data_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     all_predictions = []
#     all_labels = []
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(tqdm(data_loader, dynamic_ncols=True, ascii=True)):
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             all_predictions.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#             batch_start_idx = batch_idx * data_loader.batch_size
#             for i, pred in enumerate(predicted.cpu().numpy()):
#                 img_idx = batch_start_idx + i
#                 if img_idx >= len(data_loader.dataset.images):  # 检查索引是否超出范围
#                     print(f"Index {img_idx} out of range.")
#                     continue
#
#                 img_path = data_loader.dataset.images[img_idx]
#                 pred_class = class_names[pred]
#                 target_dir = os.path.join(target_root_dir, pred_class)
#                 target_path = os.path.join(target_dir, os.path.basename(img_path))
#                 if not os.path.exists(img_path):
#                     print(f"Image path does not exist: {img_path}")
#                     continue
#
#                 try:
#                     shutil.copy(img_path, target_path)
#                     print(f"Successfully copied {img_path} to {target_path}")
#                 except Exception as e:
#                     print(f"Failed to copy {img_path} to {target_path}: {e}")
#
#     # 计算全局指标
#     accuracy = 100 * correct / total
#     precision = precision_score(all_labels, all_predictions, average=None)
#     recall = recall_score(all_labels, all_predictions, average=None)
#     f1 = f1_score(all_labels, all_predictions, average=None)
#     conf_matrix = confusion_matrix(all_labels, all_predictions)
#
#     print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
#     print('Precision by class:', precision)
#     print('Recall by class:', recall)
#     print('F1 Score by class:', f1)
#     print('Confusion Matrix:\n', conf_matrix)
#
#
# test_dataset = ImageDataset(root_dir='/data1/zhangjiansong/py_project/Hosptial_other_project/Fujian_2/Doc_WangYanli/data/OvarianCancerUS/External_test_0619', transform=test_transform)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#
# # 指定分类后的图像存储路径
# target_root_dir = './out_0619_Gen2text_images'
# class_names = test_dataset.classes  # 使用 dataset 的 classes 属性
# # 创建类别目录
# for class_name in class_names:
#     os.makedirs(os.path.join(target_root_dir, class_name), exist_ok=True)
#
# test_model(model, test_loader)
import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset, val_transform, test_transform
import os
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import shutil

import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset, val_transform, test_transform
import os
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import shutil
import uuid

model_name = 'resnet50'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
Test = True
model = timm.create_model(model_name, pretrained=False, num_classes=3)
if Test:
    state_dict = torch.load('./0.8775_tt_newest_best_model_resnet50_acc.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    print('Model Loaded!!!')


def test_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    copied_files = set()
    skipped_images = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, dynamic_ncols=True, ascii=True)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_start_idx = batch_idx * data_loader.batch_size
            for i, pred in enumerate(predicted.cpu().numpy()):
                img_idx = batch_start_idx + i
                if img_idx >= len(data_loader.dataset.images):
                    print(f"Index {img_idx} out of range.")
                    continue

                img_path = data_loader.dataset.images[img_idx]
                pred_class = class_names[pred]
                target_dir = os.path.join(target_root_dir, pred_class)
                unique_id = str(uuid.uuid4())  # 生成唯一ID
                new_file_name = f"{pred_class}_{unique_id}_{os.path.basename(img_path)}"
                target_path = os.path.join(target_dir, new_file_name)

                if not os.path.exists(img_path):
                    print(f"Image path does not exist: {img_path}")
                    skipped_images.append(img_path)
                    continue

                if img_path not in copied_files:
                    try:
                        shutil.copy(img_path, target_path)
                        copied_files.add(img_path)
                        print(f"Successfully copied {img_path} to {target_path}")
                    except Exception as e:
                        print(f"Failed to copy {img_path} to {target_path}: {e}")
                        skipped_images.append(img_path)

    # 计算全局指标
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average=None)
    recall = recall_score(all_labels, all_predictions, average=None)
    f1 = f1_score(all_labels, all_predictions, average=None)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    print('Precision by class:', precision)
    print('Recall by class:', recall)
    print('F1 Score by class:', f1)
    print('Confusion Matrix:\n', conf_matrix)

    # 打印跳过的图像信息
    if skipped_images:
        print("Skipped images:")
        for img in skipped_images:
            print(img)
    else:
        print("All images processed successfully.")


test_dataset = ImageDataset(
    root_dir='/data1/zhangjiansong/py_project/Hosptial_other_project/Fujian_2/Doc_WangYanli/data/OvarianCancerUS/External_test_0619',
    transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 指定分类后的图像存储路径
target_root_dir = './out_0619_Gen2text_images'
class_names = test_dataset.classes  # 使用 dataset 的 classes 属性
# 创建类别目录
for class_name in class_names:
    os.makedirs(os.path.join(target_root_dir, class_name), exist_ok=True)

test_model(model, test_loader)

# 检查目标文件夹中的图像数量
for class_name in class_names:
    class_dir = os.path.join(target_root_dir, class_name)
    num_images = len(os.listdir(class_dir))
    print(f'Number of images in {class_dir}: {num_images}')




