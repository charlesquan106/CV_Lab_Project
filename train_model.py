import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def load_data(data_path: str) -> dict:
    """讀取zip壓縮的資料集

    Args:
        data_path (str): 資料集所在路徑

    Returns:
        dict: 處理過後的影像字典
    """
    # 建立用於存放影像資料的字典
    faces = {}
    with zipfile.ZipFile(data_path) as facezip:
        for filename in facezip.namelist():
            # 判斷是否為影像資料
            if not filename.endswith(".jpg"):
                continue
            with facezip.open(filename) as image:
                # 將影像名稱作為key, 影像的陣列作為value存成dict型別
                faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return faces


def preprocess(datas: dict) -> tuple:
    """顯示部分訓練照片、訓練資料基本資訊(大小、類別、照片數量), 並且將訓練照片拉平後存入facematrix用於SVD分解

    Args:
        datas (dict): 從load_data()讀取進來並轉成字典形式的資料

    Returns:
        tuple: (所有照片拉平過後的矩陣, 每張照片的原始大小)
    """
    # 顯示最後16張訓練照片
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(16, 24))
    fig.suptitle('Show the sample images', fontsize=24)
    faceimages = list(datas.values())[-16:]
    for i in range(16):
        axes[i % 4][i//4].imshow(faceimages[i], cmap="gray")
    plt.show()

    # 取得影像大小
    faceshape = list(datas.values())[0].shape
    print("Face image shape:", faceshape)

    # 取得影像的類別長度及數量長度
    classes = set(filename.split("/")[1] for filename in datas.keys())
    print("Number of classes:", len(classes))
    print("Number of images:", len(datas))

    # 用於存入影像拉平後的向量
    facematrix = []
    for data in datas.values():
        facematrix.append(data.flatten())

    # 將矩陣轉成pytorch張量
    facematrix = torch.from_numpy(np.array(facematrix)).clone().to(torch.float32)

    return facematrix, faceshape


def train_models(data_matrix: torch.Tensor, data_shape: tuple, device: str, k_values: int) -> tuple:
    """將人臉矩陣使用SVD分解, 並取出前k個值來重構。此外也畫出重構後的eigenfaces

    Args:
        data_matrix (torch.Tensor): 放有人臉矩陣的張量
        data_shape (tuple): 原始影像大小
        device (str): 要用於計算矩陣乘法的裝置(cpu or cuda)
        k_values (int): 要近似的前k個值

    Returns:
        tuple: (重構後的人臉矩陣, 原始人臉矩陣)
    """
    # 將人臉矩陣存入指定裝置
    facematrix = data_matrix.to(device=device)

    # 設定要取得的k值
    k = k_values

    # 進行SVD分解
    U, S, V = torch.svd_lowrank(facematrix, q=k)
    U_k = U[:, :k]  # 取U的前k列
    S_k = S[:k]    # 取S的前k个元素
    V_k = V[:, :k]  # 取V的前k列

    # 重構原始數據
    X_reconstructed = torch.matmul(U_k, torch.diag(S_k)).matmul(V_k.T).cpu()

    # 顯示後16張eigenfaces
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(16, 24))
    fig.suptitle('Show the eigenfaces', fontsize=24)
    for i in range(16):
        axes[i % 4][i//4].imshow(X_reconstructed[len(X_reconstructed)-1-i].reshape(data_shape), cmap="gray")
    print("Showing the eigenfaces")
    plt.show()

    # 將權重生成為 KxN 矩陣，其中 K 是特徵臉的數量，N 是樣本的數量
    weights = torch.mm(X_reconstructed.to(device), (facematrix.to(device) - X_reconstructed.to(device).mean()).T)
    print("Shape of the weight matrix:", weights.shape)

    return X_reconstructed, weights


def save_models(path: str, reconstructed_data: torch.Tensor, weights: torch.Tensor) -> None:
    """儲存訓練好的模型

    Args:
        path (str): 要存放模型的路徑
        reconstructed_data (torch.Tensor): 重建過後的人臉張量模型
        facematrix (torch.Tensor): 原始人臉張量模型
    """
    if not Path.exists(Path.cwd() / './models'):
        Path.mkdir(Path.cwd() / './models')
    torch.save(reconstructed_data, fr'{path}\reconstructed_model.pth')
    torch.save(weights, fr'{path}\weights_model.pth')


if __name__ == "__main__":
    # 設定要訓練的裝置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 設定資料集位置及名稱
    dataset_path = "dataset.zip"
    # 設定要存放模型的路徑
    save_models_path = "models"

    faces_data = load_data(data_path=dataset_path)

    facematrix, faceshape = preprocess(datas=faces_data)

    reconstructed_data, weights = train_models(data_matrix=facematrix,
                                               data_shape=faceshape,
                                               device=device,
                                               k_values=135)

    save_models(path=save_models_path,
                reconstructed_data=reconstructed_data,
                weights=weights)
