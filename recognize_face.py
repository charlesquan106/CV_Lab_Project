import zipfile
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO : 可以載入模型並進行處理


def load_models(path: str) -> tuple:
    """載入訓練好的模型

    Args:
        path (str): 模型存放路徑

    Returns:
        tuple: (重構後的人臉張量模型, 原始人臉張量模型, 權重模型)
    """
    reconstructed_model = torch.load(fr'{path}\reconstructed_model.pth')
    weights_model = torch.load(fr'{path}\weights_model.pth')

    return reconstructed_model, weights_model


def get_labels(data_path: str) -> list:
    """取得資料集的標籤

    Args:
        data_path (str): 資料集路徑

    Returns:
        list: 標籤存放的list
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

    # 建立用於存放標籤的list
    facelabel = []
    for key in faces.keys():
        # 以dataset.zip內的dataset之下的資料夾名稱作為標籤
        facelabel.append(key.split("/")[1])

    return facelabel


def real_time_detecting(reconstructed_model: torch.Tensor, weights_model: torch.Tensor, face_labels: list, device: str) -> None:
    """透過MTCNN進行人臉偵測, 並將偵測到的人臉進行resize後再與模型進行歐式距離計算並得到最相近的類別

    Args:
        reconstructed_model (torch.Tensor): 重構後的人臉張量模型
        weights_model (torch.Tensor): 權重模型
        face_labels (list): 類別標籤(內為數字)
        device (str): 要進行張量計算的裝置
    """
    # 建立人臉檢測物件
    mtcnn = MTCNN(keep_all=True, device=device)

    # 建立筆電攝影機物件
    cap = cv2.VideoCapture(0)

    # 建立標籤對應到的名稱的字典
    label_transport = {1: "Charles", 2: "Leo", 3: "Van"}

    # 讀取攝影機影像
    while True:
        # 讀取攝影機影像
        ret, frame = cap.read()

        # 判斷是否發生例外
        if not ret:
            print("無法讀取影像")
            break

        # 轉換RGB順序(因應opencv規定)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用人臉檢測模型進行人臉檢測
        boxes, _ = mtcnn.detect(frame_rgb, landmarks=False)

        # 判斷是否有檢測到人臉
        if boxes is not None:
            for box in boxes:
                # 取得預測的人臉的左上角及右下角座標
                x0, y0, x1, y1 = box.astype(int)

                # 判斷是否有視窗外的人臉(座標會<0)，有的話忽略
                if x0 <= 0 or y0 <= 0 or x1 <= 0 or y1 <= 0:
                    continue

                # 依照座標畫出物件框
                # cv2.rectangle(frame, (x0-15, y0-50), (x1+15, y1+20), (0, 255, 0), 2)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

                # 將物件框內的影像調整為符合資料集的大小用以進行判斷
                face = cv2.resize(frame[y0:y1, x0:x1], (184, 232))

                # 將物件框內的影像改為灰階
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # 將物件框內的影像轉換為張量
                query_face = torch.from_numpy(face.reshape(1, -1)).clone().to(device=device)

                # 計算權重向量
                query_weight = torch.mm(reconstructed_model.to(device), (query_face - reconstructed_model.to(device).mean()).T).to(device)

                # 計算權重模型與偵測到的人臉計算出的權重的歐式距離
                euclidean_distance = torch.norm(weights_model - query_weight, dim=0).to(device)

                # 取得讓歐式距離產生最小值的參數(標籤)
                best_match = torch.argmin(euclidean_distance).to(device)

                # 將標籤顯示於影像上
                cv2.putText(frame, str(label_transport[int(face_labels[best_match])]), (x0, y0-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 顯示影像
                cv2.imshow("Face Recognization", frame)

        # 判斷是否按下q鍵 -> 是: 關閉鏡頭視窗
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # 釋放鏡頭資源
    cap.release()

    # 關閉所有opencv相關的視窗
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 設定模型路徑
    models_path = 'models'
    # 設定資料集路徑
    dataset_path = 'dataset.zip'
    # 設定要訓練的裝置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    reconstructed_model, weights_model = load_models(models_path)

    face_labels = get_labels(dataset_path)

    real_time_detecting(reconstructed_model,
                        weights_model,
                        face_labels,
                        device)
