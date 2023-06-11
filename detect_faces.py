import os
import time
import cv2
import torch
from facenet_pytorch import MTCNN
import zipfile

# 每個人攝影機拍攝的照片數量
num_of_pic = 1000

# 需要拍攝的人數
num_of_person = 3

print("num_of_pic: ", num_of_pic)
print("num_of_person: ", num_of_person)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture(0)

# for pic_rename_num in range(num_of_pic)
pic_rename_num = 1
# while True:
for capture_person_num in range(num_of_person):
    
    output_path = 'dataset\\' + str(capture_person_num+1)
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)
        pass

    for pic_rename_num in range(num_of_pic):

        # 逐一幀率讀取影片
        ret, frame = cap.read()

        # 檢驗是否成功讀取影片幀
        if not ret:
            print("【Error】無法讀取影片幀", flush=True)
            break

        # 將幀轉換成RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 檢測人臉的位置
        boxes, probs = mtcnn.detect(frame_rgb, landmarks=False)
        print("【Log File】 框選範圍是人臉的機率", probs, flush=True)

        # 框選檢測到的人臉
        if boxes is not None and probs.all()>0.9:
            for box in boxes:
                x0, y0, x1, y1 = box.astype(int)

                if x0 <= 0 or y0 <= 0 or x1 <= 0 or y1 <= 0:
                    continue
                # print(x0, y0, x1, y1)
                cv2.rectangle(frame, (x0-15, y0-50), (x1+15, y1+20), (0, 255, 0), 2)
                # cv2.imshow("grap face", frame[x:w, y:h])
                cv2.imwrite(fr"dataset\{capture_person_num+1}\{pic_rename_num+1}.jpg", frame[y0:y1, x0:x1])
                print("【Log File】 框選人臉影像的儲存路徑", fr"{capture_person_num+1}\{pic_rename_num+1}.jpg", flush=True)
                #pic_rename_num+=1

        # 顯示影像
        cv2.imshow("Face Detection", frame)
        # Enter 'q', quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    print("Next Person Ready, wait 6 seconds", flush=True)

    if (capture_person_num+1)!= num_of_person:
        time.sleep(6)
    else:
        print("【Log File】資料集建構完成！！！", flush=True)

# 釋放資源，關閉視窗
cap.release()
cv2.destroyAllWindows()

input_path = 'dataset\\'

if os.path.isdir(input_path):
    folders_list = os.listdir(input_path)
print("【Log File】 folders_list : ", folders_list, flush=True)   


for folder in folders_list:
    if str(folder).isdigit():
        folder_path = fr"dataset\{folder}"
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            print("【Log File】【For Resize_images】folders_list: ", file_path, flush=True)
            if os.path.isfile(file_path):    
                img = cv2.imread(file_path)

                new_width = 184
                new_height = 232

                # 保持長寬比縮放
                # new_height = int(new_width * img.shape[0] / img.shape[1])

                resized_img = cv2.resize(img, (new_width, new_height))
                cv2.imwrite(file_path, resized_img)

# 指定被壓縮的檔案List(folders_list)
# 指定壓縮後檔案的路徑和名稱
output_path = f'dataset.zip'

# Creat Zip Document
with zipfile.ZipFile(output_path, 'w') as zipf:
    # 遍歷所有資料夾並加入Zip內
    for folder in folders_list:
        if str(folder).isdigit():
            folder_path = fr"dataset\{folder}"
            file_list = os.listdir(folder_path)
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                print("【Log File】【For Zip】 file_path: ", file_path, flush=True)
                zipf.write(file_path, arcname=os.path.join(folder_path, file_name))
