import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import pydicom
import numpy as np
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from matplotlib import pyplot as plt
import platform
import os



class PDATA():
    IMG_SIZE = 224
    BATCH_SIZE = 25
    

# 모델 불러오기 
xception_model = load_model("xception_model.h5")
labels = ['Normal', 'Pneumonia']

# 이미지 경로 불러오는 함수 설정
def IMG_PATH_GETTER():
    PDATA.IMG_PATH = filedialog.askopenfile(initialdir='path', title='SELECT IMG', filetypes=(('png files', '*.png'), ('DICOM files', '*.dcm')))
    PDATA.IMG_PATH = str(PDATA.IMG_PATH).split("'")[1]
    print(PDATA.IMG_PATH)

    label_target_img_path.configure(text = "Target image\n" + PDATA.IMG_PATH)
    # 이미지 전처리부 표시 
    button_img_preprocess.pack()
    label_img_preprocess.pack()


# 폐렴일 확률 구하기 DICOM일 경우
def load_dicom_image(dicom_path, target_size):
    try:
        dicom = pydicom.dcmread(dicom_path)
        img_array = dicom.pixel_array
        img_array = (img_array / img_array.max()) * 255.0
        img_array = np.stack((img_array,) * 3, axis=-1)
        img_array = resize(img_array, target_size, anti_aliasing=True)
        img_array = img_array.astype(np.uint8)
        return img_array
    except Exception as e:
        print(f"Error loading DICOM file: {e}")
        return None

# PNG일 경우
def load_png_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array.astype(np.uint8)
        return img_array
    except Exception as e:
        print(f"Error loading PNG file: {e}")
        return None

def IMG_PREPROCESS(label_img_preprocess,image_path):
    print(image_path)
    label_img_preprocess.configure(text="전처리 진행중...")

    PDATA.img_array = None
    if image_path.lower().endswith('.dcm'):
        PDATA.img_array = load_dicom_image(image_path, (PDATA.IMG_SIZE, PDATA.IMG_SIZE))
        label_img_preprocess.configure(text="전처리 완료")
    elif image_path.lower().endswith('.png'):
        PDATA.img_array = load_png_image(image_path, (PDATA.IMG_SIZE, PDATA.IMG_SIZE))
        label_img_preprocess.configure(text="전처리 완료")
    else:
        print("Unsupported file format. Please provide a PNG or DICOM file.")
        label_img_preprocess.configure(text="전처리 실패")
        return
    
    if PDATA.img_array is None:
        print("Failed to load image.")
        label_img_preprocess.configure(text="전처리 실패")
        return

    img_array_expanded = np.expand_dims(PDATA.img_array, axis=0)
    PDATA.img_array_preprocessed = xception_preprocess(img_array_expanded)

    button_start_analyze.pack()

def get_grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def ANALYZE():
    try:
        button_start_analyze.configure(text="분석중...")

        PDATA.prediction = xception_model.predict(PDATA.img_array_preprocessed)
        PDATA.predicted_class = int(PDATA.prediction > 0.5)  # 0.5보다 클 때: pneumonia, 0.5보다 작을 때: normal 반환
        PDATA.predicted_label = labels[PDATA.predicted_class]
        PDATA.pneumonia_probability = PDATA.prediction[0][0] * 100  # 퍼센트 반환

        PDATA.heatmap = get_grad_cam(xception_model, PDATA.img_array_preprocessed, 'block14_sepconv2_act')  # Xception 모델의 마지막 합성곱층 이름
        PDATA.heatmap = resize(PDATA.heatmap, (PDATA.IMG_SIZE, PDATA.IMG_SIZE))

        # 위험등급 나누기
        if PDATA.pneumonia_probability > 75:
            PDATA.risk_level = "매우 위험"
            PDATA.color = 'darkred'
        elif PDATA.pneumonia_probability >= 70:
            PDATA.risk_level = "상"
            PDATA.color = 'red'
        elif PDATA.pneumonia_probability >= 50:
            PDATA.risk_level = "중"
            PDATA.color = 'orange'
        else:
            PDATA.risk_level = "하"
            PDATA.color = 'green'

        button_start_analyze.configure(text="결과 보기", command=result_window_opener)
    except Exception as e:
        print(f"Error: {e}")

        button_start_analyze.configure(text="실패")
        return None


def result_window_opener():



    osType = str(platform.system())

    if osType == "Windows":
        plt.rc('font', family="Malgun Gothic")
    elif osType == "Darwin":
        plt.rc('font', family="Apple Gothic")
    else:
        plt.rc('font', family="Nanum Gothic")
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(PDATA.img_array, interpolation='nearest')
    plt.title(f"예측: {PDATA.predicted_label}\n폐렴 확률: {PDATA.pneumonia_probability:.2f}%")
    plt.subplot(1, 2, 2)
    plt.imshow(PDATA.img_array, interpolation='nearest')
    plt.imshow(PDATA.heatmap, cmap='jet', alpha=0.5)  # 투명도를 조정하여 원본 이미지 위에 히트맵을 보여줌
    plt.title("Grad-CAM")
    plt.axis('off')

    plt.figure(figsize=(10, 10))
    plt.imshow(PDATA.img_array, interpolation='nearest')
    plt.title(f"예측: {PDATA.predicted_label}\n폐렴 확률: {PDATA.pneumonia_probability:.2f}%\n위험등급: {PDATA.risk_level}", fontsize=16, color=PDATA.color)
    plt.axis('off')
    plt.show()
    
def RESTART():
    root.destroy()
    path = str(os.path.realpath(__file__)).replace(" ", "\u0020")
    os.system()


# 창 생성
root = tk.Tk()

# 창 이름 설정
root.title("Pneumonia detector")

# 창 크기 설정 및 크기 고정
root.geometry("500x900")
root.resizable(False, False)


# 프로그램 제목 표시 
label_title = tk.Label(root, text="AI 폐렴 진단 프로그램")
label_title.pack()

# GUI 구성요소 정의 
label_img_input = tk.Label(root, text="아래 버튼을 통해 이미지 경로를 입력해주세요.")
button_img_input = tk.Button(root, text="경로 설정.", command=IMG_PATH_GETTER)
label_target_img_path = tk.Label(text="No image selected", width=400)
button_restart = tk.Button(text="재시작", command=RESTART)

# 조건부 활성화 요소 
label_img_preprocess = tk.Label(root, text="전처리 진행 안됨")
button_img_preprocess = tk.Button(root, text="전처리 진행", command=lambda : IMG_PREPROCESS(label_img_preprocess, PDATA.IMG_PATH))
button_start_analyze = tk.Button(text="분석", command=ANALYZE)


# 초반 구성 요소 활성화 
button_restart.pack()
label_img_input.pack()
button_img_input.pack()
label_target_img_path.pack()

# 창 유지
root.mainloop()