import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pydicom
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
import tkinter as tk
from tkinter import filedialog, messagebox

plt.rcParams['figure.figsize'] = (20, 20)
plt.rc('font', family='Malgun Gothic')

img_size = 224
labels = ['Normal', 'Pneumonia']

# Xception 모델 불러오기
xception_model = load_model('xception_model.h5')

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

def load_png_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array.astype(np.uint8)
        return img_array
    except Exception as e:
        print(f"Error loading PNG file: {e}")
        return None
    
def load_jpeg_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array.astype(np.uint8)
        return img_array
    except Exception as e:
        print(f"Error loading JPEG file: {e}")
        return None

def get_grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def image_print_decision(image_path, model, labels):
    img_array = None
    if image_path.lower().endswith('.dcm'):
        img_array = load_dicom_image(image_path, (img_size, img_size))
    elif image_path.lower().endswith('.png'):
        img_array = load_png_image(image_path, (img_size, img_size))
    elif image_path.lower().endswith('.jpeg') or image_path.lower().endswith('.jpg'):
        img_array = load_jpeg_image(image_path, (img_size, img_size))
    else:
        messagebox.showerror("Error", "Unsupported file format. Please provide a PNG, JPEG, or DICOM file.")
        return
    
    if img_array is None:
        messagebox.showerror("Error", "Failed to load image.")
        return

    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = xception_preprocess(img_array_expanded)
    
    prediction = model.predict(img_array_preprocessed)
    predicted_class = int(prediction > 0.5)
    predicted_label = labels[predicted_class]
    pneumonia_probability = prediction[0][0] * 100
    
    heatmap = get_grad_cam(model, img_array_preprocessed, 'block14_sepconv2_act')
    heatmap = resize(heatmap, (img_size, img_size))

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array, interpolation='nearest')
    plt.title(f"예측: {predicted_label}\n폐렴 확률: {pneumonia_probability:.2f}%")
    plt.subplot(1, 2, 2)
    plt.imshow(img_array, interpolation='nearest')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis('off')

    if pneumonia_probability > 75:
        risk_level = "매우 위험"
        color = 'darkred'
    elif pneumonia_probability <= 75 and pneumonia_probability > 50:
        risk_level = "상"
        color = 'red'
    elif pneumonia_probability <= 50 and pneumonia_probability > 25:
        risk_level = "중"
        color = 'orange'
    else:
        risk_level = "하"
        color = 'green'
        
    plt.figure(figsize=(10, 10))
    plt.imshow(img_array, interpolation='nearest')
    plt.title(f"예측: {predicted_label}\n폐렴 확률: {pneumonia_probability:.2f}%\n위험등급: {risk_level}", fontsize=16, color=color)
    plt.axis('off')
    plt.show()

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_print_decision(file_path, xception_model, labels)

# GUI 설정
root = tk.Tk()
root.title("AI 폐렴 진단 프로그램")
root.geometry("300x150")

btn_open = tk.Button(root, text="클릭 후 이미지를 선택해주세요.", command=open_file)
btn_open.pack(pady=20)

root.mainloop()