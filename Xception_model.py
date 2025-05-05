import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pydicom
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

plt.rcParams['figure.figsize'] = (20, 20) # 그래프 크기 설정

img_size = 224 # 알맞게 이미지 크기 설정
batch_size = 25 # 배치 크기 설정

# 학습 이미지 데이터 설정
train_gen = ImageDataGenerator(
    preprocessing_function=xception_preprocess,
    rotation_range=60, # 이미지 회전 범위(±60도)
    width_shift_range=0.2, # 이미지 가로 방향 이동범위 설정(±20%)
    height_shift_range=0.2, # 이미지 세로 방향 이동범위 설정
    horizontal_flip=True, # 이미지 수평 뒤집기
    zoom_range=0.2,  # 이미지 확대, 축소(±20%)
    shear_range=0.2 # 이미지 전단 변환 범위 설정(±20%)
)

# 테스트 및 검증 이미지 데이터 생성기 생성
test_gen = ImageDataGenerator(preprocessing_function=xception_preprocess)
val_gen = ImageDataGenerator(preprocessing_function=xception_preprocess)

# 훈련 데이터 로드 및 생성기 설정
train_data = train_gen.flow_from_directory(
    'C:/페렴이미지/archive/chest_xray/train', # train 할 폐렴이미지 경로
    target_size=(img_size, img_size),
    batch_size=batch_size,  
    shuffle=True, 
    class_mode='binary',
    color_mode='rgb' 
)

test_data = test_gen.flow_from_directory(
    'C:/페렴이미지/archive/chest_xray/test', #test 할 폐렴 이미지 경로
    target_size=(img_size, img_size), 
    batch_size=batch_size, 
    shuffle=False, 
    class_mode='binary', 
    color_mode='rgb' 
)

val_data = val_gen.flow_from_directory(
    'C:/페렴이미지/archive/chest_xray/val',  #validation 할 이미지 경로
    target_size=(img_size, img_size),
    batch_size=batch_size,  
    shuffle=False, 
    class_mode='binary', 
    color_mode='rgb'
)

print(f"Train images_dataset: {train_data.samples}") # train 데이터 개수
print(f"Validation images_dataset: {val_data.samples}") # validation 데이터 개수
print(f"Test images_dataset: {test_data.samples}") # test 데이터 개수

labels = ['Normal', 'Pneumonia'] # 폐렴이냐? 정상이냐? 라벨링
samples = train_data.__next__() # 다음 배치의 데이터 가져오기

images = samples[0] # 이미지 데이터
target = samples[1] # labels 데이터

# 이미지 데이터의 픽셀 값을 [0, 255] 범위로 정규화하여 시각화하기 쉽도록 조정
images = images - images.min()
images = images / images.max()
images = images * 255

# 이미지 시각화
plt.figure(figsize=(20, 20))
for i in range(15):
    plt.subplot(3, 5, i + 1) # 3행 5열의 그리드에서 i+1번째 subplot 선택
    plt.subplots_adjust(hspace=0.5, wspace=0.5) # subplot 간의 간격 조정
    plt.imshow(images[i].astype('uint8'), interpolation='nearest') # 이미지 플로팅
    class_index = int(target[i]) # 현재 이미지의 클래스 인덱스
    plt.title(f"Class: {labels[class_index]}", fontsize=16) # 클래스 이름으로 타이틀 설정
    plt.axis('off') # 축 숨김
plt.show()

# Xception 모델 정의 및 컴파일
xception_base = Xception(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3)) # ImageNet 데이터셋에서 사전 훈련된 가중치를 사용
x = xception_base.output
x = GlobalAveragePooling2D()(x) # 각 채널의 평균값을 계산하여 공간 차원을 제거(하나의 숫자로 줄임)
x = Dropout(0.2)(x) # 학습 과정에서 일부 뉴런을 랜덤하게 비활성화하여 과적합 방지
output = Dense(1, activation='sigmoid')(x) # 이진 분류이기 때문에 sigmoid 함수 사용

xception_model = Model(inputs=xception_base.input, outputs=output)
xception_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['acc']) 
# 경사하강법(optimizer은 Adam 설정), 손실함수는 이진분류이므로 binary_crossentropy 사용

# 모델 훈련
xception_history = xception_model.fit(train_data, validation_data=val_data, epochs=10) # 10번 반복

# # 모델 저장
xception_model.save('xception_model.h5')

# 모델 불러오기
xception_model = load_model('xception_model.h5') 

# 모델 구조확인(gred-cam 부분에서 마지막 층의 이름을 알기위해서)
xception_model.summary() 

# 평가 함수
def evaluate_model(model, data, labels):
    true_class = data.classes # 데이터 클래스 가져오기
    predictions = model.predict(data) # 모델 예측
    predictions_class = (predictions > 0.5).astype(int).flatten() # 이진 분류-> 확률이 0.5보다 클 때: 1, 작을 때: 0 반환
    print(classification_report(true_class, predictions_class, target_names=labels))
    print(confusion_matrix(true_class, predictions_class)) # 혼동 행렬 출력 
    
    # 정확도 계산
    accuracy = np.sum(predictions_class == true_class) / len(true_class) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Xception 평가
print("Validation 데이터 평가:")
evaluate_model(xception_model, val_data, labels)
print("Test 데이터 평가:")
evaluate_model(xception_model, test_data, labels)

# test_data에서 랜덤 10개 뽑은 것들 라벨링
random_samples = random.sample(range(len(test_data)), 10) # test data 무작위 10개 선택
plt.figure(figsize=(20, 10)) # 크기 설정
plt.rc('font', family='Malgun Gothic') # 한글 폰트  는 맑은 고딕 설정
# 무작위 test_data 라벨링
for idx, sample_idx in enumerate(random_samples):
    batch_images, batch_labels = test_data[sample_idx] # 무작위 이미지 및 라벨 가져오기
    predictions = xception_model.predict(batch_images) # 모델을 사용하여 예측 수행
    predictions_class = (predictions > 0.5).astype(int).flatten() # 이진 분류로 진행
    
    for i in range(len(batch_images)):
        image = batch_images[i] # 현재 이미지 가져오기
        image = (image - image.min()) / (image.max() - image.min()) * 255.0 # 이미지 범위 조정
        image = image.astype('uint8') # 이미지를 정수 형태로 변환
        
        # 이미지 예측 결과 출력
        plt.subplot(2, 5, idx + 1)
        plt.imshow(image)
        plt.title(f"예측: {labels[predictions_class[i]]}\n실제값: {labels[int(batch_labels[i])]}")
        plt.axis('off')
plt.show()

# DICOM 파일 로드 함수
def load_dicom_image(dicom_path, target_size):
    try:
        dicom = pydicom.dcmread(dicom_path) # DICOM 파일 읽기
        img_array = dicom.pixel_array # 이미지 배열로 변환
        img_array = (img_array / img_array.max()) * 255.0 # 이미지 범위 조정
        img_array = np.stack((img_array,) * 3, axis=-1)
        img_array = resize(img_array, target_size, anti_aliasing=True) # 이미지 크기 조정
        img_array = img_array.astype(np.uint8) # 정수형 변환
        return img_array
    except Exception as e:
        print(f"DICOM 파일 로드 중 오류 발생: {e}")
        return None

# PNG 파일 로드 함수
def load_png_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size) # PNG 파일 읽기
        img_array = img_to_array(img) # 이미지 배열로 변환
        img_array = img_array.astype(np.uint8) # 정수형 변환
        return img_array
    except Exception as e:
        print(f"PNG 파일 로드 중 오류 발생: {e}")
        return None

# Grad-CAM 함수
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

# 이미지 처리 및 예측 함수
def image_print_decision(image_path, model, labels):
    img_array = None
    if image_path.lower().endswith('.dcm'):
        img_array = load_dicom_image(image_path, (img_size, img_size))
    elif image_path.lower().endswith('.png'):
        img_array = load_png_image(image_path, (img_size, img_size))
    else:
        print("지원되지 않는 파일 형식입니다. PNG 또는 DICOM 파일을 제공해주세요.")
        return
    
    if img_array is None:
        print("이미지를 로드하는 데 실패했습니다.")
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

# 예제 사용
image_path_png = 'C:/test1_data/chest AP 폐렴.png' # png 형태의 파일 경로 입력
image_path_dicom = 'C:/test1_data/MEE.dcm' # dicom(dcm 형태)의 파일 경로 입력

# PNG 파일 예측
image_print_decision(image_path_png, xception_model, labels)

# DICOM 파일 예측
image_print_decision(image_path_dicom, xception_model, labels)