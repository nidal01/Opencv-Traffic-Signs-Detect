import cv2
from ultralytics import YOLO
import time

# YOLOv8 modelini yükleme
model = YOLO('yolov8n.pt')  # Daha hızlı bir model seçimi

# Tespit edilecek trafik işareti sınıfları
traffic_sign_classes = [
    'traffic light', 'stop', 'speed limit', 'no parking', 
    'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Keep right', 'Keep left',
    'Speed limit (30km/h)', 'Speed limit (80km/h)', 'Yield', 'No entry', 'Turn left', 'Turn right'
]

# Webcam'den video akışını açma
cap = cv2.VideoCapture(0)  # 0, genellikle varsayılan webcam'i belirtir

# Video özelliklerini ayarlama (isteğe bağlı)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_skip = 2  # Kaç karede bir nesne tespiti yapılacak
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Kare atlama mekanizması
    if frame_counter % frame_skip == 0:
        # YOLOv8 modelini kullanarak nesne tespiti
        results = model(frame)

        # Tespit edilen nesneleri işleme
        for result in results:  # Tespit edilen her sonuç için
            boxes = result.boxes  # Tespit edilen kutuları al

            for box in boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0]  # Koordinatları al
                confidence = box.conf[0]  # Güven skoru al
                class_id = box.cls[0]  # Sınıf ID'sini al
                label = model.names[int(class_id)]  # Sınıf adını al

                # Trafik işareti sınıfına göre işlem yapma
                if label in traffic_sign_classes:
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Sonuçları ekranda gösterme
    cv2.imshow('frame', frame)

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
