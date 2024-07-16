import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from base64 import b64encode
import logging

# Desativar mensagens de log de nível INFO do YOLO
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Caminho do modelo treinado
model_path = "TrainYoloV8CustomDataset/runs/detect/train/weights/last.pt"
PRE_TRAINED_MODEL = YOLO(model_path)

# Labels do sistema em português
LABELS = ['com_oculos_protetores', 'sem_oculos_protetores', 'com_capacete', 'sem_capacete']

# Dicionário para armazenar contagens acumuladas
accumulated_counts = {'com_oculos_protetores': 0, 'sem_oculos_protetores': 0, 'com_capacete': 0, 'sem_capacete': 0}
track_id_history = {'com_oculos_protetores': set(), 'sem_oculos_protetores': set(), 'com_capacete': set(), 'sem_capacete': set()}

# Cores para cada tipo de detecção
COLORS = {
    'com_oculos_protetores': (0, 255, 0),  # Verde
    'sem_oculos_protetores': (0, 0, 255),  # Vermelho
    'com_capacete': (255, 0, 0),  # Azul
    'sem_capacete': (0, 255, 255)  # Amarelo
}

def count_labels(track_id, class_id):
    global track_id_history
    label_counts = {'com_oculos_protetores': 0, 'sem_oculos_protetores': 0, 'com_capacete': 0, 'sem_capacete': 0}

    class_labels = ['com_oculos_protetores', 'sem_oculos_protetores', 'com_capacete', 'sem_capacete']
    class_label = class_labels[int(class_id)]

    if track_id not in track_id_history[class_label]:
        track_id_history[class_label].add(track_id)
        label_counts[class_label] += 1

    return label_counts

def update_accumulated_counts(label_counts):
    global accumulated_counts
    for label, count in label_counts.items():
        accumulated_counts[label] += count

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a webcam")
        return

    # Configura a resolução para a máxima possível
    screen_width = 1920
    screen_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

    cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Realiza a detecção na imagem capturada
        detections = PRE_TRAINED_MODEL.track(np.array(frame), persist=True)

        for det in detections:
            for d in det.boxes.data.tolist():
                if len(d) < 7:
                    continue
                
                x1, y1, x2, y2, track_id, score, class_id = d[:7]
                
                # Verifica se o score é maior que 0.7
                if score < 0.75:
                    continue
                
                label_counts = count_labels(track_id, class_id)
                # Atualiza as contagens acumuladas
                update_accumulated_counts(label_counts)
                
                class_label = LABELS[int(class_id)]
                color = COLORS[class_label]  # Cor específica para o tipo de detecção
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{class_label} ID: {int(track_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Atualiza a exibição das contagens acumuladas
        label_text = '\n'.join([f'{label}: {count}' for label, count in accumulated_counts.items()])
        
        # Dimensões do retângulo de fundo
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        rect_w = max(text_size[0] + 20, 150)
        rect_h = (text_size[1] + 10) * len(accumulated_counts) + 20
        cv2.rectangle(frame, (5, 5), (rect_w, rect_h), (50, 50, 50), -1)  # Retângulo cinza escuro

        y0, dy = 20, 20
        for i, line in enumerate(label_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # Texto branco

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()