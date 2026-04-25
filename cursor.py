import cv2
import os
import sys  # 🔥 IMPORTANTE
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- função de saída ---
def sair(cap):
    
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()  # 🔥 mata o processo

# --- Conexões da mão ---
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


class DetectorMaos:
    def __init__(self, max_maos=1):
        model_path = os.path.join(os.getcwd(), "hand_landmarker.task")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_maos
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.resultado = None

    def encontrar_maos(self, imagem, desenho=True):
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imagem_rgb
        )

        self.resultado = self.detector.detect(mp_image)

        if self.resultado.hand_landmarks:
            for hand in self.resultado.hand_landmarks:
                h, w, _ = imagem.shape
                pontos = []

                for ponto in hand:
                    x, y = int(ponto.x * w), int(ponto.y * h)
                    pontos.append((x, y))
                    if desenho:
                        cv2.circle(imagem, (x, y), 4, (0, 255, 0), -1)

                if desenho:
                    for conexao in HAND_CONNECTIONS:
                        p1 = pontos[conexao[0]]
                        p2 = pontos[conexao[1]]
                        cv2.line(imagem, p1, p2, (255, 255, 255), 2)

        return imagem

    def encontrar_pontos(self, imagem, mao_num=0, desenho=True, cor=(255, 0, 255), raio=7, ponto_detectado=0):
        lista_pontos = []

        if self.resultado and self.resultado.hand_landmarks:
            if mao_num < len(self.resultado.hand_landmarks):
                mao = self.resultado.hand_landmarks[mao_num]

                for id, ponto in enumerate(mao):
                    if id == ponto_detectado:
                        altura, largura, _ = imagem.shape
                        centro_x = int(ponto.x * largura)
                        centro_y = int(ponto.y * altura)

                        lista_pontos.append([id, centro_x, centro_y])

                        if desenho:
                            cv2.circle(imagem, (centro_x, centro_y), raio, cor, cv2.FILLED)

        return lista_pontos


def main():
    cap = cv2.VideoCapture(0)
    detector = DetectorMaos()

    while True:
        ret, imagem = cap.read()
        if not ret:
            sair(cap)

        imagem = cv2.flip(imagem, 1)
        imagem = detector.encontrar_maos(imagem)

        # 👉 dedo indicador
        lista_pontos = detector.encontrar_pontos(imagem, ponto_detectado=8)

        if lista_pontos:
            _, x, y = lista_pontos[0]
            h, w, _ = imagem.shape

            screen_w, screen_h = pyautogui.size()

            mouse_x = int(x * screen_w / w)
            mouse_y = int(y * screen_h / h)

            pyautogui.moveTo(mouse_x, mouse_y)

        # clique
        p1 = detector.encontrar_pontos(imagem, ponto_detectado=8)
        p2 = detector.encontrar_pontos(imagem, ponto_detectado=4)

        if p1 and p2:
            x1, y1 = p1[0][1], p1[0][2]
            x2, y2 = p2[0][1], p2[0][2]

            distancia = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

            if distancia < 30:
                pyautogui.click()

        cv2.imshow('mao_cursor', imagem)

        key = cv2.waitKey(1) & 0xFF

        

        # ⌨️ ESC fecha
        if key == 27:
            print("Fechado com ESC")
            sair(cap)

        # ❌ botão X fecha
        if cv2.getWindowProperty('mao_cursor', cv2.WND_PROP_VISIBLE) < 1:
            
            sair(cap)


if __name__ == '__main__':
    main()
