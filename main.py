import cv2
import mediapipe as mp
import random
import numpy as np

class CapturaMovimento:
    def __init__(self):
        # Configurações iniciais da câmera e MediaPipe
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configurações do jogo
        self.target_numbers = self._gerar_numeros_aleatorios()
        self.indice_atual = 0
        self.contagem_frames = 0
        self.FRAMES_CONFIRMACAO = 15
        self.pontuacao = 0
        
        # Configurações da tela de desenho
        self.canvas = None
        self.cor_selecionada = None
        self.ultimo_ponto = None
        self.cores = {
            'borracha': (0, 0, 0),
            'laranja': (0, 127, 255)
        }
        self.caixas_cores = {
            'borracha': [(580, 50), (630, 100)],
            'laranja': [(580, 150), (630, 200)]
        }
        self.timer_clique = 0
        self.tempo_ultimo_clique = 0
        self.contagem_esquerda = 0
        self.contagem_direita = 0

    def _verificar_coracao(self, landmarks_mao, lado_mao):
        """Verifica se a mão está fazendo o formato de coração"""
        # Pontos dos dedos
        ponta_indicador = landmarks_mao.landmark[8]
        ponta_polegar = landmarks_mao.landmark[4]
        ponta_medio = landmarks_mao.landmark[12]
        
        # Bases dos dedos
        base_indicador = landmarks_mao.landmark[5]
        base_polegar = landmarks_mao.landmark[2]
        base_medio = landmarks_mao.landmark[9]
        
        # Verifica se indicador e médio estão juntos e levantados
        dedos_juntos = abs(ponta_indicador.x - ponta_medio.x) < 0.05
        dedos_levantados = (ponta_indicador.y < base_indicador.y and 
                        ponta_medio.y < base_medio.y)
        
        # Verifica posição do polegar
        if lado_mao == 'Right':
            polegar_posicao = ponta_polegar.x < base_polegar.x
        else:
            polegar_posicao = ponta_polegar.x > base_polegar.x
        
        # Verifica se outros dedos estão fechados
        outros_dedos_fechados = all(
            landmarks_mao.landmark[ponta].y > landmarks_mao.landmark[base].y
            for ponta, base in zip([16, 20], [13, 17])
        )
        
        return dedos_juntos and dedos_levantados and polegar_posicao and outros_dedos_fechados

    def _mostrar_mensagem_coracao(self, frame):
        """Mostra uma mensagem quando detectar o formato de coração"""
        self._desenhar_texto_com_fundo(
            frame,
            "Te amo",
            (frame.shape[1]//2 - 150, frame.shape[0]//4 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),  # Vermelho
            (0, 0, 0)     # Fundo preto
        )


    def _verificar_sinal_L(self, landmarks_mao, lado_mao):
        """Verifica se a mão está fazendo o sinal de L"""
        # Pontos dos dedos
        ponta_indicador = landmarks_mao.landmark[8]
        base_indicador = landmarks_mao.landmark[5]
        ponta_polegar = landmarks_mao.landmark[4]
        base_polegar = landmarks_mao.landmark[2]
        
        # Verifica se o indicador está para cima
        indicador_cima = ponta_indicador.y < base_indicador.y
        
        # Verifica se o polegar está para o lado
        if lado_mao == 'Right':
            polegar_lado = ponta_polegar.x < base_polegar.x
        else:
            polegar_lado = ponta_polegar.x > base_polegar.x
        
        # Verifica se os outros dedos estão fechados
        outros_dedos_fechados = all(
            landmarks_mao.landmark[ponta].y > landmarks_mao.landmark[base].y
            for ponta, base in zip([12, 16, 20], [9, 13, 17])
        )
        
        return indicador_cima and polegar_lado and outros_dedos_fechados

    def _mostrar_mensagem_L(self, frame):
        """Mostra a mensagem 'FAZ O L!' na tela"""
        self._desenhar_texto_com_fundo(
            frame,
            "FAZ O L!",
            (frame.shape[1]//2 - 150, frame.shape[0]//4 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 255),  # Amarelo
            (0, 0, 0)       # Fundo preto
        )

    def _gerar_numeros_aleatorios(self):
        return [random.randint(1, 10) for _ in range(5)]

    def _inicializar_canvas(self, frame):
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def _desenhar_caixas_cores(self, frame):
        for nome_cor, coords in self.caixas_cores.items():
            cv2.rectangle(frame, coords[0], coords[1], self.cores[nome_cor], -1)
            if self.cor_selecionada == self.cores[nome_cor]:
                cv2.rectangle(frame, 
                            (coords[0][0]-2, coords[0][1]-2), 
                            (coords[1][0]+2, coords[1][1]+2), 
                            (255, 255, 255), 2)

    def _verificar_selecao_cor(self, ponta_dedo, frame):
        x, y = int(ponta_dedo.x * frame.shape[1]), int(ponta_dedo.y * frame.shape[0])
        
        for nome_cor, coords in self.caixas_cores.items():
            if (coords[0][0] <= x <= coords[1][0] and coords[0][1] <= y <= coords[1][1]):
                tempo_atual = cv2.getTickCount() / cv2.getTickFrequency()
                if tempo_atual - self.tempo_ultimo_clique < 0.3:
                    self.cor_selecionada = self.cores[nome_cor]
                    self.timer_clique = 0
                self.tempo_ultimo_clique = tempo_atual

    def _desenhar(self, frame, landmarks_mao):
        if self.cor_selecionada is None:
            return
        
        ponta_indicador = landmarks_mao.landmark[8]
        base_indicador = landmarks_mao.landmark[5]
        ponta_polegar = landmarks_mao.landmark[4]
        base_polegar = landmarks_mao.landmark[2]
        
        indicador_levantado = ponta_indicador.y < base_indicador.y
        polegar_levantado = ponta_polegar.y < base_polegar.y
        
        if indicador_levantado and polegar_levantado:
            x = int(ponta_indicador.x * frame.shape[1])
            y = int(ponta_indicador.y * frame.shape[0])
            
            if self.ultimo_ponto is not None:
                if self.cor_selecionada == self.cores['borracha']:
                    cv2.circle(self.canvas, (x, y), 20, (0,0,0), -1)
                    cv2.line(self.canvas, self.ultimo_ponto, (x, y), (0,0,0), 40)
                else:
                    cv2.line(self.canvas, self.ultimo_ponto, (x, y), self.cor_selecionada, 2)
            self.ultimo_ponto = (x, y)
        else:
            self.ultimo_ponto = None

    def _contar_dedos(self, landmarks_mao, lado_mao):
        """Conta os dedos levantados com maior precisão"""
        pontas_dedos = [8, 12, 16, 20]  # índice, médio, anelar, mindinho
        bases_dedos = [6, 10, 14, 18]   # bases correspondentes
        contagem = 0
        
        # Lógica específica para o polegar baseada no lado da mão
        polegar_ponta = landmarks_mao.landmark[4]
        polegar_base = landmarks_mao.landmark[2]
        
        # Ajuste da verificação do polegar
        if lado_mao == 'Right':
            if polegar_ponta.x < polegar_base.x and polegar_ponta.y < polegar_base.y:
                contagem += 1
        else:  # Left hand
            if polegar_ponta.x > polegar_base.x and polegar_ponta.y < polegar_base.y:
                contagem += 1
        
        # Verificação mais precisa dos outros dedos
        for ponta, base in zip(pontas_dedos, bases_dedos):
            if landmarks_mao.landmark[ponta].y < landmarks_mao.landmark[base].y:
                contagem += 1
                
        return contagem

    def _mostrar_contagem(self, frame, contagem, lado):
        """Mostra a contagem atual de dedos e verifica sequência de números"""
        if lado == 'Left':
            self.contagem_esquerda = contagem
        else:
            self.contagem_direita = contagem
        
        total = self.contagem_esquerda + self.contagem_direita
        
        # Verifica match e atualiza pontuação
        if total == self.target_numbers[self.indice_atual]:
            self.contagem_frames += 1
            if self.contagem_frames >= self.FRAMES_CONFIRMACAO:
                self.pontuacao += 1  # Incrementa pontuação
                self.indice_atual = (self.indice_atual + 1) % 5
                self.contagem_frames = 0
                if self.indice_atual == 0:
                    self.target_numbers = self._gerar_numeros_aleatorios()
        else:
            self.contagem_frames = 0
        
        # Mostra o total atual no centro
        centro_x = frame.shape[1] // 2
        centro_y = frame.shape[0] // 2
        
        self._desenhar_texto_com_fundo(
            frame, 
            str(total), 
            (centro_x - 50, centro_y // 4), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (255, 255, 255),
            (0, 0, 0)
        )

    def _mostrar_score(self, frame):
        """Mostra a pontuação atual constantemente na tela"""
        self._desenhar_texto_com_fundo(
            frame,
            f"Pontos: {self.pontuacao}",
            (frame.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),  # Verde
            (0, 0, 0)     # Fundo preto
        )


    def _desenhar_texto_com_fundo(self, frame, texto, posicao, fonte, escala, cor_texto, cor_fundo):
        espessura = 3
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        (largura_texto, altura_texto), _ = cv2.getTextSize(texto, fonte, escala, espessura)
        x, y = posicao
        padding = 10
        
        cv2.rectangle(frame, 
                     (x - padding, y - altura_texto - padding),
                     (x + largura_texto + padding, y + padding),
                     cor_fundo, -1)
        cv2.putText(frame, texto, (x, y), fonte, escala, cor_texto, espessura)

    def _processar_maos(self, frame, resultados):
        for idx, landmarks in enumerate(resultados.multi_hand_landmarks):
            lado = resultados.multi_handedness[idx].classification[0].label
            
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            self._verificar_selecao_cor(landmarks.landmark[8], frame)
            self._desenhar(frame, landmarks)
            
            contagem = self._contar_dedos(landmarks, lado)
            self._mostrar_contagem(frame, contagem, lado)
            
            # Verifica o sinal de L
            #if self._verificar_sinal_L(landmarks, lado):
            #    self._mostrar_mensagem_L(frame)

            # Verifica formato de coração
            #if self._verificar_coracao(landmarks, lado):
            #    self._mostrar_mensagem_coracao(frame)

    def _desenhar_numeros_alvo(self, frame):
        largura = frame.shape[1]
        espacamento = largura // 6
        for i, num in enumerate(self.target_numbers):
            x = espacamento * (i + 1)
            y = frame.shape[0] - 30
            cor = (0, 255, 0) if i == self.indice_atual else (200, 200, 200)
            cor_fundo = (0, 0, 0) if i == self.indice_atual else (100, 100, 100)
            self._desenhar_texto_com_fundo(frame, str(num), (x, y), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, cor, cor_fundo)

    def executar(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self._inicializar_canvas(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = self.hands.process(frame_rgb)

            self._desenhar_caixas_cores(frame)
            frame = cv2.addWeighted(frame, 1, self.canvas, 0.8, 0)

            if resultados.multi_hand_landmarks:
                self._processar_maos(frame, resultados)

            self._desenhar_numeros_alvo(frame)
            self._mostrar_score(frame)  # Adiciona chamada para mostrar score
            cv2.imshow('Jogo de Soma com Dedos', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = CapturaMovimento()
    app.executar()
