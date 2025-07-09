# 🎮 CaptureMoviment - Interactive Gesture-Controlled Gaming System

Um sistema interativo de jogos controlados por gestos usando visão computacional e rastreamento de mãos em tempo real.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Jogos Disponíveis](#jogos-disponíveis)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Controles por Gestos](#controles-por-gestos)
- [Configuração](#configuração)
- [Contribuição](#contribuição)
- [Licença](#licença)

## 🎯 Sobre o Projeto

O **CaptureMoviment** é um sistema inovador que combina visão computacional com jogos interativos, permitindo que os usuários controlem aplicações usando apenas gestos das mãos. O projeto utiliza a biblioteca MediaPipe para rastreamento de mãos em tempo real e oferece três experiências distintas:

- **Virtual Paint**: Pintura digital com gestos
- **Number Game**: Jogo educativo de matemática
- **People Detection**: Sistema de detecção de pessoas

## ✨ Funcionalidades

### 🎨 Características Principais

- **Interface Adaptativa**: Ajusta-se automaticamente a diferentes resoluções de tela
- **Controle por Gestos**: Navegação intuitiva usando movimentos das mãos
- **Sistema de Confirmação**: Requer 15 frames consistentes para confirmar seleções
- **Múltiplos Jogos**: Três aplicações diferentes em um só sistema
- **Detecção Robusta**: Usa YOLO e detecção de movimento como fallback
- **Feedback Visual**: Círculos de progresso e indicadores visuais

### 🖥️ Compatibilidade

- **Resolução**: Adapta-se automaticamente à resolução da tela
- **Câmera**: Funciona com webcams padrão
- **Sistema**: Windows, macOS, Linux

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**: Linguagem principal
- **OpenCV**: Processamento de imagem e vídeo
- **MediaPipe**: Rastreamento de mãos e pose
- **NumPy**: Computação numérica
- **Tkinter**: Interface gráfica para detecção de tela
- **YOLO (YOLOv8)**: Detecção de pessoas (opcional)
- **Ultralytics**: Framework para YOLO

## 📦 Instalação

### Pré-requisitos

```bash
python >= 3.8
pip
webcam conectada
```

### Instalação das Dependências

```bash
# Clone o repositório
git clone https://github.com/listendope/captureMoviment.git
cd captureMoviment

# Instale as dependências básicas
pip install opencv-python mediapipe numpy

# Para detecção avançada de pessoas (opcional)
pip install ultralytics

# Ou instale todas as dependências de uma vez
pip install -r requirements.txt
```

### Arquivo requirements.txt

```txt
opencv-python>=4.5.0
mediapipe>=0.8.0
numpy>=1.21.0
ultralytics>=8.0.0
```

## 🚀 Como Usar

### Execução

```bash
python main.py
```

### Controles Básicos

1. **Navegação**: Aponte o dedo indicador para os botões
2. **Seleção**: Mantenha o dedo sobre o botão por 1 segundo
3. **Retorno**: Aponte para "Voltar menu" em qualquer jogo
4. **Saída**: Pressione 'q' para sair

## 🎮 Jogos Disponíveis

### 🎨 Virtual Paint

**Descrição**: Sistema de pintura digital controlado por gestos

**Controles**:
- **Mão Direita**: Pincel para desenhar
- **Mão Esquerda**: Borracha para apagar
- **Gesto de Desenho**: Dedo indicador levantado + outros dedos abaixados
- **Seleção de Cor**: Aponte para as cores na paleta

**Cores Disponíveis**:
- Vermelho, Verde, Azul, Amarelo
- Roxo, Branco, Laranja, Rosa, Ciano

**Atalhos**:
- `c`: Limpar tela
- `q`: Sair

### 🔢 Number Game

**Descrição**: Jogo educativo de matemática usando contagem de dedos

**Como Jogar**:
1. Uma equação matemática aparece na tela
2. Mostre o número de dedos correspondente à resposta
3. O sistema conta automaticamente os dedos de ambas as mãos
4. Pontuação aumenta a cada resposta correta

**Características**:
- Operações: Adição e subtração
- Números de 0 a 10
- Sistema de pontuação progressiva
- Detecção inteligente de dedos

### 👥 People Detection

**Descrição**: Sistema avançado de detecção de pessoas

**Métodos de Detecção**:

1. **YOLO (Primário)**:
   - Alta precisão
   - Múltiplas pessoas simultaneamente
   - Caixas delimitadoras precisas

2. **Detecção de Movimento (Fallback)**:
   - Subtração de fundo
   - Requer movimento para detecção
   - Funciona sem YOLO

**Controles**:
- `r`: Resetar fundo (modo movimento)
- `q`: Sair

**Informações Exibidas**:
- Contagem de pessoas em tempo real
- Método de detecção ativo
- Caixas delimitadoras coloridas

## 📁 Estrutura do Projeto

```
captureMoviment/
│
├── main.py              # Menu principal e navegação
├── VirtualPaint.py      # Jogo de pintura digital
├── NumberGame.py        # Jogo educativo de matemática
├── PeopleDetection.py   # Sistema de detecção de pessoas
├── README.md           # Documentação do projeto
└── requirements.txt    # Dependências do projeto
```

### 🏗️ Arquitetura do Sistema

```
┌─────────────────┐
│    main.py      │ ← Menu Principal
│   (Navegação)   │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │   Jogos   │
    └─────┬─────┘
          │
    ┌─────▼─────┬─────────────┬─────────────┐
    │Virtual    │Number       │People       │
    │Paint      │Game         │Detection    │
    └───────────┴─────────────┴─────────────┘
```

## 🤲 Controles por Gestos

### Detecção de Mãos

O sistema usa **MediaPipe** para detectar 21 pontos de referência em cada mão:

```python
# Pontos principais utilizados
INDEX_TIP = 8    # Ponta do indicador
INDEX_PIP = 6    # Articulação do indicador
MIDDLE_TIP = 12  # Ponta do médio
THUMB_TIP = 4    # Ponta do polegar
```

### Gestos Reconhecidos

1. **Apontar**: Indicador levantado para navegação
2. **Desenhar**: Indicador levantado + outros dedos abaixados
3. **Contar**: Todos os dedos para contagem numérica
4. **Apagar**: Gesto específico da mão esquerda

### Sistema de Confirmação

```python
FRAMES_TO_CONFIRM = 15  # ~0.5 segundos a 30fps
```

## ⚙️ Configuração

### Ajustes de Tela

O sistema detecta automaticamente a resolução e ajusta:

```python
# Detecção automática
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Elementos adaptativos
font_scale = max(1.0, screen_width / 1280)
button_size = max(40, screen_width // 30)
```

### Configurações de Câmera

```python
# Configuração padrão
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
```

### Sensibilidade de Detecção

```python
# MediaPipe Hand Detection
min_detection_confidence = 0.7
min_tracking_confidence = 0.5

# YOLO People Detection
confidence_threshold = 0.5
```

## 🔧 Personalização

### Adicionando Novas Cores (Virtual Paint)

```python
self.colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'nova_cor': (B, G, R),  # Adicione aqui
    # ...
}
```

### Modificando Dificuldade (Number Game)

```python
# Em _generate_equation()
answer = random.randint(1, 15)  # Aumentar range
```

### Configurando Zonas de Detecção

```python
# Exemplo de zona personalizada
zones = [
    {
        'coords': (100, 100, 400, 300),
        'label': 'Zona 1',
        'color': (255, 255, 0)
    }
]
```

## 🐛 Solução de Problemas

### Problemas Comuns

1. **Câmera não detectada**:
   ```python
   # Tente diferentes índices
   cap = cv2.VideoCapture(1)  # ou 2, 3...
   ```

2. **YOLO não carrega**:
   ```bash
   pip install ultralytics
   # O sistema usará detecção de movimento automaticamente
   ```

3. **Performance baixa**:
   - Reduza a resolução da câmera
   - Feche outros programas que usam a câmera
   - Verifique se há boa iluminação

4. **Gestos não reconhecidos**:
   - Certifique-se de ter boa iluminação
   - Mantenha as mãos visíveis para a câmera
   - Evite fundos muito complexos

## 📊 Performance

### Requisitos Mínimos

- **CPU**: Intel i3 ou equivalente
- **RAM**: 4GB
- **Câmera**: 720p, 30fps
- **Python**: 3.8+

### Otimizações

- Frame rate adaptativo
- Processamento otimizado de imagem
- Detecção seletiva de características
- Cache de modelos ML

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. **Fork** o projeto
2. Crie uma **branch** para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. Abra um **Pull Request**

### Diretrizes de Contribuição

- Mantenha o código limpo e comentado
- Teste todas as funcionalidades
- Atualize a documentação quando necessário
- Siga o padrão de código existente

## 📝 Roadmap

### Próximas Funcionalidades

- [ ] **Novos Jogos**: Jogo da memória, quebra-cabeça
- [ ] **Multiplayer**: Suporte para múltiplos jogadores
- [ ] **Gravação**: Salvar sessões de jogo
- [ ] **Configurações**: Interface para ajustes
- [ ] **Estatísticas**: Tracking de performance
- [ ] **Temas**: Diferentes temas visuais
- [ ] **Acessibilidade**: Melhor suporte para diferentes usuários

### Melhorias Técnicas

- [ ] **Otimização**: Melhor performance
- [ ] **Estabilidade**: Tratamento de erros aprimorado
- [ ] **Modularidade**: Arquitetura mais flexível
- [ ] **Testes**: Suite de testes automatizados

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**listendope**
- GitHub: [@listendope](https://github.com/listendope)
- Projeto: [captureMoviment](https://github.com/listendope/captureMoviment)

## 🙏 Agradecimentos

- **MediaPipe** pela excelente biblioteca de ML
- **OpenCV** pela base de visão computacional
- **Ultralytics** pelo framework YOLO
- **Comunidade Python** pelo suporte e recursos

## 📞 Suporte

Se você encontrar problemas ou tiver dúvidas:

1. Verifique a seção [Solução de Problemas](#-solução-de-problemas)
2. Abra uma [Issue](https://github.com/listendope/captureMoviment/issues)
3. Consulte a [documentação](https://github.com/listendope/captureMoviment/wiki)

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!**

---

*Desenvolvido com ❤️ usando Python e Computer Vision*
