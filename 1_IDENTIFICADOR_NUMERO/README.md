# Reconocedor de dígitos escritos a mano (MNIST)

Proyecto que reconoce dígitos manuscritos (0–9) usando una **red neuronal densa** entrenada con el dataset **MNIST**, más una **interfaz de escritorio nativa** (Tkinter, sin navegador) para cargar una imagen y ver la predicción junto con la probabilidad de cada clase.

El modelo es una red de tres capas completamente conectadas con arquitectura:

```
784  ->  64  ->  16  ->  10
      ReLU    ReLU   Softmax
```

---

## Estructura del proyecto

| Archivo | Rol |
|---|---|
| `codetam.py` | Define la red, **entrena** (normalizando los píxeles con `÷255`) y **guarda** el modelo en `modelo_mnist.keras`. |
| `interfaz_digit.py` | **Interfaz de escritorio (Tkinter)**. Contiene la función `preprocesar()` (convierte la imagen al formato MNIST), carga el modelo entrenado y muestra la predicción. |
| `modelo_mnist.keras` | Pesos del modelo entrenado (generado por `codetam.py`). |

---

## Requisitos e instalación

- Python 3.13
- Dependencias:

```bash
pip install tensorflow numpy pillow scipy
```

> `tkinter` viene incluido con Python en Windows; no requiere instalación aparte.

---

## Cómo usar

**1) Entrenar el modelo** (genera/actualiza `modelo_mnist.keras`):

```bash
python codetam.py
```

Corre 30 epochs en CPU (unos 2–3 minutos) y al final imprime la ruta donde guardó el modelo.

**2) Abrir la interfaz:**

```bash
python interfaz_digit.py
```

Se abre una ventana nativa donde se carga una imagen con el botón **Cargar imagen...** y se predice con **Predecir**. La interfaz muestra:

- el dígito predicho y su nivel de confianza,
- las probabilidades de las 10 clases (0–9),
- la imagen 28×28 tal como la "ve" el modelo.

> La interfaz necesita que exista `modelo_mnist.keras`. Si no está, avisa que se ejecute primero `codetam.py`.

---

## Preprocesado de la imagen

Antes de predecir, cada imagen se transforma al mismo formato con el que se entrenó MNIST (28×28, fondo negro, trazo blanco, dígito centrado). Pasos:

1. Convertir a escala de grises.
2. Si el fondo es claro, invertir la imagen (MNIST es trazo blanco sobre negro).
3. Estirar el contraste a 0–255 y descartar ruido débil.
4. Recortar al rectángulo que contiene el dígito.
5. Escalar para que el lado mayor mida 20 px (dentro de un lienzo de 28×28):

$$s = \frac{20}{\max(h, w)}, \qquad w' = \mathrm{round}(w \cdot s),\quad h' = \mathrm{round}(h \cdot s)$$

6. Centrar por **centro de masa** (así están centrados los dígitos de MNIST). El centro de masa de la imagen de intensidades $I(i,j)$ es:

$$\bar{y} = \frac{\sum_{i}\sum_{j} i \cdot I(i,j)}{\sum_{i}\sum_{j} I(i,j)}, \qquad
\bar{x} = \frac{\sum_{i}\sum_{j} j \cdot I(i,j)}{\sum_{i}\sum_{j} I(i,j)}$$

y se desplaza la imagen en $(\Delta y, \Delta x) = (14 - \bar{y},\, 14 - \bar{x})$ para que quede centrada en el píxel (14, 14).

Al final se obtiene un vector de entrada $\mathbf{x} \in \mathbb{R}^{784}$ (28×28 aplanado) con valores en 0–255, que luego se normaliza con $\mathbf{x} / 255$.

---

## Matemática del modelo

### 1. Normalización de entrada

Los píxeles vienen en el rango $[0, 255]$. Se escalan a $[0, 1]$:

$$\mathbf{x}' = \frac{\mathbf{x}}{255}$$

Esto estabiliza y acelera el entrenamiento (entradas de magnitud grande producen gradientes inestables).

### 2. Capas densas (forward pass)

Cada capa calcula una combinación lineal seguida de una activación. Para la capa $\ell$:

$$\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}, \qquad
\mathbf{a}^{(\ell)} = f\!\left(\mathbf{z}^{(\ell)}\right)$$

con las dimensiones de los pesos:

- $W^{(1)} \in \mathbb{R}^{64 \times 784}$, $\mathbf{b}^{(1)} \in \mathbb{R}^{64}$
- $W^{(2)} \in \mathbb{R}^{16 \times 64}$, $\mathbf{b}^{(2)} \in \mathbb{R}^{16}$
- $W^{(3)} \in \mathbb{R}^{10 \times 16}$, $\mathbf{b}^{(3)} \in \mathbb{R}^{10}$

donde $\mathbf{a}^{(0)} = \mathbf{x}'$ es la entrada normalizada.

### 3. Funciones de activación

**ReLU** en las capas ocultas (anula los valores negativos):

$$f(z) = \max(0, z), \qquad f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}$$

**Softmax** en la capa de salida (convierte los $z$ en probabilidades que suman 1):

$$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{10} e^{z_j}}, \qquad k = 0, \dots, 9$$

El vector de salida $\hat{\mathbf{y}} \in \mathbb{R}^{10}$ es la probabilidad estimada de cada dígito.

### 4. Función de pérdida

Se usa **entropía cruzada categórica dispersa** (*sparse categorical cross-entropy*). Si $c$ es la clase correcta (índice real, sin one-hot):

$$\mathcal{L} = -\log\!\left(\hat{y}_c\right)$$

o, de forma general con un objetivo one-hot $t_k$:

$$\mathcal{L} = -\sum_{k=1}^{10} t_k \log\!\left(\hat{y}_k\right)$$

### 5. Optimización (Adam)

Los parámetros $\theta = \{W^{(\ell)}, \mathbf{b}^{(\ell)}\}$ se ajustan para minimizar $\mathcal{L}$. Se usa el optimizador **Adam**, que adapta el paso por parámetro combinando momento y magnitud del gradiente:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\,\mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t}, \qquad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{aligned}
$$

donde $\mathbf{g}_t = \nabla_\theta \mathcal{L}$ es el gradiente, y los valores por defecto de Keras son $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$ y $\eta = 10^{-3}$.

### 6. Predicción

Una vez entrenado, para una imagen nueva se hace **una sola pasada hacia adelante** (forward pass) y se elige la clase más probable:

$$\hat{c} = \arg\max_{k}\; \hat{y}_k$$

Es decir, **la predicción no busca en la base de datos de MNIST**: MNIST solo se usa para aprender los pesos durante el entrenamiento. En producción, únicamente se cargan los pesos guardados en `modelo_mnist.keras`.

---

## Resultados

- Exactitud aproximada en el conjunto de prueba: **~98–99%**.
- La confianza que muestra Softmax es una estimación del modelo; **no garantiza** que la predicción sea correcta.

---

## Notas y limitaciones

- En Windows nativo, TensorFlow ≥ 2.11 **no usa GPU**; el entrenamiento corre en CPU.
- La precisión reportada corresponde a dígitos con un formato similar a MNIST (trazo definido, buen contraste, centrado). Con imágenes muy ruidosas, de bajo contraste o con el dígito cortado, la precisión disminuye.
