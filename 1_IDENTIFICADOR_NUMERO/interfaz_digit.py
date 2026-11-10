# -*- coding: utf-8 -*-
"""Interfaz de escritorio (Tkinter) para reconocer dígitos escritos a mano.

Abre una ventana nativa de Windows (NO usa navegador): cargás una imagen desde
el disco y el modelo te dice qué dígito es, con las probabilidades por clase y
la vista 28x28 que realmente ve la red.

Requisito: haber ejecutado antes `python codetam.py` (entrena y guarda modelo_mnist.keras).
Ejecutar con:
    python interfaz_digit.py
"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from scipy import ndimage
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "modelo_mnist.keras")

TAM_LIENZO = 280


def preprocesar(imagen):
    """Convierte una imagen de un dígito al formato MNIST:
    28x28, fondo negro, trazo blanco, dígito en caja de 20x20 y centrado por
    centro de masa. Devuelve un array 28x28 uint8 con valores 0-255."""
    if isinstance(imagen, dict):
        imagen = imagen.get("composite", imagen.get("background"))

    if isinstance(imagen, str):
        imagen = Image.open(imagen)

    if isinstance(imagen, Image.Image):
        img = imagen
    else:
        img = Image.fromarray(np.asarray(imagen).astype("uint8"))

    if img.mode in ("RGBA", "LA"):
        fondo = Image.new("RGB", img.size, (255, 255, 255))
        fondo.paste(img, mask=img.split()[-1])
        img = fondo

    arr = np.asarray(img.convert("L")).astype("float32")

    borde = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.median(borde) > 127:
        arr = 255 - arr

    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    arr[arr < 40] = 0

    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        return np.zeros((28, 28), dtype="uint8")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = arr.shape
    escala = 20.0 / max(h, w)
    nuevo_w = max(1, int(round(w * escala)))
    nuevo_h = max(1, int(round(h * escala)))
    arr = np.asarray(
        Image.fromarray(arr.astype("uint8")).resize(
            (nuevo_w, nuevo_h), Image.Resampling.LANCZOS
        )
    ).astype("float32")

    lienzo = np.zeros((28, 28), dtype="float32")
    y0, x0 = (28 - nuevo_h) // 2, (28 - nuevo_w) // 2
    lienzo[y0:y0 + nuevo_h, x0:x0 + nuevo_w] = arr

    cy, cx = ndimage.center_of_mass(lienzo)
    lienzo = ndimage.shift(lienzo, (14 - cy, 14 - cx), order=1, mode="constant")

    return np.clip(lienzo, 0, 255).astype("uint8")


class App:
    def __init__(self, raiz, modelo):
        self.raiz = raiz
        self.modelo = modelo
        self.img28 = None
        self.foto_original = None
        self.foto_procesada = None

        raiz.title("Reconocedor de dígitos - MNIST")
        raiz.geometry("800x720")
        raiz.resizable(True, True)

        marco_botones = ttk.Frame(raiz, padding=10)
        marco_botones.pack(fill="x")

        ttk.Button(
            marco_botones, text="Cargar imagen...", command=self.cargar_imagen
        ).pack(side="left")
        ttk.Button(
            marco_botones, text="Predecir", command=self.predecir
        ).pack(side="left", padx=8)

        self.etiqueta_estado = ttk.Label(
            marco_botones, text="Elegí una imagen para empezar."
        )
        self.etiqueta_estado.pack(side="left", padx=12)

        marco_imagenes = ttk.Frame(raiz, padding=10)
        marco_imagenes.pack(fill="both", expand=True)

        col_izq = ttk.Frame(marco_imagenes)
        col_izq.pack(side="left", padx=10)
        ttk.Label(col_izq, text="Imagen original").pack()
        self.canvas_original = tk.Canvas(
            col_izq,
            width=TAM_LIENZO,
            height=TAM_LIENZO,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        self.canvas_original.pack(pady=6)

        col_der = ttk.Frame(marco_imagenes)
        col_der.pack(side="left", padx=10)
        ttk.Label(col_der, text="Lo que ve el modelo (28x28)").pack()
        self.canvas_procesada = tk.Canvas(
            col_der,
            width=TAM_LIENZO,
            height=TAM_LIENZO,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        self.canvas_procesada.pack(pady=6)

        marco_resultado = ttk.Frame(raiz, padding=10)
        marco_resultado.pack(fill="both", expand=True)

        self.etiqueta_digito = ttk.Label(
            marco_resultado, text="Dígito: -", font=("Segoe UI", 20, "bold")
        )
        self.etiqueta_digito.pack(anchor="w")

        self.etiqueta_probs = ttk.Label(
            marco_resultado, text="", font=("Consolas", 11), justify="left"
        )
        self.etiqueta_probs.pack(anchor="w", pady=6)

    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Elegí una imagen de un dígito",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not ruta:
            return

        try:
            imagen = Image.open(ruta)
            imagen.load()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return

        try:
            self.mostrar_original(imagen)
            self.img28 = preprocesar(imagen)
            self.mostrar_procesada(self.img28)
            self.etiqueta_estado.config(text=os.path.basename(ruta))
            self.etiqueta_digito.config(text="Dígito: -")
            self.etiqueta_probs.config(text="")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo procesar la imagen:\n{e}")

    def predecir(self):
        if self.img28 is None:
            messagebox.showinfo("Sin imagen", "Primero cargá una imagen.")
            return

        entrada = self.img28.reshape(1, 784).astype("float32") / 255.0
        probs = self.modelo.predict(entrada, verbose=0)[0]

        digito = int(np.argmax(probs))
        self.etiqueta_digito.config(
            text=f"Dígito: {digito}   ({probs[digito] * 100:.1f}% de confianza)"
        )

        lineas = []
        for d in range(10):
            marca = "   <--" if d == digito else ""
            lineas.append(f"  {d}:  {probs[d] * 100:6.2f}%{marca}")
        self.etiqueta_probs.config(text="\n".join(lineas))
        self.etiqueta_estado.config(text="Predicción lista.")

    def mostrar_original(self, pil_img):
        img = pil_img.convert("RGB")
        img.thumbnail((TAM_LIENZO, TAM_LIENZO))
        self.foto_original = ImageTk.PhotoImage(img)
        self.canvas_original.delete("all")
        self.canvas_original.create_image(
            TAM_LIENZO // 2, TAM_LIENZO // 2, image=self.foto_original
        )

    def mostrar_procesada(self, arr28):
        img = Image.fromarray(arr28).resize(
            (TAM_LIENZO, TAM_LIENZO), Image.Resampling.LANCZOS
        ).convert("RGB")
        self.foto_procesada = ImageTk.PhotoImage(img)
        self.canvas_procesada.delete("all")
        self.canvas_procesada.create_image(
            TAM_LIENZO // 2, TAM_LIENZO // 2, image=self.foto_procesada
        )


def main():
    if not os.path.exists(RUTA_MODELO):
        raiz_error = tk.Tk()
        raiz_error.withdraw()
        messagebox.showerror(
            "Falta el modelo",
            "No se encontró modelo_mnist.keras.\n\n"
            "Ejecutá primero:\n    python entrenar_modelo.py",
        )
        raiz_error.destroy()
        return

    print("Cargando modelo...")
    modelo = tf.keras.models.load_model(RUTA_MODELO)
    print("Modelo cargado. Abriendo ventana...")

    raiz = tk.Tk()
    App(raiz, modelo)
    raiz.mainloop()


if __name__ == "__main__":
    main()
