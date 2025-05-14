import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# === Настройка путей (укажите ваши) ===
IMG_PATH   = r"C:\Users\USER\clones\thefittest\output_runs\5_ens.png"
MODEL_PATH = r"C:\Users\USER\clones\thefittest\output_runs\model.pkl"

# Признаки
FEATURE_NAMES = [
    "X1: Радиус звукового кармана, мм",
    "X2: Глубина звукового кармана, мм",
    "X3: Площадь звуковых карманов, %",
    "X4: Толщина клеевого слоя, мм",
    "X5: Толщина панели 1, мм",
    "X6: Толщина панели 2, мм",
    "X7: Толщина панели 3, мм",
    "X8: Влажность, %",
    "X9: Частота, Гц",
]

class StubApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GPENN")
        self.geometry("700x650")
        self.resizable(False, False)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.create_train_tab()
        self.create_predict_tab()

    def create_train_tab(self):
        tab = ttk.Frame(self.notebook, width=700, height=620)
        tab.pack_propagate(False)
        self.notebook.add(tab, text="Обучение")

        # Параметры PDPSHAGP
        ttk.Label(tab, text="PDPSHAGP популяция:").place(x=10, y=10)
        self.ent_pdpsha_pop = ttk.Entry(tab, width=10); self.ent_pdpsha_pop.insert(0, "100"); self.ent_pdpsha_pop.place(x=200, y=10)

        ttk.Label(tab, text="PDPSHAGP индивиды:").place(x=10, y=45)
        self.ent_pdpsha_ind = ttk.Entry(tab, width=10); self.ent_pdpsha_ind.insert(0, "20"); self.ent_pdpsha_ind.place(x=200, y=45)

        # Параметры SHADE
        ttk.Label(tab, text="SelfCSHAGA популяция:").place(x=10, y=80)
        self.ent_shade_pop = ttk.Entry(tab, width=10); self.ent_shade_pop.insert(0, "300"); self.ent_shade_pop.place(x=200, y=80)

        ttk.Label(tab, text="SelfCSHAGA индивиды:").place(x=10, y=115)
        self.ent_shade_ind = ttk.Entry(tab, width=10); self.ent_shade_ind.insert(0, "300"); self.ent_shade_ind.place(x=200, y=115)

        # Кнопка запуска
        ttk.Button(tab, text="Запустить обучение", command=self.stub_train).place(x=10, y=160)

        # Результаты
        self.frame_results = ttk.LabelFrame(tab, text="Результаты", width=680, height=400)
        self.frame_results.place(x=10, y=200)
        self.frame_results.pack_propagate(False)
        self.frame_results.place_forget()

        # Ансамбль изображение
        self.lbl_ens_img = ttk.Label(self.frame_results)
        self.lbl_ens_img.place(x=10, y=10)

        # Подпись RMSE и кнопка сохранить под картинкой
        ttk.Label(self.frame_results, text="RMSE:").place(x=10, y=330)
        self.lbl_rmse = ttk.Label(self.frame_results, text="0.0")
        self.lbl_rmse.place(x=60, y=330)

        ttk.Button(self.frame_results, text="Сохранить модель", command=self.save_model).place(x=150, y=325)

    def create_predict_tab(self):
        tab = ttk.Frame(self.notebook, width=700, height=620)
        tab.pack_propagate(False)
        self.notebook.add(tab, text="Прогнозирование")

        # Модель уже загружена (макет)
        ttk.Label(tab, text="Model: model.pkl").place(x=10, y=10)

        # Поля для входных переменных
        self.input_vars = {}
        y_offset = 50
        row_height = 30
        for i, name in enumerate(FEATURE_NAMES):
            y_pos = y_offset + i * row_height
            ttk.Label(tab, text=name + ":").place(x=10, y=y_pos)
            ent = ttk.Entry(tab, width=15)
            ent.place(x=300, y=y_pos)
            self.input_vars[name] = ent

        btn_y = y_offset + len(FEATURE_NAMES) * row_height + 20
        ttk.Button(tab, text="Прогнозировать", command=self.stub_predict).place(x=10, y=btn_y)
        
        out_y = btn_y + 40
        ttk.Label(tab, text="Y: Среднее Звук. Давление выход, dB").place(x=10, y=out_y)
        self.lbl_output = ttk.Label(tab, text="0.0")
        self.lbl_output.place(x=300, y=out_y)

    def stub_train(self):
        self.frame_results.place(x=10, y=200)
        try:
            img = Image.open(IMG_PATH)
        except:
            img = Image.new('RGB', (400,300), 'gray')
        self.en_img = ImageTk.PhotoImage(img.resize((400,300)))
        self.lbl_ens_img.config(image=self.en_img)
        self.lbl_rmse.config(text="1.56")

    def save_model(self):
        messagebox.showinfo("Сохранение", "Модель сохранена (заглушка)")

    def stub_predict(self):
        self.lbl_output.config(text="75.3521")

if __name__ == '__main__':
    app = StubApp()
    app.mainloop()
