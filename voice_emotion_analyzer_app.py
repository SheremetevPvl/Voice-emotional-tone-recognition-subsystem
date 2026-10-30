import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

from voice_emotion_ONNXAPI import VoiceEmotionONNXAPI

class EmotionRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ эмоций (ONNX / CPU оптимизация)")
        self.root.geometry("520x480")
        self.root.resizable(False, False)

        # Инициализация API (трансформер через ONNX)
        self.api = VoiceEmotionONNXAPI()

        self.create_widgets()

    def create_widgets(self):
        """
        Виджет для взаимодействия
        """
        title_label = tk.Label(self.root, text="Распознавание эмоций (Wav2Vec2 + ONNX)", font=("Arial", 13, "bold"))
        title_label.pack(pady=15)

        frame_file = tk.Frame(self.root)
        frame_file.pack(pady=10)

        self.file_path_var = tk.StringVar(value="Аудиофайл не выбран...")
        btn_browse = tk.Button(frame_file, text="Выбрать файл", command=self.load_file, width=15, bg="#4CAF50", fg="white")
        btn_browse.pack(side=tk.LEFT, padx=5)
        
        lbl_path = tk.Label(frame_file, textvariable=self.file_path_var, width=32, anchor="w", fg="#555")
        lbl_path.pack(side=tk.LEFT, padx=5)

        # Кнопка запуска анализа
        self.btn_analyze = tk.Button(self.root, text="Запустить анализ (CPU)", command=self.start_async_analysis, state=tk.DISABLED, width=28, height=2, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        self.btn_analyze.pack(pady=15)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=420, mode="indeterminate")
        
        frame_results = tk.LabelFrame(self.root, text=" Результаты анализа (Вероятности) ", font=("Arial", 10, "bold"))
        frame_results.pack(fill="both", expand=True, padx=20, pady=10)

        self.result_text = tk.Text(frame_results, height=8, width=52, state=tk.DISABLED, font=("Courier", 10))
        self.result_text.pack(padx=10, pady=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
        if file_path:
            self.full_file_path = file_path
            file_name = file_path.split("/")[-1].split("\\")[-1]
            self.file_path_var.set(file_name)
            self.btn_analyze.config(state=tk.NORMAL)

    def start_async_analysis(self):
        """Запуск анализа в отдельном потоке, чтобы интерфейс не зависал."""
        self.btn_analyze.config(state=tk.DISABLED)
        self.progress.pack(pady=5)
        self.progress.start(10)
        
        file_path = self.full_file_path 
        
        threading.Thread(target=self.run_analysis, args=(file_path,), daemon=True).start()

    def run_analysis(self, file_path):
        try:
            # Вызов метода API (инференс ONNX)
            results = self.api.analyze_audio_file(file_path)
            
            # Обновление UI с возвратом в главный поток
            self.root.after(0, lambda: self.show_results(results))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка обработки: {e}"))
            self.root.after(0, self.reset_ui)

    def show_results(self, results):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        
        for emotion, score in results.items():
            bar = "█" * int(score // 4) # Визуализация шкалы уверенности модели
            self.result_text.insert(tk.END, f"{emotion:<12} | {score:>5.2f}% {bar}\n")
            
        self.result_text.config(state=tk.DISABLED)
        self.reset_ui()

    def reset_ui(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.btn_analyze.config(state=tk.NORMAL)