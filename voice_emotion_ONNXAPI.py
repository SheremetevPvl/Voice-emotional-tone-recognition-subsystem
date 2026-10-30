import os
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoFeatureExtractor
import torch.nn.functional as F
import wave

class VoiceEmotionONNXAPI:
    def __init__(self, model_path: str = "hubert_emotion_model.onnx", model_id: str = "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned"):
        self.model_path = model_path
        
        # Загружаем оригинальный конфиг и процессор из HF (нужен для правильной подготовки аудио)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        
        # Получаем список эмоций (id2label) из конфигурации модели
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id)
        self.id2label = config.id2label # Словарь вида {0: 'neutral', 1: 'anger', ...}

        # Инициализируем ONNX Runtime с оптимизацией под все ядра CPU
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # Задействуем 4 ядра процессора
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print(f"Загрузка ONNX модели с CPU ускорением...")
        self.session = ort.InferenceSession(self.model_path, options, providers=['CPUExecutionProvider'])
        print("Модель успешно запущена и готова к работе!")

    def analyze_audio_file(self, file_path: str) -> dict:
        """анализ аудиофайла с помощью HuBERT-ONNX."""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                sr = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)
                
                # Переводим байты в numpy-массив (16-bit PCM)
                speech_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Если стерео, усредняем до моно
                if wav_file.getnchannels() > 1:
                    speech_array = speech_array.reshape(-1, wav_file.getnchannels()).mean(axis=1)
                    
        except Exception as e:
            # Если это не чистый WAV, пробуем через librosa, нормализуя путь
            try:
                import librosa
                speech_array, sr = librosa.load(os.path.normpath(file_path), sr=None, mono=True)
            except Exception as inner_e:
                raise RuntimeError(
                    f"Не удалось прочитать файл. Убедитесь, что это стандартный PCM 16-bit WAV файл.\n"
                    f"Техническая ошибка: {e} | {inner_e}"
                )
        
        # 2. Официальный feature_extractor для нормализации
        inputs = self.feature_extractor(
            speech_array, 
            sampling_rate=16000, 
            return_tensors="np", 
            padding=True
        )
        audio_input = inputs["input_values"].astype(np.float32)

        # 3. Прогон через ONNX сессию на процессоре
        ort_inputs = {self.session.get_inputs()[0].name: audio_input}
        ort_outs = self.session.run(None, ort_inputs)
        logits = ort_outs[0] # Получаем сырые логиты

        # 4. Переводим логиты в вероятности (Softmax)
        logits_tensor = torch.from_numpy(logits)
        probabilities = F.softmax(logits_tensor, dim=-1).numpy()[0]

        # 5. Сопоставляем результаты с названиями эмоций
        results = {}
        for idx, score in enumerate(probabilities):
            emotion_name = self.id2label.get(idx, f"Emo_{idx}")
            results[emotion_name] = round(float(score) * 100, 2)

        # Сортируем по убыванию вероятности
        sorted_probs = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        return sorted_probs

    def close(self):
        print("Ресурсы освобождены.")