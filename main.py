import tkinter as tk

from voice_emotion_analyzer_app import EmotionRecognizerApp

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognizerApp(root)
    root.mainloop()