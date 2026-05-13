import matplotlib
matplotlib.use("Agg") # График сохраняется в файл, без показа окна
import matplotlib.pyplot as plt  # matplotlib - построение графиков
import numpy as np  # numpy - работа с массивами и вычисление среднего
import os
from collections import Counter

import nltk  # nltk - инструменты для обработки естественного языка
from nltk.tokenize import sent_tokenize, word_tokenize  # Готовые функции для разбиения текста на предложения и слова
nltk.download("punkt", quiet=True)  # Загружаем модель "punkt" для разбиения текста

LANG = "russian" # Язык токенизации
INPUT_FILE = "input.txt" # Имя файла с текстом, если есть
OUTPUT_FILE = "hist.png" # Имя файла для сохранения графика


# 0) Чтение текста
def read_text(path: str) -> str:
    
    # Текст из файла или возвращаем пустую строку, если файл отсутствует, то пустую
    if not path or not os.path.exists(path):
        print(f"[warn] Файл '{path}' не найден")
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
            if not data.strip():
                print(f"[warn] Файл '{path}' пуст")
                return ""
            return data
    except Exception as e:
        print(f"[error] Не удалось прочитать '{path}': {e}")
        return ""


# 1) Разбиваем исходный текст на отдельные предложения (список строк)
# 2) Считаем длину каждого предложения
# - используем word_tokenize, чтобы разбить предложение на токены
# - по умолчанию туда попадают и слова, и знаки препинания, и числа
# - чтобы считать только слова:
# 1. t.isalpha() оставляет только токены, состоящие из букв
# 2. пунктуация и числа будут отсеяны
def tokenize_and_lengths(text: str, lang: str) -> tuple[list[str], list[int]]:
    try:
        sentences = sent_tokenize(text, language=lang) if text.strip() else []
    except Exception as e:
        print(f"[error] Ошибка токенизации предложений: {e}")
        sentences = []

    lengths: list[int] = []
    for i, s in enumerate(sentences, 1):
        try:
            words_only = [t for t in word_tokenize(s, language=lang) if t.isalpha()]
            length = len(words_only)
        except Exception as e:
            print(f"[error] Ошибка токенизации слов в предложении #{i}: {e}")
            length = 0
        lengths.append(length)
        print(f"{i}) {length} слов — {s}")
    return sentences, lengths


# 3) Гистограмма
def histogram(lengths: list[int], out_path: str):
    
    #Если текста нет — сохраняем пустой график
    plt.figure()
    if lengths:
        counter = Counter(lengths)  # считаем, сколько предложений оказалось длиной 4 слова, 5 слов, 6 слов и т.д.
        plt.bar(counter.keys(), counter.values(), width=0.05, edgecolor="black")
        plt.title("Распределение длины предложений")
        plt.xlabel("Количество слов в предложении")
        plt.ylabel("Частота (количество одинаковых предложений по длине)")
        plt.xticks(range(min(lengths), max(lengths) + 1))
    else:
        plt.title("Распределение длины предложений (нет данных)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[info] График сохранён в '{out_path}'")


def main():
    # 0) Чтение текста
    text = read_text(INPUT_FILE)

    # 1-2) Токенизация и подсчёт длин
    sentences, lengths = tokenize_and_lengths(text, LANG)

    # 4) Расчёт средней длины предложения
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    print("Средняя длина предложения:", round(avg_len, 2))

    # 3) Построение и сохранение гистограммы
    histogram(lengths, OUTPUT_FILE)

if __name__ == "__main__":
    main()
