import sys # для завершения программы через sys.exit()
import argparse # для парсинга аргументов командной строки
import pandas as pd # для работы с таблицами (чтение, обработка, фильтрация данных)
import matplotlib # для построения графиков
matplotlib.use("Agg") # отключаем графический интерфейс
import matplotlib.pyplot as plt

"""
Создаём и настраивает парсер аргументов командной строки
Разделяем аргументы на группы: ввод/вывод, столбцы, очистка данных.
Возвращаем объект argparse.Namespace с параметрами пользовател
"""
# ----------------- Парсер аргументов -----------------
def ParseArgs():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # пути и разделители
    grp_io = p.add_argument_group("Ввод/Вывод")
    grp_io.add_argument("--path", required=True, help="Путь к входному txt файлу")
    grp_io.add_argument("--sep", default=r"\s+", help="Разделитель столбцов, по умолчанию пробел/tab")
    grp_io.add_argument("--save-plot", default=None, help="Путь для сохранения графика png")

    # имена столбцов
    grp_cols = p.add_argument_group("Столбцы")
    grp_cols.add_argument("--id-col", default=None, help="Идентификатор (например, 'Имя')")
    grp_cols.add_argument("--col-a", required=True, help="Первый столбец (например, 'Рост')")
    grp_cols.add_argument("--col-b", required=True, help="Второй столбец (например, 'Рост_отца')")
    grp_cols.add_argument("--new-col", required=True, help="Имя нового столбца (например, 'Разница_роста')")

    # очистка и фильтрация
    grp_clean = p.add_argument_group("Очистка данных")
    grp_clean.add_argument("--drop-na", action="store_true", help="Удалять строки с NaN после приведения/фильтрации")
    grp_clean.add_argument("--min-val", type=float, default=None, help="Нижняя граница значений")
    grp_clean.add_argument("--max-val", type=float, default=None, help="Верхняя граница значений")
    return p.parse_args()


"""
Загружаем данные из текстового файла в DataFrame
Проверяем наличие файла, корректность чтения и непустоту таблицы
Аргументы:
    path - путь к файлу
    sep - разделитель столбцов
Возвращаем DataFrame с очищенными заголовками столбцов
"""
# ----------------- Загрузка и проверка данных -----------------
# e - переменная, хранящая текст пойманной ошибки, чтобы можно было её вывести человеку
def LoadTable(path, sep):
    try:
        df = pd.read_csv(path, sep=sep) # читаем таблицу
    except FileNotFoundError:
        sys.exit(f"[Ошибка] Файл не найден: {path}")
    except Exception as e:
        sys.exit(f"[Ошибка] Не удалось прочитать '{path}': {e}")
    if df.empty:
        sys.exit(f"[Ошибка] Таблица из '{path}' пуста.")
    df.columns = [str(c).strip() for c in df.columns] # убираем лишние пробелы
    return df


def CheckColumns(df, required):
    # проверка наличия обязательных столбцов
    missing = [c for c in required if c and c not in df.columns]
    if missing:
        sys.exit(f"[Ошибка] Нет столбцов: {missing}. Доступные: {list(df.columns)}")


"""
'Nan' - not a numbers т.е не число
Приводим указанные столбцы к числовому типу и проверяет наличие NaN
Проверяем значения вне диапазона, при необходимости удаляем строки или помечаем их как NaN
Аргументы:
    df - DataFrame для проверки
    cols - список проверяемых столбцов
    drop_na - удалять ли некорректные строки
    min_val, max_val - допустимые границы значений
Возвращаем обновлённый DataFrame
"""
# ----------------- Проверка NaN и диапазона -----------------
def CheckNaN(df, cols, drop_na, min_val=None, max_val=None):
    import numpy as np
    df = df.copy()

    # приводим к числовому типу
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    # ищем NaN после приведения
    # n_bad - количество строк, где есть нечисловое или отсутствующее значение в столбцах
    n_bad = int(df[cols].isna().any(axis=1).sum())
    if n_bad:
        msg = f"[Предупреждение] Обнаружены строки с NaN в {cols}: {n_bad}"
        if drop_na:
            print(msg + " — строки будут удалены.")
            df = df.dropna(subset=cols)
        else:
            print(msg + " — строки не удаляются. Чтобы удалить, запустить с флагом --drop-na.")

    # фильтрация по диапазону
    # low - нижняя граница
    # high - верхняя граница
    # count - счётчик строк, которые вне диапазона
    if min_val is not None or max_val is not None:
        low = min_val if min_val is not None else -np.inf
        high = max_val if max_val is not None else np.inf
        out_of_range = ~df[cols].apply(lambda s: s.between(low, high)).all(axis=1)
        count = int(out_of_range.sum())
        if count:
            print(f"[Предупреждение] Значение вне диапазона [{low}, {high}]: {count}.")
            if drop_na:
                df = df[~out_of_range] # удаляем строки
            else:
                df.loc[out_of_range, cols] = pd.NA # помечаем как NaN
    return df

"""
Создаём новый столбец, содержащий разницу между col_b и col_a.
Аргументы:
    df — исходный DataFrame;
    col_a, col_b — имена числовых столбцов;
    new_col — имя нового столбца.
Возвращаем DataFrame с добавленным столбцом разницы
"""
# ----------------- Расчёт нового признака -----------------
def DiffHeight(df, col_a, col_b, new_col):
    df = df.copy()
    df[new_col] = df[col_b] - df[col_a]  # разница между столбцами
    return df


"""
Строим гистограмму распределения значений выбранного столбца
Автоматически подбираем количество корзин(диапозон) по правилу Фридмана–Дьякониса
Сохраняем график в PNG при наличии аргумента save_path
Аргументы:
    df — DataFrame с данными
    col — имя числового столбца
    save_path — путь для сохранения png
    rwidth — ширина столбцов
"""
# ----------------- Построение гистограммы -----------------
# values - это список всех числовых значений из выбранного столбца col, без пропусков
# n — количество элементов в выборке, например все значения роста
# h — оптимальная ширина одной «корзины» aka диапазона, bins - их количество
# counts — сколько значений попало в каждую корзину 
# edges — границы корзин
def Histogram(df, col, save_path=None):
    import numpy as np

    values = df[col].dropna()
    n = len(values)
    if n == 0:
        print("[Предупреждение] Нет данных для гистограммы")
        return

    # подбор числа корзин aka диапозона
    q75, q25 = values.quantile(0.75), values.quantile(0.25)
    iqr = q75 - q25
    if iqr > 0:
        h = 2 * iqr / (n ** (1/3))
        bins = max(5, int((values.max() - values.min()) / h)) if h > 0 else int(np.sqrt(n))
    else:
        bins = int(np.sqrt(n))

    # создаём полотно 7×4 дюйма для гистограммы
    fig, ax = plt.subplots(figsize=(7, 4))
    counts, edges, _ = ax.hist(values, bins=bins, edgecolor="black", alpha=0.8, rwidth=0.3, align="mid")

    # выравнивание и округление подписей оси X
    centers = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{x:.2f}" for x in centers])

    ax.set_title(f"Гистограмма: {col}", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel("Частота", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    # сохранение графика
    if save_path:
        try:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"График сохранён в: {save_path}")
        finally:
            plt.close()
    else:
        plt.close()

"""
Главная функция программы
Выполняет этапы:
    1. Парсинг аргументов
    2. Загрузка таблицы
    3. Проверка и очистка данных
    4. Расчёт нового признака
    5. Построение гистограммы
    6. Вывод результата
"""
# ----------------- Основная функция -----------------
def main():
    args = ParseArgs()

    df = LoadTable(args.path, args.sep)
    CheckColumns(df, [args.col_a, args.col_b, args.id_col])

    df = CheckNaN(df, [args.col_a, args.col_b],
                  drop_na=args.drop_na,
                  min_val=args.min_val,
                  max_val=args.max_val)

    df = DiffHeight(df, args.col_a, args.col_b, args.new_col)
    Histogram(df, args.new_col, save_path=args.save_plot)

    # вывод таблицы
    cols_to_show = [c for c in [args.id_col, args.col_a, args.col_b, args.new_col] if c and c in df.columns]
    if cols_to_show:
        print("\nРезультат:")
        print(df[cols_to_show].to_string(index=False))

if __name__ == "__main__":
    main()
