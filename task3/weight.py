import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ----------/базовые проверки/----------
def ensureCols(df: pd.DataFrame, cols: list[str]):
    """Проверяет, что в DataFrame присутствуют все указанные столбцы"""
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Отсутвующие столбцы: {miss}")

# s: временная серия — результат приведения столбца к числам
def ensureNumeric(df: pd.DataFrame, cols: list[str], allow_nan: bool = False):
    """Преобразует выбранные столбцы в числовой формат и проверяет отсутствие NaN"""
    bad = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if not allow_nan and s.isna().any():
            bad.append(c)
        df[c] = s
    if bad:
        raise ValueError(f"В столбцах обнаружены нечисловые значения или NaN: {bad}")
    return df

def diff(a: float, b: float, eps: float = 1e-12) -> float:
    """Возвращает относительное различие между двумя числами в процентах"""
    return abs(a - b) / max(abs(a), abs(b), eps) * 100.0

# ----------/веса/----------
# by: столбец группировки
# g: объект группировки df.groupby(by) (удобно для transform)
# m: поэлементный вектор минимумов base_col внутри своей группы
# M: поэлементный вектор максимумов base_col внутри своей группы
# z: нормированное значение base_col в [0,1] внутри своей группы
def addWeights(df: pd.DataFrame, base_col: str, by: str, wmin: float, wmax: float) -> pd.DataFrame:
    """Вычисляет веса '__weight__' на основе столбца base_col внутри каждой группы by"""
    ensureCols(df, [base_col, by])
    if not (np.isfinite(wmin) and np.isfinite(wmax) and wmax > wmin):
        raise ValueError("Диапазон весов должен быть конечным и удовлетворять wmax>wmin")

    df = df.copy()
    df = ensureNumeric(df, [base_col])
    g = df.groupby(by, observed=True, sort=False)
    m = g[base_col].transform("min")
    M = g[base_col].transform("max")

    mid = (wmin + wmax) / 2.0
    z = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    mask = (M != m)
    z[mask] = (df[base_col][mask] - m[mask]) / (M[mask] - m[mask])
    df["__weight__"] = np.where(mask, wmin + z * (wmax - wmin), mid)

    sums = df.groupby(by, observed=True)["__weight__"].sum()
    if (sums <= 0).any() or (~np.isfinite(sums)).any():
        raise ValueError("Неположительные или некорректные суммы весов в некоторых группах")
    return df

# ----------/агрегаты/----------
# features: список числовых признаков для усреднения
# g: объект группировки df.groupby(by)
# w_products: покомпонентные произведения признаков на вес
# wmean: таблица взвешенных средних
# out: объединение обычных и взвешенных средних по группам
# f: текущее имя признака из features
# denom: знаменатель
# num: числитель
def Weighted_and_Mean(df: pd.DataFrame, by: str, features: list[str]) -> pd.DataFrame:
    """Сравнивает обычные и взвешенные средние по признакам features для каждой группы"""
    ensureCols(df, features + [by, "__weight__"])
    df = ensureNumeric(df, features)

    g = df.groupby(by, observed=True, sort=False)
    mean = g[features].mean().rename(columns=lambda c: f"{c}__mean")

    w_products = df[features].multiply(df["__weight__"], axis=0)
    num = w_products.groupby(df[by], observed=True, sort=False).sum()
    denom = df.groupby(by, observed=True, sort=False)["__weight__"].sum()
    wmean = num.div(denom, axis=0).rename(columns=lambda c: f"{c}__wmean")
    out = mean.join(wmean)

    records = []
    for grp, row in out.iterrows():
        for f in features:
            a = float(row[f"{f}__mean"])
            b = float(row[f"{f}__wmean"])
            records.append({
                "group": grp,
                "feature": f,
                "mean": round(a, 6),
                "wmean": round(b, 6),
                "diff_%": round(diff(a, b), 6)
            })
    return pd.DataFrame.from_records(records)

# f"__wmean_by_{by}": суффикс - чтобы показать взвешанное среднее по группе
# df_out: исходные строки + присоединённые групповые взвешенные средние
def attachWmeans(df: pd.DataFrame, by: str, features: list[str]) -> pd.DataFrame:
    """Добавляет к исходному DataFrame взвешенные средние по каждой группе"""
    ensureCols(df, features + [by, "__weight__"])
    df = ensureNumeric(df, features)

    w_products = df[features].multiply(df["__weight__"], axis=0)
    num = w_products.groupby(df[by], observed=True, sort=False).sum()
    den = df.groupby(by, observed=True, sort=False)["__weight__"].sum()
    wmeans = num.div(den, axis=0)
    wmeans.columns = [f"{c}__wmean_by_{by}" for c in wmeans.columns]

    df_out = df.merge(wmeans, left_on=by, right_index=True, how="left")
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[num_cols] = df_out[num_cols].round(6)
    return df_out


# ----------/чтение и запись CSV/----------
def readСsv(path: Path) -> pd.DataFrame:
    """Читает CSV-файл, проверяет его корректность и возвращает DataFrame"""
    if not path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError("Входной CSV пуст.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Ошибка при разборе CSV: {e}")
    if df.empty:
        raise ValueError("Входной файл не содержит данных")
    return df


# df_fmt: DataFrame после применения форматирования к числам
def safeСsv(df: pd.DataFrame, path: Path):
    """Сохраняет DataFrame в CSV с выравниванием чисел и пробелом после ';'"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    def fmt_float(x):
        """Возвращает float с шестью знаками после запятой (фиксированная длина)"""
        if isinstance(x, float):
            return f"{x:.6f}".rstrip("0")
        return x

    try:
        df_fmt = df.map(fmt_float)
    except AttributeError:
        df_fmt = df.applymap(fmt_float)

    from io import StringIO
    buf = StringIO()
    df_fmt.to_csv(buf, sep=";", index=False)
    text = buf.getvalue().replace(";", "; ")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    tmp.replace(path)

# ----------/аргументы командной строки/----------
def parseАrgs():
    """Разбирает аргументы командной строки для анализа взвешенных средних"""
    p = argparse.ArgumentParser(description="DA-3-23: сравнение обычных и взвешенных средних")
    p.add_argument("--in-data", required=True, help="входной CSV-файл с данными")
    p.add_argument("--group-col", required=True, help="столбец группировки")
    p.add_argument("--features", nargs="+", required=True, help="числовые признаки для анализа")
    p.add_argument("--weight-base-col", required=True, help="столбец, по которому рассчитываются веса")
    p.add_argument("--wmin", type=float, default=1.0, help="минимальный вес (по умолчанию 1.0)")
    p.add_argument("--wmax", type=float, default=5.0, help="максимальный вес (по умолчанию 5.0)")
    p.add_argument("--threshold", type=float, default=5.0, help="порог различия в процентах (по умолчанию 5)")
    p.add_argument("--out-data", default="weighted_features.csv", help="файл с добавленными признаками")
    p.add_argument("--out-report", default="compare_means.csv", help="файл отчёта со сравнением средних")
    return p.parse_args()

# a: объект с распарсенными аргументами
def validateАrgs(a):
    """Проверяет диапазоны и значения аргументов командной строки"""
    if a.wmin >= a.wmax:
        raise ValueError("wmin должен быть меньше wmax")
    if a.threshold < 0 or a.threshold > 100:
        raise ValueError("threshold должен быть в пределах [0, 100]")

# ----------/основной блок выполнения/----------
def main():
    """Основная функция: читает данные, рассчитывает веса и средние, сохраняет результаты"""
    args = parseАrgs()
    validateАrgs(args)

    df = readСsv(Path(args.in_data))
    ensureCols(df, [args.group_col, args.weight_base_col] + args.features)
    df = ensureNumeric(df, [args.weight_base_col] + args.features, allow_nan=False)

    if df[args.group_col].isna().any():
        raise ValueError(f"В столбце группы обнаружены пустые значения: '{args.group_col}'")

    df_w = addWeights(df, args.weight_base_col, args.group_col, args.wmin, args.wmax)
    cmp_df = Weighted_and_Mean(df_w, args.group_col, args.features)
    df_out = attachWmeans(df_w, args.group_col, args.features)

    safeСsv(df_out, Path(args.out_data))
    safeСsv(cmp_df, Path(args.out_report))

    k = int((cmp_df["diff_%"] > float(args.threshold)).sum())
    print(f"Rows = {len(df_out)} / Diff > {args.threshold}% = {k}")
    print(f"Saved:\n - {args.out_data}\n - {args.out_report}")
    sys.exit(0)

if __name__ == "__main__":
    main()
