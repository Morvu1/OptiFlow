import pandas as pd
import numpy as np
import calendar
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------
# Параметры файла
# -------------------------
INPUT_FILE = "Rare.xlsx"   # входной файл (широкая таблица: Дата, product1, product2, ...)
OUTPUT_EXCEL = "forecast_results.xlsx"
OUTPUT_PDF = "forecast_charts.pdf"
FIGURES_DIR = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------
# Загрузить данные
# -------------------------
df = pd.read_excel(INPUT_FILE)
if "Дата" not in df.columns and "date" in df.columns:
    df.rename(columns={"date": "Дата"}, inplace=True)

df["Дата"] = pd.to_datetime(df["Дата"])
product_cols = [c for c in df.columns if c != "Дата"]
if not product_cols:
    raise SystemExit("В файле нет колонок товаров (только 'Дата').")

# -------------------------
# Функции признаков
# -------------------------
def add_date_features(df_in):
    df = df_in.copy()
    df["Год"] = df["Дата"].dt.year
    df["Месяц"] = df["Дата"].dt.month
    df["День"] = df["Дата"].dt.day
    df["Выходной"] = (df["Дата"].dt.weekday >= 5).astype(int)  # 1 если суб/вс
    # базовые праздничные месяцы (можно изменить/расширить позже)
    df["Праздники"] = df["Месяц"].isin([11, 12, 1]).astype(int)
    return df

df = add_date_features(df)

# -------------------------
# Ввод от пользователя (консоль)
# -------------------------
print("\n=== Ввод параметров прогноза ===")
# Горизонт прогноза (в месяцах)
forecast_months = int(input("Сколько месяцев прогнозировать вперед (целое число, например 1): ").strip())
# Реклама
ads_pct = float(input("Рост клиентов из-за рекламы (в %, 0 если нет): ").strip() or 0) / 100.0
# Изменение цен (в процентах). Положительное = повышение цен (обычно снижает продажи), отрицательное = скидка
price_change_pct = float(input("Изменение цен (в %, например -5 для скидки): ").strip() or 0) / 100.0
# Акции компании
promo_active = input("Есть ли локальные акции компании? (y/n): ").strip().lower() == "y"

# Сезонность: спрос зависит от месяца?
global_seasonal = input("Зависит ли популярность товаров от времени года глобально? (y/n): ").strip().lower() == "y"
global_season_months = []
if global_seasonal:
    s = input("В какие месяцы глобально спрос повышен (через запятую, например 11,12): ").strip()
    global_season_months = [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]

# Сезонность по товару (дополнительно)
per_product_season = {}
print("\nЕсли некоторым товарам нужна своя сезонность, укажи для них отдельные месяцы.")
for p in product_cols:
    ans = input(f"У товара '{p}' есть своя сезонность? (y/n): ").strip().lower()
    if ans == "y":
        s = input(f"В какие месяцы спрос на '{p}' повышен (напр. 6,7,12): ").strip()
        months = [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
    else:
        months = []
    per_product_season[p] = months

# Новые модели конкурентов — спрашиваем по каждому продаваемому товару:
# (т.е. если у нас есть iphone, спрашиваем вышел ли новый iphone и его популярность)
new_model_pop = {}
print("\nУкажи, для каждого продаваемого товара — вышел ли новый конкурент и его популярность (0.0 - 1.0).")
for p in product_cols:
    ans = input(f"Вышел ли новый конкурент для '{p}'? (y/n): ").strip().lower()
    if ans == "y":
        pop = float(input(f"Популярность нового телефона для '{p}' (0.0 - 1.0): ").strip())
        pop = max(0.0, min(1.0, pop))
    else:
        pop = 0.0
    new_model_pop[p] = pop

# Доп. корректировки (коэффициенты можно расширять)
# Эффект от выхода нового конкурента: максимально -30% умножается на популярность
max_competitor_drop = 0.30
# Эффект от акций компании
promo_factor = 1.10 if promo_active else 1.0

# -------------------------
# Подготовка будущих дат (по дням)
# -------------------------
last_date = df["Дата"].max()
# Начало следующего дня после last_date
start_date = (last_date + pd.Timedelta(days=1)).replace(day=1) if last_date.day != 1 else last_date + pd.offsets.MonthBegin(1)
# строим список дат на forecast_months месяцев вперёд (по дням)
future_dates = []
current = start_date
for _ in range(forecast_months):
    year = current.year
    month = current.month
    days = calendar.monthrange(year, month)[1]
    month_dates = pd.date_range(start=current, periods=days, freq="D")
    future_dates.extend(month_dates)
    # перейти к следующему месяцу
    current = (current + pd.offsets.MonthBegin(1)).replace(day=1)

future = pd.DataFrame({"Дата": future_dates})
future = add_date_features(future)

# Добавляем глобальную и per-product сезонность признаки
if global_season_months:
    future["Глобальная_Сезонность"] = future["Месяц"].apply(lambda m: 1 if m in global_season_months else 0)
else:
    future["Глобальная_Сезонность"] = 0

# -------------------------
# Обучение моделей и прогноз
# -------------------------
summary = []  # для Excel: строки с итогами по каждому месяцу и продукту
product_daily_forecasts = {}  # по товарам: дата -> прогноз дневной

for p in product_cols:
    # Признаки для обучения
    X = df[["Год", "Месяц", "День", "Выходной", "Праздники"]].copy()
    # Добавим сезонность по товару
    df["Сезонность_Товара"] = df["Месяц"].apply(lambda m, pm=per_product_season[p]: 1 if m in pm else 0)
    X["Сезонность_Товара"] = df["Сезонность_Товара"].values
    # Целевая переменная
    y = df[p].fillna(0).values

    # Обучаем LinearRegression
    model = LinearRegression()
    model.fit(X, y)

    # Формируем X для будущих дат
    Xf = future[["Год", "Месяц", "День", "Выходной", "Праздники"]].copy()
    Xf["Сезонность_Товара"] = future["Месяц"].apply(lambda m, pm=per_product_season[p]: 1 if m in pm else 0).values

    # Предсказания по дням
    daily_pred = model.predict(Xf)
    # Если модель дала отрицательное — обрезаем в 0
    daily_pred = np.where(daily_pred < 0, 0.0, daily_pred)

    # Применяем пользовательские корректировки:
    adjust = 1.0
    adjust *= (1.0 + ads_pct)              # реклама
    # price_change_pct: если положительное — повышение цен, естественно может уменьшать продажи.
    # интерпретируем так: увеличение цен на +10% -> продажи уменьшаются на 10% (умножаем на (1 - price_change_pct))
    adjust *= max(0.0, 1.0 - price_change_pct)
    adjust *= promo_factor                 # акции
    adjust *= max(0.0, 1.0 - max_competitor_drop * new_model_pop.get(p, 0.0))  # новый конкурент

    daily_pred_adjusted = daily_pred * adjust

    # Суммирование по месяцам в горизонте
    future_df = future[["Дата", "Год", "Месяц"]].copy()
    future_df[f"{p}_pred"] = daily_pred_adjusted
    product_daily_forecasts[p] = future_df[["Дата", f"{p}_pred"]].copy()

    # Сводка по месяцам
    monthly = future_df.groupby(["Год", "Месяц"])[f"{p}_pred"].sum().reset_index()
    monthly["product"] = p
    monthly.rename(columns={f"{p}_pred": "predicted_sales"}, inplace=True)
    for _, row in monthly.iterrows():
        summary.append({
            "product": p,
            "year": int(row["Год"]),
            "month": int(row["Месяц"]),
            "predicted_sales": float(row["predicted_sales"])
        })

    # Строим график "прошлые продажи (последние N дней) + прогноз (будущие дни)"
    # Возьмём последние 90 дней для отображения истории, если есть
    hist_days = 90
    hist_df = df[["Дата", p]].copy()
    hist_df = hist_df.sort_values("Дата")
    hist_recent = hist_df[hist_df["Дата"] >= (last_date - pd.Timedelta(days=hist_days))]
    # Если мало данных, возьмём всё
    if hist_recent.shape[0] < 2:
        hist_recent = hist_df

    # График
    plt.figure(figsize=(10, 5))
    plt.plot(hist_recent["Дата"], hist_recent[p], label="История (реальные продажи)")
    plt.plot(product_daily_forecasts[p]["Дата"], product_daily_forecasts[p][f"{p}_pred"],
             label="Прогноз (предсказанные ежедневные продажи)", linestyle="--")
    plt.title(f"Продажи: {p} — история + прогноз ({forecast_months} мес.)")
    plt.xlabel("Дата")
    plt.ylabel("Штук в день")
    plt.legend()
    plt.grid(alpha=0.3)
    fig_path = os.path.join(FIGURES_DIR, f"{p}_forecast.png")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

# -------------------------
# Подготовка итоговой таблицы summary
# -------------------------
summary_df = pd.DataFrame(summary)
# Pivot: строки = (year, month), колонки = product
if not summary_df.empty:
    pivot = summary_df.pivot_table(index=["year", "month"], columns="product", values="predicted_sales", aggfunc="sum")
    pivot = pivot.reset_index().sort_values(["year", "month"])
else:
    pivot = pd.DataFrame()

# -------------------------
# Экспорт в Excel
# -------------------------
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    # Summary sheet
    if not pivot.empty:
        pivot.to_excel(writer, sheet_name="summary", index=False)
    else:
        pd.DataFrame().to_excel(writer, sheet_name="summary", index=False)

    # Каждый продукт — лист с дневным прогнозом
    for p, dfp in product_daily_forecasts.items():
        dfp.rename(columns={f"{p}_pred": "predicted_daily"}, inplace=True)
        dfp.to_excel(writer, sheet_name=f"{p}_daily", index=False)

print(f"\nРезультаты экспортированы в Excel: {OUTPUT_EXCEL}")

# -------------------------
# Экспорт графиков в PDF
# -------------------------
with PdfPages(OUTPUT_PDF) as pdf:
    # Можно положить сначала сводную таблицу как текст-страницу
    try:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
        ax.axis("off")
        if not pivot.empty:
            table = ax.table(cellText=pivot.round(0).values,
                             colLabels=pivot.columns,
                             loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            ax.set_title("Сводка прогноза (помесячно, штук)")
        else:
            ax.text(0.5, 0.5, "Нет данных для сводки", ha="center")
        pdf.savefig(fig)
        plt.close(fig)
    except Exception:
        pass
1122
    # Добавляем все PNG-графики в PDF
    for p in product_cols:
        fig_path = os.path.join(FIGURES_DIR, f"{p}_forecast.png")
        if os.path.exists(fig_path):
            fig = plt.figure()
            img = plt.imread(fig_path)
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

print(f"Графики и отчёт сохранены в PDF: {OUTPUT_PDF}")

# -------------------------
# Вывод в консоль
# -------------------------
print("\n=== КОНСОЛЬНЫЙ ОТЧЕТ (итог по месяцам) ===")
if not pivot.empty:
    print(pivot.to_string(index=False, float_format="%.0f"))
else:
    print("Нет результатов для отображения.")

print("\nГотово — файлы:")
print(f" - Excel: {OUTPUT_EXCEL}")
print(f" - PDF:   {OUTPUT_PDF}")
print(f" - PNG-файлы в папке: {FIGURES_DIR}")