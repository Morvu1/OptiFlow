# forecast.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import calendar
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime

os.makedirs("static/figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def add_date_features(df):
    df = df.copy()
    df["Год"] = df["Дата"].dt.year
    df["Месяц"] = df["Дата"].dt.month
    df["День"] = df["Дата"].dt.day
    df["Выходной"] = (df["Дата"].dt.weekday >= 5).astype(int)
    df["Праздники"] = df["Месяц"].isin([11,12,1]).astype(int)
    return df

def run_forecast(df,
                 forecast_months=1,
                 ads_pct=0.0,
                 price_change_pct=0.0,
                 promo_active=False,
                 global_season_months=None,
                 per_product_season=None,
                 new_model_pop=None):
    """
    df - DataFrame с колонками: 'Дата' + product columns
    возвращает: summary_df (month-product table), product_daily_forecasts dict,
                paths: excel_path, pdf_path, figures list
    """
    if global_season_months is None:
        global_season_months = []
    if per_product_season is None:
        per_product_season = {}
    if new_model_pop is None:
        new_model_pop = {}

    df = df.copy()
    df["Дата"] = pd.to_datetime(df["Дата"])
    df = add_date_features(df)
    product_cols = [c for c in df.columns if c != "Дата" and c not in ["Год","Месяц","День","Выходной","Праздники"]]

    last_date = df["Дата"].max()
    # дата начала следующего месяца
    next_month_start = (last_date + pd.offsets.MonthBegin(1)).replace(day=1)
    future_dates = []
    current = next_month_start
    for _ in range(forecast_months):
        year = current.year
        month = current.month
        days = calendar.monthrange(year, month)[1]
        month_dates = pd.date_range(start=current, periods=days, freq="D")
        future_dates.extend(month_dates)
        current = (current + pd.offsets.MonthBegin(1)).replace(day=1)
    future = pd.DataFrame({"Дата": future_dates})
    future = add_date_features(future)
    # global season feature if provided
    future["Глобальная_Сезонность"] = future["Месяц"].apply(lambda m: 1 if m in global_season_months else 0)

    max_competitor_drop = 0.30
    promo_factor = 1.10 if promo_active else 1.0

    summary = []
    product_daily_forecasts = {}
    figures = []

    for p in product_cols:
        # prepare training X, y
        tmp = df.copy()
        tmp["Сезонность_Товара"] = tmp["Месяц"].apply(lambda m: 1 if m in per_product_season.get(p, []) else 0)
        X = tmp[["Год","Месяц","День","Выходной","Праздники","Сезонность_Товара"]]
        y = tmp[p].fillna(0).values

        model = LinearRegression()
        model.fit(X, y)

        # future X
        Xf = future[["Год","Месяц","День","Выходной","Праздники"]].copy()
        Xf["Сезонность_Товара"] = future["Месяц"].apply(lambda m: 1 if m in per_product_season.get(p, []) else 0)

        daily_pred = model.predict(Xf)
        daily_pred = np.where(daily_pred < 0, 0.0, daily_pred)

        adjust = 1.0
        adjust *= (1.0 + ads_pct)
        adjust *= max(0.0, 1.0 - price_change_pct)
        adjust *= promo_factor
        adjust *= max(0.0, 1.0 - max_competitor_drop * new_model_pop.get(p, 0.0))

        daily_pred_adjusted = daily_pred * adjust

        future_df = future[["Дата","Год","Месяц"]].copy()
        future_df[f"{p}_pred"] = daily_pred_adjusted
        product_daily_forecasts[p] = future_df[["Дата", f"{p}_pred"]].copy()

        monthly = future_df.groupby(["Год","Месяц"])[f"{p}_pred"].sum().reset_index()
        monthly["product"] = p
        monthly.rename(columns={f"{p}_pred":"predicted_sales"}, inplace=True)
        for _, row in monthly.iterrows():
            summary.append({
                "product": p,
                "year": int(row["Год"]),
                "month": int(row["Месяц"]),
                "predicted_sales": float(row["predicted_sales"])
            })

        # график (последние 90 дней)
        hist = df[["Дата", p]].sort_values("Дата")
        hist_recent = hist[hist["Дата"] >= (last_date - pd.Timedelta(days=90))]
        if hist_recent.shape[0] < 2:
            hist_recent = hist

        plt.figure(figsize=(10,4))
        plt.plot(hist_recent["Дата"], hist_recent[p], label="История")
        plt.plot(product_daily_forecasts[p]["Дата"], product_daily_forecasts[p][f"{p}_pred"], '--', label="Прогноз")
        plt.title(f"{p} — история + прогноз")
        plt.xlabel("Дата")
        plt.ylabel("Штук в день")
        plt.legend()
        plt.grid(alpha=0.3)
        fig_path = f"static/figures/{p}_forecast.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        figures.append(fig_path)

    # итогные таблицы
    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        pivot = summary_df.pivot_table(index=["year","month"], columns="product", values="predicted_sales", aggfunc="sum").reset_index()
        pivot = pivot.sort_values(["year","month"])
    else:
        pivot = pd.DataFrame()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"outputs/forecast_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if not pivot.empty:
            pivot.to_excel(writer, sheet_name="summary", index=False)
        for p, dfd in product_daily_forecasts.items():
            dfd.rename(columns={f"{p}_pred":"predicted_daily"}, inplace=True)
            dfd.to_excel(writer, sheet_name=f"{p}_daily", index=False)

    # PDF (сводка + графики)
    pdf_path = f"outputs/forecast_{timestamp}.pdf"
    with PdfPages(pdf_path) as pdf:
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(11.69,8.27))
            ax.axis("off")
            table = ax.table(cellText=pivot.round(0).values, colLabels=pivot.columns, loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1,1.5)
            ax.set_title("Сводка прогноза (помесячно, штук)")
            pdf.savefig(fig)
            plt.close(fig)
        for fp in figures:
            if os.path.exists(fp):
                fig = plt.figure()
                img = plt.imread(fp)
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig(fig)
                plt.close(fig)

    return {
        "summary_pivot": pivot,
        "product_daily_forecasts": product_daily_forecasts,
        "excel_path": excel_path,
        "pdf_path": pdf_path,
        "figures": figures
    }
