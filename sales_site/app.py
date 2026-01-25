# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import pandas as pd
from werkzeug.utils import secure_filename
import forecast  # наш модуль с функцией run_forecast

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"xlsx", "xls", "csv"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace_this_with_random_secret"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("Файл не выбран")
        return redirect(url_for("index"))
    f = request.files["file"]
    if f.filename == "":
        flash("Файл не выбран")
        return redirect(url_for("index"))
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        f.save(path)
        # читаем колонки
        if fname.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        if "Дата" not in df.columns and "date" in df.columns:
            df.rename(columns={"date":"Дата"}, inplace=True)
        product_cols = [c for c in df.columns if c != "Дата"]
        # сохраняем имя файла и cols в сессии через скрытое поле: передадим имя файла на следующую форму
        return render_template("configure.html", filename=fname, products=product_cols)
    else:
        flash("Неподдерживаемый формат файла")
        return redirect(url_for("index"))

@app.route("/run", methods=["POST"])
def run():
    # получаем загруженный файл
    filename = request.form.get("filename")
    if not filename:
        flash("Ошибка: файл не найден")
        return redirect(url_for("index"))
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(path):
        flash("Ошибка: файл отсутствует на сервере")
        return redirect(url_for("index"))

    # читаем данные
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    if "Дата" not in df.columns and "date" in df.columns:
        df.rename(columns={"date":"Дата"}, inplace=True)
    df["Дата"] = pd.to_datetime(df["Дата"])

    # параметры общего уровня
    forecast_months = int(request.form.get("forecast_months", 1))
    ads_pct = float(request.form.get("ads_pct", 0.0)) / 100.0
    price_change_pct = float(request.form.get("price_change_pct", 0.0)) / 100.0
    promo_active = request.form.get("promo_active") == "on"

    global_season_months = []
    gseason = request.form.get("global_season_months", "").strip()
    if gseason:
        global_season_months = [int(x.strip()) for x in gseason.split(",") if x.strip().isdigit()]

    # per-product season and new model pop (from dynamic fields)
    per_product_season = {}
    new_model_pop = {}
    products = request.form.get("products_list", "")
    if products:
        products = products.split(",")
        for p in products:
            key_season = f"season_{p}"
            key_new = f"new_{p}"
            key_pop = f"pop_{p}"
            months_raw = request.form.get(key_season, "").strip()
            months = [int(x.strip()) for x in months_raw.split(",") if x.strip().isdigit()] if months_raw else []
            per_product_season[p] = months
            has_new = request.form.get(key_new) == "on"
            pop = float(request.form.get(key_pop, 0.0)) if has_new else 0.0
            new_model_pop[p] = pop

    # запускаем прогноз
    res = forecast.run_forecast(df,
                                forecast_months=forecast_months,
                                ads_pct=ads_pct,
                                price_change_pct=price_change_pct,
                                promo_active=promo_active,
                                global_season_months=global_season_months,
                                per_product_season=per_product_season,
                                new_model_pop=new_model_pop)

    # сохраняем пути для отображения
    excel_path = res["excel_path"]
    pdf_path = res["pdf_path"]
    pivot = res["summary_pivot"]
    products_daily = res["product_daily_forecasts"]
    figures = res["figures"]

    # Для удобства передаём названия продуктов
    prod_names = list(products_daily.keys())

    # передаём всё в шаблон
    return render_template("results.html",
                           pivot=pivot.to_html(index=False, float_format="%.0f") if not pivot.empty else "",
                           excel_path=excel_path,
                           pdf_path=pdf_path,
                           figures=figures,
                           prod_names=prod_names)

@app.route("/download/<path:filename>")
def download_file(filename):
    # security: allow only outputs files
    safe_path = os.path.join("outputs", os.path.basename(filename))
    if os.path.exists(safe_path):
        return send_file(safe_path, as_attachment=True)
    # also allow pdfs from outputs
    safe_path2 = os.path.join("outputs", filename)
    if os.path.exists(safe_path2):
        return send_file(safe_path2, as_attachment=True)
    return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)
