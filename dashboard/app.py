"""Dashboard del Proyecto Integrador 2.21: pronóstico multihorizonte de Bitcoin con MLP.

Lee solo los resultados ya calculados en ../results y las figuras de ../notebooks/figs (no entrena nada).
Si la variable de entorno API_URL apunta a la API desplegada (app/api.py), la sección "Probar el modelo"
consulta en vivo los modelos por horizonte.

Local:      python dashboard/app.py   -> http://127.0.0.1:8050
Render:     gunicorn --chdir dashboard app:server
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
from flask import abort, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = ROOT / "notebooks" / "figs"
APP_DIR = ROOT / "app"
BOOK = "https://aduartech81.github.io/deep-learning-02-forecasting-mlp/"
REPO = "https://github.com/aduartech81/deep-learning-02-forecasting-mlp"
API_URL = os.environ.get("API_URL", "").rstrip("/")

# ----------------------------------------------------------------------------- estilo
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7", "#ffffff"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
STRAT = {"forwardChaining": ("Forward Chaining", S[0]), "kFold": ("K-Fold", S[1]), "groupKFold": ("Group K-Fold", S[2])}
MODEL = {"mlp": ("MLP Keras", S[0]), "sklearn": ("MLPRegressor", S[3]), "naive": ("Ingenuo", MUTED)}
LAG_COLOR = {7: SEQ[1], 14: SEQ[2], 21: SEQ[3], 28: SEQ[4]}
HS, LAGS = [7, 14, 21, 28], [7, 14, 21, 28]


def base_layout(fig, height=380, **kw):
    fig.update_layout(
        template="plotly_white", height=height, font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=INK2),
        margin=dict(l=10, r=10, t=16, b=10), paper_bgcolor=SURF, plot_bgcolor=SURF, separators=",.",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=12)), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=BASE, zeroline=False, ticks="outside", tickcolor=BASE)
    fig.update_yaxes(gridcolor=GRID, linecolor=BASE, zeroline=False)
    return fig


def fmt(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def graph(fig, gid=None):
    kw = {"id": gid} if gid else {}
    return dcc.Graph(figure=fig, style={"height": f"{fig.layout.height or 380}px"}, config={"displayModeBar": False}, **kw)


def kpi(value, label, accent=False):
    return html.Div([html.Div(value, className="kpi-value" + (" accent" if accent else "")),
                     html.Div(label, className="kpi-label")], className="kpi")


def section_head(eyebrow, title, lead):
    return html.Div([html.Div(eyebrow, className="eyebrow"), html.H1(title), html.P(lead, className="lead")],
                    className="section-head")


def card(title, body, note=None):
    return html.Div([html.H3(title), body] + ([html.P(note, className="note")] if note else []), className="viz-card")


def figure_img(name, caption, iid=None):
    kw = {"id": iid} if iid else {}
    return html.Figure([html.Img(src=f"/figs/{name}", alt=caption, **kw), html.Figcaption(caption)], className="static-fig")


def table(df, page_size=12):
    return dash_table.DataTable(
        data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns], page_size=page_size,
        sort_action="native", style_as_list_view=True, style_table={"overflowX": "auto"},
        style_header={"fontWeight": 600, "backgroundColor": "#f6f6f3", "color": INK, "border": "none",
                      "fontFamily": "Inter, system-ui, sans-serif", "fontSize": 12.5},
        style_cell={"fontFamily": "Inter, system-ui, sans-serif", "fontSize": 13, "padding": "7px 10px", "color": INK2,
                    "textAlign": "left", "borderBottom": f"1px solid {GRID}", "whiteSpace": "normal", "height": "auto"})


# ----------------------------------------------------------------------------- datos
serie = pd.read_csv(RES / "serie_modelado.csv", parse_dates=["Date"])
part = pd.read_csv(RES / "particion.csv")
pruebas = pd.read_csv(RES / "eda_pruebas.csv")
best = pd.read_csv(RES / "mejor_arquitectura.csv")
top5 = pd.read_csv(RES / "top5_arquitecturas.csv")
efecto = pd.read_csv(RES / "efecto_topologia.csv")
cons = pd.read_csv(RES / "consolidado_lag_h.csv")
pfold = pd.read_csv(RES / "cv_metricas_por_fold.csv")
ph = pd.read_csv(RES / "cv_metricas_por_h.csv")
vs = pd.read_csv(RES / "mlp_vs_referencias.csv")
integ = pd.read_csv(RES / "cv_integridad.csv")
hold16 = pd.read_csv(RES / "holdout_16_configuraciones.csv")
dep = pd.read_csv(RES / "modelos_desplegados.csv")
bds = pd.read_csv(RES / "bds_validacion.csv")
vol_h = pd.read_csv(RES / "volatilidad_holdout.csv")
card_json = json.loads((APP_DIR / "model_card.json").read_text(encoding="utf8"))
n_cv = int(part.iloc[0]["días"])
cut = serie.Date.iloc[n_cv]
n_redes = sum(len(pd.read_csv(p)) for p in RES.glob("architecture_search_lag*_h*.csv"))
fc = vs[vs.estrategia == "forwardChaining"]
mape = {m: fc[fc.modelo == m].set_index("H").MAPE.reindex(HS) for m in MODEL}
gana_hold = int((dep["RMSE hold-out"] < dep["RMSE ingenuo hold-out"]).sum())


# ----------------------------------------------------------------------------- páginas
def page_contexto():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.Date[:n_cv], y=serie.Close[:n_cv], name="Serie de validación cruzada",
                             line=dict(color=S[0], width=1.6), hovertemplate="%{x|%d %b %Y}: %{y:,.0f} USD<extra></extra>"))
    fig.add_trace(go.Scatter(x=serie.Date[n_cv:], y=serie.Close[n_cv:], name="Hold-out final (reservado)",
                             line=dict(color=S[2], width=1.8), hovertemplate="%{x|%d %b %Y}: %{y:,.0f} USD<extra></extra>"))
    fig.add_vrect(x0=cut, x1=serie.Date.iloc[-1], fillcolor=S[2], opacity=0.08, line_width=0)
    base_layout(fig, height=340)
    fig.update_yaxes(type="log", title="Cierre diario (USD, escala log)", tickvals=[3000, 10000, 30000, 100000],
                     ticktext=["3 mil", "10 mil", "30 mil", "100 mil"])
    qs = [
        ("¿El MLP le gana a «mañana igual que hoy»?", f"No en el precio: pierde en los 4 horizontes del hold-out (MAPE "
         f"{fmt(dep['MAPE hold-out'].mean())} % frente a {fmt(dep['MAPE ingenuo hold-out'].mean())} % del ingenuo)."),
        ("¿Qué validación es realista?", "Forward Chaining: entrena siempre con el pasado. K-Fold es optimista y Group K-Fold "
         "la peor estimación."),
        ("¿Qué arquitectura gana?", "Redes de 2–3 capas anchas (256 neuronas) con Dropout en la última capa; con H largo "
         "casi ningún candidato queda balanceado."),
        ("¿Y la volatilidad?", f"Ahí sí: el MLP supera al ingenuo en RMSE en {int(vol_h.MLP_mejor_que_ingenuo.sum())} de 16 "
         "configuraciones del hold-out."),
    ]
    return html.Div([
        html.Div([
            html.Div([
                html.Div("Andrés Duarte Chaves · Deep Learning · Maestría en Ingeniería Biomédica, Uninorte", className="eyebrow"),
                html.H1("¿Se puede predecir Bitcoin solo con su pasado?", className="hero-title"),
                html.P(["Una red neuronal MLP multisalida pronostica el cierre diario de BTCUSDT a ",
                        html.B("7, 14, 21 y 28 días"), " con ventanas de 7 a 28 días previos. Se comparan tres esquemas de "
                        "validación temporal, se prueban ", html.B(f"{n_redes:,} redes".replace(",", ".")),
                        " y todo se mide contra la referencia más simple posible: repetir el último precio."], className="lead"),
                html.Div([html.Span(t, className="chip") for t in ["Series de tiempo", "Pronóstico multihorizonte",
                                                                   "Validación cruzada temporal", "MLOps"]], className="chips"),
                html.Div([dbc.Button("Ver el resultado principal →", href="/pronostico", className="btn-primary-soft"),
                          dbc.Button("Jupyter Book completo", href=BOOK, target="_blank", className="btn-ghost")],
                         className="hero-actions"),
            ], className="hero-text"),
            html.Div(html.Img(src="/assets/hero.svg", alt="Ilustración: serie de precios, ventana de entrada y red neuronal"),
                     className="hero-art"),
        ], className="hero"),
        html.Div([
            kpi(f"{len(serie) + 1:,}".replace(",", "."), f"velas diarias de BTCUSDT ({serie.Date.iloc[0]:%Y} → {serie.Date.iloc[-1]:%b %Y})"),
            kpi(f"{n_redes:,}".replace(",", "."), "redes entrenadas en la búsqueda de arquitectura (16 configuraciones)"),
            kpi(f"{gana_hold} de 4", "horizontes en que el MLP le gana al pronóstico ingenuo (hold-out, precio)", accent=True),
            kpi(f"{int(vol_h.MLP_mejor_que_ingenuo.sum())} de 16", "configuraciones en que sí le gana para la volatilidad"),
        ], className="kpi-row"),
        card("La serie y el tramo reservado", graph(fig),
             f"El hold-out ({part.iloc[1]['desde']} → {part.iloc[1]['hasta']}, {part.iloc[1]['días']} días) no se tocó hasta la "
             "evaluación final: ni validación cruzada, ni búsqueda de arquitectura, ni escaladores."),
        html.H2("Lo que se encontró", className="sub"),
        html.Div([html.Div([html.P(q, className="q-text"), html.P(a, className="q-answer")], className="q-card")
                  for q, a in qs], className="q-grid"),
    ])


def page_datos():
    r = serie.LogReturn.dropna()
    fh = go.Figure(go.Histogram(x=r, nbinsx=150, marker_color=S[0], hovertemplate="retorno %{x:.3f}: %{y} días<extra></extra>"))
    xs = np.linspace(r.min(), r.max(), 300)
    pdf = np.exp(-0.5 * ((xs - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi)) * len(r) * (r.max() - r.min()) / 150
    fh.add_trace(go.Scatter(x=xs, y=pdf, line=dict(color=INK2, width=1.5), name="Normal equivalente", hoverinfo="skip"))
    base_layout(fh, height=340, showlegend=False, bargap=0.02)
    fh.update_xaxes(title="Retorno logarítmico diario", range=[-0.2, 0.2]); fh.update_yaxes(title="Días")
    fv = go.Figure(go.Scatter(x=serie.Date, y=serie.Volatility, line=dict(color=S[1], width=1.4),
                              hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra></extra>"))
    base_layout(fv, height=340); fv.update_yaxes(title="Volatilidad anualizada (30 d)")
    t = pruebas.copy()
    t["p_valor"] = t.p_valor.map(lambda p: "< 1e-15" if p < 1e-15 else f"{p:.2g}")
    t["estadístico"] = t["estadístico"].round(2)
    return html.Div([
        section_head("Datos", "Precios sin memoria útil, volatilidad con memoria",
                     "El nivel del precio no es estacionario; los retornos sí, pero con colas pesadas y episodios de volatilidad "
                     "que se agrupan. Esa diferencia anticipa todo el resultado: la dirección del precio casi no se puede "
                     "anticipar con su pasado, la magnitud de sus movimientos sí."),
        html.Div([card("Retornos diarios: colas mucho más pesadas que la normal", graph(fh),
                       "Curtosis en exceso de 18: los días extremos son mucho más frecuentes de lo que predice una normal."),
                  card("Volatilidad histórica realizada", graph(fv),
                       "Se agrupa en episodios (2018, 2020, 2022): el objetivo alterno del capítulo 7.")], className="grid-2"),
        card("Pruebas estadísticas", table(t[["prueba", "serie", "estadístico", "p_valor", "decisión (α = 0,05)"]])),
        card("Autocorrelación del precio, de los retornos y de sus cuadrados", figure_img("01_acf.png", "Capítulo 1 del Jupyter Book")),
    ])


def page_validacion():
    sm = integ.groupby("estrategia").agg(train=("ventanas_train", "mean"), val=("ventanas_val", "mean"),
                                         sin_purga=("purgado", lambda s: int((~s).sum())),
                                         pasado=("train_todo_en_el_pasado", "all")).reset_index()
    sm["estrategia"] = sm.estrategia.map(lambda s: STRAT[s][0])
    sm = sm.rename(columns={"train": "ventanas de entrenamiento (media)", "val": "ventanas de validación (media)",
                            "sin_purga": "folds con fuga (sin purga posible)", "pasado": "entrena solo con el pasado"}).round(0)
    controls = html.Div([html.Div([html.Label("Lag"), dcc.Dropdown(id="v-lag", options=LAGS, value=7, clearable=False)], className="ctl"),
                         html.Div([html.Label("Horizonte H"), dcc.Dropdown(id="v-h", options=HS, value=7, clearable=False)], className="ctl")],
                        className="controls")
    return html.Div([
        section_head("Validación", "Tres formas de validar, tres respuestas distintas",
                     "Las tres estrategias de timeseries-cv se aplicaron con el mismo modelo. Solo Forward Chaining imita el uso "
                     "real (entrenar con el pasado, predecir el futuro). K-Fold valida los mismos días pero entrena también con "
                     "el futuro; Group K-Fold intercala ventanas por toda la serie."),
        card("Qué tramo usa cada estrategia", figure_img("02_esquema_estrategias.png",
                                                         "Azul: entrenamiento; naranja: validación; verde: hold-out final")),
        card("Integridad de los folds", table(sm)),
        card("Error por fold: Forward Chaining frente a K-Fold", html.Div([controls, dcc.Graph(id="v-fold", style={"height": "360px"},
                                                                                            config={"displayModeBar": False})]),
             "Mismas ventanas de validación. Donde K-Fold queda por debajo, su error es optimista: aprendió del futuro. Su último "
             "fold coincide con el de Forward Chaining porque ahí no hay futuro disponible."),
    ])


def page_arquitectura():
    mat = best.pivot(index="lag", columns="H", values="best_val_rmse")
    txt = best.pivot(index="lag", columns="H", values="arquitectura")
    fh = go.Figure(go.Heatmap(z=mat.values, x=[f"H = {h}" for h in mat.columns], y=[f"lag {l}" for l in mat.index],
                              colorscale=[[0, SEQ[0]], [1, SEQ[4]]], text=txt.values, texttemplate="%{text}",
                              textfont=dict(size=11), hovertemplate="%{y} · %{x}<br>%{text}<br>RMSE val (z): %{z:.3f}<extra></extra>",
                              colorbar=dict(title="RMSE val (z)", thickness=10)))
    base_layout(fh, height=360); fh.update_yaxes(autorange="reversed")
    e = efecto.copy()
    fe = go.Figure()
    for fac, col in zip(e.factor.unique(), [S[0], S[1], S[2]]):
        d = e[e.factor == fac]
        lab = {"capas": "capas", "ancho máx.": "neuronas", "capas con Dropout": "c/Dropout"}[fac]
        fe.add_bar(x=[f"{n} {lab}" for n in d.nivel], y=d["RMSE val relativo (mediana)"], name=fac, marker_color=col,
                   hovertemplate="%{x}<br>RMSE relativo al mejor: %{y:.2f}<extra></extra>")
    base_layout(fe, height=340); fe.update_yaxes(title="RMSE val / mejor de su config. (mediana)")
    t5 = top5.copy(); t5[["best_val_rmse", "gap_ratio", "holdout_rmse"]] = t5[["best_val_rmse", "gap_ratio", "holdout_rmse"]].round(3)
    return html.Div([
        section_head("Arquitectura", f"{n_redes:,} redes para elegir 16".replace(",", "."),
                     "Para cada combinación de lag y horizonte se probaron 584 topologías: 1 a 3 capas de 32 a 256 neuronas y "
                     "todas las combinaciones de Dropout. Se elige la de menor error de validación entre las que no sobreajustan "
                     "(gap_ratio < 0,15)."),
        html.Div([card("La mejor red por configuración", graph(fh),
                       "Color = error de validación de la red elegida; crece con el horizonte."),
                  card("Efecto de la topología (todas las redes)", graph(fe),
                       "En la mayoría de candidatos, más capas y más Dropout empeoran; las ganadoras son excepciones anchas.")],
                 className="grid-2"),
        card("Curvas de aprendizaje del Top-3", figure_img("03_curvas_top3.png", "Validación (continua) y entrenamiento (discontinua)"),
             "La validación se dispara después de pocas épocas mientras el entrenamiento sigue bajando: el tramo de validación "
             "(2024–2025) tiene precios por encima de todo lo visto en entrenamiento y la red no extrapola. Por eso las épocas "
             "óptimas son tempranas (3 a 12). Las celdas vacías son configuraciones sin ningún candidato balanceado."),
        html.H2("Top-5 de cada configuración", className="sub"), table(t5, 10),
    ])


def page_pronostico():
    fm = go.Figure()
    for m, (lab, col) in MODEL.items():
        fm.add_trace(go.Scatter(x=HS, y=mape[m].values, mode="lines+markers", name=lab, line=dict(color=col, width=2.2,
                                dash="dash" if m == "naive" else "solid"), marker=dict(size=9),
                                hovertemplate=f"{lab} · H = %{{x}}: MAPE %{{y:.2f}} %<extra></extra>"))
    base_layout(fm, height=360); fm.update_xaxes(title="Horizonte H (días)", tickvals=HS); fm.update_yaxes(title="MAPE de validación (%)")
    fhd = go.Figure()
    fhd.add_bar(x=[f"H = {h}" for h in dep.H], y=dep["MAPE hold-out"], name="MLP desplegado", marker_color=S[0],
                hovertemplate="%{x}: MAPE %{y:.2f} %<extra>MLP</extra>")
    fhd.add_bar(x=[f"H = {h}" for h in dep.H], y=dep["MAPE ingenuo hold-out"], name="Ingenuo", marker_color=MUTED,
                hovertemplate="%{x}: MAPE %{y:.2f} %<extra>Ingenuo</extra>")
    base_layout(fhd, height=360, barmode="group"); fhd.update_yaxes(title="MAPE en el hold-out (%)")
    ctl = html.Div([html.Div([html.Label("Lag"), dcc.Dropdown(id="p-lag", options=LAGS, value=7, clearable=False)], className="ctl"),
                    html.Div([html.Label("Horizonte H"), dcc.Dropdown(id="p-h", options=HS, value=7, clearable=False)], className="ctl")],
                   className="controls")
    h16 = hold16[["lag", "H", "arquitectura", "MAPE_hold", "MAPE_naive_hold", "RMSE_hold", "RMSE_naive_hold"]].copy()
    h16.columns = ["lag", "H", "arquitectura", "MAPE MLP (%)", "MAPE ingenuo (%)", "RMSE MLP (USD)", "RMSE ingenuo (USD)"]
    h16 = h16.round(2)
    return html.Div([
        section_head("Pronóstico", "«Mañana igual que hoy» le gana al MLP",
                     "En validación (Forward Chaining) y en el hold-out, el pronóstico ingenuo tiene menos error que el MLP en "
                     "todos los horizontes. Con solo su propio pasado, el precio de Bitcoin se comporta casi como un paseo "
                     "aleatorio: la red reproduce el último nivel con ruido y arrastra sesgo cuando el mercado cambia de régimen."),
        html.Div([
            kpi(f"{fmt(mape['mlp'].iloc[0])} %", "MAPE del MLP a 7 días (validación)"),
            kpi(f"{fmt(mape['naive'].iloc[0])} %", "MAPE del ingenuo a 7 días (validación)", accent=True),
            kpi(f"{gana_hold} de 4", "horizontes en que el MLP gana en el hold-out"),
            kpi(f"{fmt(mape['mlp'].iloc[2] / mape['mlp'].iloc[0], 1)}×", "crecimiento del error del MLP de H = 7 a H = 21"),
        ], className="kpi-row"),
        html.Div([card("Validación: error frente al horizonte", graph(fm), "Forward Chaining, promedio sobre los 4 lags."),
                  card("Hold-out: modelos desplegados frente al ingenuo", graph(fhd),
                       "Un modelo por horizonte, con el lag de menor error de validación.")], className="grid-2"),
        card("Explorar una configuración", html.Div([ctl, dcc.Graph(id="p-step", style={"height": "320px"}, config={"displayModeBar": False}),
                                                     html.Img(id="p-img", style={"width": "100%", "borderRadius": "8px", "marginTop": "8px"})]),
             "Arriba: RMSE por paso del horizonte en validación (MLP, MLPRegressor e ingenuo). Abajo: serie real frente a predicha "
             "en el mejor, el mediano y el peor fold, y en el hold-out."),
        html.H2("Hold-out de las 16 configuraciones", className="sub"), table(h16, 16),
    ])


def page_vol_res():
    v = vol_h.copy(); v["config"] = "lag " + v.lag.astype(str) + " · H " + v.H.astype(str)
    fv = go.Figure()
    fv.add_bar(x=v.config, y=v.RMSE_hold, name="MLP", marker_color=S[1], hovertemplate="%{x}: RMSE %{y:.4f}<extra>MLP</extra>")
    fv.add_bar(x=v.config, y=v.RMSE_naive, name="Ingenuo", marker_color=MUTED, hovertemplate="%{x}: RMSE %{y:.4f}<extra>Ingenuo</extra>")
    base_layout(fv, height=360, barmode="group"); fv.update_yaxes(title="RMSE en el hold-out (volatilidad)")
    b = bds.copy()
    share = (b.groupby("estrategia").folds_iid.sum() / b.groupby("estrategia").folds.sum() * 100).reset_index(name="pct")
    fb = go.Figure(go.Bar(x=share.estrategia, y=share.pct, marker_color=[S[0] if "Forward" in s else S[1] if s == "K-Fold" else S[2]
                                                                        for s in share.estrategia],
                          hovertemplate="%{x}: %{y:.1f} % de folds con residuos iid<extra></extra>"))
    base_layout(fb, height=300); fb.update_yaxes(title="% de folds con p BDS > 0,05", range=[0, 100])
    return html.Div([
        section_head("Volatilidad y residuos", "Donde sí hay señal: la volatilidad",
                     "Con la misma arquitectura, el objetivo cambia a la volatilidad realizada a 30 días. Aquí el MLP sí mejora al "
                     "ingenuo en RMSE en la mayoría de configuraciones. En el precio, en cambio, los residuos del MLP conservan "
                     "dependencia: el test BDS rechaza que sean ruido independiente casi siempre."),
        card("Volatilidad: MLP frente a ingenuo en el hold-out", graph(fv),
             "El MLP reduce los errores grandes (RMSE) aunque su error relativo (MAPE) es mayor en los periodos tranquilos."),
        html.Div([card("Test BDS en validación (precio)", graph(fb), "Casi ningún fold deja residuos compatibles con ruido iid."),
                  card("Residuos de los modelos desplegados en el hold-out", figure_img("06_residuos_holdout.png", "Capítulo 6"))],
                 className="grid-2"),
    ])


def page_api():
    rows = [{"H": int(h), "lags que espera": v["lags"], "arquitectura": v["arquitectura"], "Dropout": v["dropout_mask"],
             "RMSE val (USD)": round(v["RMSE_val_USD"]), "RMSE hold-out (USD)": round(v["RMSE_holdout_USD"])}
            for h, v in card_json["horizontes"].items()]
    live = html.Div([
        html.Div([html.Label("Horizonte H"), dcc.Dropdown(id="a-h", options=HS, value=7, clearable=False)], className="ctl"),
        dbc.Button("Pronosticar con los últimos cierres conocidos", id="a-go", className="btn-primary-soft", n_clicks=0),
    ], className="controls")
    return html.Div([
        section_head("API", "El modelo, listo para usarse",
                     "La API (FastAPI) recibe los últimos cierres y el horizonte, y devuelve H predicciones con la mejor red de "
                     "ese horizonte. Se distribuye como imagen Docker, con pruebas unitarias e integración continua en GitHub."),
        card("Modelos desplegados", table(pd.DataFrame(rows))),
        card("Probar el modelo", html.Div([live, dcc.Loading(html.Div(id="a-out"))]),
             "Consulta la API en vivo si está desplegada (variable API_URL); la primera llamada puede tardar mientras el servicio despierta."),
        card("Uso desde código", dcc.Markdown(
            "```bash\ncurl -X POST <API_URL>/predict -H 'Content-Type: application/json' \\\n"
            "     -d '{\"lags\": [ ... últimos cierres ... ], \"horizon\": 14}'\n```\n"
            "`GET /info` dice cuántos lags espera cada horizonte; `GET /docs` abre la documentación interactiva.")),
    ])


def page_conclusiones():
    items = [
        ("Lo que se encontró", [
            "Con solo su propio pasado, el precio de Bitcoin se comporta casi como un paseo aleatorio: el pronóstico ingenuo le "
            "gana al MLP en validación y en el hold-out, en los cuatro horizontes.",
            "Forward Chaining es la validación realista; K-Fold resulta optimista cuando el mercado cambia de régimen, y Group "
            "K-Fold da la peor estimación con la implementación de timeseries-cv.",
            "Las redes ganadoras son de 2–3 capas anchas con Dropout en la última capa; a horizontes de 21–28 días casi ningún "
            "candidato queda balanceado.",
            "La volatilidad sí tiene estructura aprovechable: el MLP supera al ingenuo en RMSE en 12 de 16 configuraciones.",
        ]),
        ("Hallazgos técnicos", [
            "timeseries-cv 0.1.5: firma posicional, un fold por ventana en Forward Chaining y K-Fold (miles de folds), y "
            "dependencia declarada numpy < 2 que choca con TensorFlow reciente.",
            "El test BDS no está en arch.bootstrap como sugiere la guía; se usó statsmodels.",
            "Keras mide el RMSE de entrenamiento con Dropout activo: el gap train–validación se infla al usar Dropout.",
        ]),
        ("Siguiente versión", [
            "Modelar retornos o diferencias logarítmicas en lugar del nivel.",
            "Agregar variables exógenas (volumen, otros activos, tasas).",
            "Reportar siempre la mejora relativa frente al ingenuo como métrica principal.",
        ]),
    ]
    return html.Div([
        section_head("Conclusiones", "Un resultado negativo bien medido vale más que uno positivo mal medido",
                     "El aporte del proyecto no es un predictor de Bitcoin, sino un protocolo que impide engañarse: hold-out "
                     "intocable, validación que respeta el tiempo, búsqueda exhaustiva y comparación contra la referencia más simple."),
        html.Div([html.Div([html.H3(t), html.Ul([html.Li(x) for x in xs])], className="viz-card") for t, xs in items], className="grid-3"),
        html.Div([html.H3("Material completo"),
                  html.P("El análisis paso a paso, con todo el código y sus salidas, está en el Jupyter Book. Ejercicio académico: "
                         "no es una recomendación de inversión."),
                  html.Div([dbc.Button("Jupyter Book del proyecto", href=BOOK, target="_blank", className="btn-primary-soft"),
                            dbc.Button("Repositorio en GitHub", href=REPO, target="_blank", className="btn-ghost")],
                           className="hero-actions")], className="viz-card cta"),
    ])


PAGES = {"/": ("Contexto", page_contexto), "/datos": ("Datos", page_datos), "/validacion": ("Validación", page_validacion),
         "/arquitectura": ("Arquitectura", page_arquitectura), "/pronostico": ("Pronóstico", page_pronostico),
         "/volatilidad": ("Volatilidad y residuos", page_vol_res), "/api": ("API", page_api),
         "/conclusiones": ("Conclusiones", page_conclusiones)}

app = dash.Dash(__name__, title="Pronóstico de Bitcoin con MLP · Deep Learning", suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP, "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
                           {"name": "description", "content": "Pronóstico multihorizonte de Bitcoin con MLP, validación temporal y API."}])
server = app.server


@server.route("/figs/<path:name>")
def figs(name):
    if not name.endswith(".png") or not (FIGS / name).exists():
        abort(404)
    return send_from_directory(FIGS, name, max_age=86400)


@server.route("/healthz")
def healthz():
    return "ok"


app.layout = html.Div([
    dcc.Location(id="url"),
    html.Header(html.Div([
        html.A([html.Span(className="brand-dot"), "BTC · MLP"], href="/", className="brand"),
        html.Nav(id="nav", className="topnav"),
        html.A("Jupyter Book ↗", href=BOOK, target="_blank", className="nav-book"),
    ], className="topbar-inner"), className="topbar"),
    html.Main(id="page", className="page"),
    html.Footer(html.Div([
        html.Span("Andrés Duarte Chaves · Proyecto Integrador 2.21 de Deep Learning · Maestría en Ingeniería Biomédica, Universidad del Norte · 2026"),
        html.Span(["Datos: API pública de Binance (BTCUSDT diario). ", html.A("Código", href=REPO, target="_blank")]),
    ], className="footer-inner"), className="footer"),
])


@app.callback(Output("page", "children"), Output("nav", "children"), Input("url", "pathname"))
def render(path):
    path = path if path in PAGES else "/"
    nav = [html.A(n, href=p, className="topnav-link" + (" active" if p == path else "")) for p, (n, _) in PAGES.items()]
    return PAGES[path][1](), nav


@app.callback(Output("v-fold", "figure"), Input("v-lag", "value"), Input("v-h", "value"))
def fold_fig(lag, H):
    d = pfold[(pfold.modelo == "mlp") & (pfold.lag == lag) & (pfold.H == H) & pfold.estrategia.isin(["forwardChaining", "kFold"])]
    fig = go.Figure()
    for st in ["forwardChaining", "kFold"]:
        x = d[d.estrategia == st].sort_values("fold")
        fig.add_bar(x=[f"Fold {k}" for k in x.fold], y=x.RMSE, name=STRAT[st][0], marker_color=STRAT[st][1],
                    hovertemplate="%{x}: RMSE %{y:,.0f} USD<extra>" + STRAT[st][0] + "</extra>")
    base_layout(fig, height=360, barmode="group"); fig.update_yaxes(title="RMSE de validación (USD)")
    return fig


@app.callback(Output("p-step", "figure"), Output("p-img", "src"), Input("p-lag", "value"), Input("p-h", "value"))
def step_fig(lag, H):
    d = ph[(ph.estrategia == "forwardChaining") & (ph.lag == lag) & (ph.H == H)]
    fig = go.Figure()
    for m, (lab, col) in MODEL.items():
        v = d[d.modelo == m].groupby("h").RMSE.mean()
        fig.add_trace(go.Scatter(x=v.index, y=v.values, mode="lines+markers", name=lab, line=dict(color=col, width=2,
                                 dash="dash" if m == "naive" else "solid"), hovertemplate=f"{lab} · h = %{{x}}: %{{y:,.0f}} USD<extra></extra>"))
    base_layout(fig, height=320); fig.update_xaxes(title="Paso h (días adelante)"); fig.update_yaxes(title="RMSE validación (USD)")
    return fig, f"/figs/05_series_lag{lag}_h{H}.png"


@app.callback(Output("a-out", "children"), Input("a-go", "n_clicks"), State("a-h", "value"), prevent_initial_call=True)
def live_predict(_, H):
    if not API_URL:
        return html.P("La API no está configurada para este tablero (falta API_URL).", className="note")
    import requests
    try:
        n = next(i["n_lags"] for i in requests.get(f"{API_URL}/info", timeout=60).json() if i["horizon"] == H)
        lags = card_json["ultimos_cierres"][-n:]
        r = requests.post(f"{API_URL}/predict", json={"lags": lags, "horizon": H}, timeout=60).json()
    except Exception as e:  # noqa: BLE001
        return html.P(f"No se pudo consultar la API: {e}", className="note")
    last = card_json["ultimo_cierre_conocido"]
    dates = pd.date_range(pd.Timestamp(last["fecha"]) + pd.Timedelta(days=1), periods=H)
    hist = serie.tail(60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.Date, y=hist.Close, name="Últimos 60 días (real)", line=dict(color=S[0], width=1.8)))
    fig.add_trace(go.Scatter(x=dates, y=r["prediction"], name=f"Pronóstico MLP ({H} días)", line=dict(color=S[1], width=2, dash="dash"),
                             hovertemplate="%{x|%d %b %Y}: %{y:,.0f} USD<extra></extra>"))
    fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[last["close"]] * 2, name="Ingenuo (último cierre)",
                             line=dict(color=MUTED, width=1.5, dash="dot")))
    base_layout(fig, height=340); fig.update_yaxes(title="USD")
    return html.Div([graph(fig), html.P(f"Entrada: los últimos {n} cierres hasta {last['fecha']}. Ejercicio académico: no es una "
                                        "recomendación de inversión.", className="note")])


if __name__ == "__main__":
    app.run(debug=False, port=8050)
