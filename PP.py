# PP.py
# 실행: python PP.py
# 접속: http://127.0.0.1:8050/

from dataclasses import dataclass, asdict
import uuid

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate

# ESC 키 감지용 (pip install dash-extensions)
from dash_extensions import EventListener


# -----------------------------
# 데이터 모델
# -----------------------------
@dataclass
class Item:
    id: str
    kind: str          # building / road / fence / gate
    w: float
    h: float
    x: float           # center x
    y: float           # center y


KIND_META = {
    "1": ("building", "건물", "rgba(30, 144, 255, 0.35)", "rgba(30, 144, 255, 1.0)"),
    "2": ("road",     "도로", "rgba(50, 205, 50, 0.25)",  "rgba(50, 205, 50, 1.0)"),
    "3": ("fence",    "담장", "rgba(255, 165, 0, 0.06)",  "rgba(255, 165, 0, 1.0)"),
    "4": ("gate",     "문",   "rgba(220, 20, 60, 0.25)",  "rgba(220, 20, 60, 1.0)"),
}


# -----------------------------
# Plotly Figure 생성
# -----------------------------
def make_figure(site_w: float, site_h: float, items: list[dict], finished: bool):
    fig = go.Figure()

    # 좌표계: (0,0) ~ (site_w, site_h)
    fig.update_xaxes(range=[0, site_w], showgrid=True, zeroline=False, title="X (m)")
    fig.update_yaxes(
        range=[0, site_h],
        showgrid=True,
        zeroline=False,
        title="Y (m)",
        scaleanchor="x",
        scaleratio=1,
    )

    # 부지 경계
    shapes = [
        dict(
            type="rect",
            x0=0, y0=0, x1=site_w, y1=site_h,
            line=dict(color="black", width=3),
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        )
    ]

    # 기존 annotation 초기화 (uirevision 유지 중 누적 방지)
    fig.layout.annotations = []

    # 아이템
    for it in items:
        fill = "rgba(128,128,128,0.2)"
        line = "rgba(128,128,128,1)"
        label = it.get("kind", "item")

        for k, v in KIND_META.items():
            if v[0] == it["kind"]:
                _, label_kr, fill, line = v
                label = label_kr
                break

        x0 = it["x"] - it["w"] / 2
        x1 = it["x"] + it["w"] / 2
        y0 = it["y"] - it["h"] / 2
        y1 = it["y"] + it["h"] / 2

        # 담장은 거의 투명한 면 + 선으로 표현
        if it["kind"] == "fence":
            fill = "rgba(255,165,0,0.02)"

        shapes.append(
            dict(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                fillcolor=fill,
                line=dict(color=line, width=2),
                opacity=1.0,
                editable=(not finished),  # 완료 상태면 잠금
                name=it["id"],
            )
        )

        fig.add_annotation(
            x=it["x"], y=it["y"],
            text=f"{label}<br>{it['w']}×{it['h']}m",
            showarrow=False,
            font=dict(size=12),
            align="center",
        )

    fig.update_layout(
        margin=dict(l=30, r=30, t=30, b=30),
        dragmode="pan",
        showlegend=False,
        shapes=shapes,
        uirevision="keep",
    )
    return fig


# -----------------------------
# relayoutData에서 shape 이동/리사이즈를 items로 반영
# -----------------------------
def apply_relayout_to_items(relayoutData: dict, items: list[dict]):
    if not relayoutData:
        return items

    touched = set()
    for k in relayoutData.keys():
        if k.startswith("shapes[") and "]." in k:
            idx = k.split("]")[0].replace("shapes[", "")
            if idx.isdigit():
                touched.add(int(idx))

    # shapes[0]은 부지 경계. items는 shapes[1:]에 대응
    for shape_idx in touched:
        item_idx = shape_idx - 1
        if item_idx < 0 or item_idx >= len(items):
            continue

        x0 = relayoutData.get(f"shapes[{shape_idx}].x0")
        x1 = relayoutData.get(f"shapes[{shape_idx}].x1")
        y0 = relayoutData.get(f"shapes[{shape_idx}].y0")
        y1 = relayoutData.get(f"shapes[{shape_idx}].y1")

        # 드래그 중 부분 업데이트가 들어올 수 있어 완전한 4개가 없으면 스킵
        if x0 is None or x1 is None or y0 is None or y1 is None:
            continue

        w = abs(x1 - x0)
        h = abs(y1 - y0)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        items[item_idx]["w"] = round(float(w), 3)
        items[item_idx]["h"] = round(float(h), 3)
        items[item_idx]["x"] = round(float(cx), 3)
        items[item_idx]["y"] = round(float(cy), 3)

    return items


# -----------------------------
# Dash App
# -----------------------------
app = Dash(__name__)
server = app.server

DEFAULT_SITE = {"w": 200.0, "h": 120.0}

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "16px auto", "fontFamily": "Arial"},
    children=[
        html.H2("Plant PlotPlan 배치 WebUI (Dash)"),

        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                # 좌측 패널
                html.Div(
                    style={"flex": "1", "minWidth": "320px", "border": "1px solid #ddd", "padding": "12px", "borderRadius": "8px"},
                    children=[
                        html.H4("Step 1) 부지 경계 생성"),
                        html.Label("부지 가로(m)"),
                        dcc.Input(id="site-w", type="number", value=500, min=1, step=1, style={"width": "100%"}),
                        html.Br(), html.Br(),
                        html.Label("부지 세로(m)"),
                        dcc.Input(id="site-h", type="number", value=300, min=1, step=1, style={"width": "100%"}),
                        html.Br(), html.Br(),
                        html.Button("부지 생성/갱신", id="btn-make-site", style={"width": "100%"}),
                        html.Hr(),

                        html.H4("Step 2) 객체 추가(클릭 배치)"),
                        html.Div("옵션: 1(건물), 2(도로), 3(담장), 4(문)"),
                        dcc.Dropdown(
                            id="kind",
                            options=[
                                {"label": "1 - 건물", "value": "1"},
                                {"label": "2 - 도로", "value": "2"},
                                {"label": "3 - 담장", "value": "3"},
                                {"label": "4 - 문", "value": "4"},
                            ],
                            value="1",
                            clearable=False,
                        ),
                        html.Br(),
                        html.Label("가로(m)"),
                        dcc.Input(id="item-w", type="number", value=30, min=0.1, step=0.1, style={"width": "100%"}),
                        html.Br(), html.Br(),
                        html.Label("세로(m)"),
                        dcc.Input(id="item-h", type="number", value=20, min=0.1, step=0.1, style={"width": "100%"}),
                        html.Br(), html.Br(),
                        html.Button("추가 모드(다음 클릭에 배치)", id="btn-arm-add", style={"width": "100%"}),
                        html.Br(), html.Br(),
                        html.Div(id="add-status", style={"fontSize": "13px", "color": "#444"}),

                        html.Hr(),
                        html.H4("Step 3) 배치 완료"),
                        html.Button("배치 완료(ESC와 동일)", id="btn-finish", style={"width": "100%"}),
                        html.Br(), html.Br(),
                        html.Button("완료 해제(편집 재개)", id="btn-unfinish", style={"width": "100%"}),
                        html.Hr(),
                        html.Button("전체 초기화", id="btn-reset", style={"width": "100%"}),
                        html.Div(id="finish-status", style={"marginTop": "10px", "fontSize": "13px"}),

                        html.Div(
                            style={"marginTop": "10px", "fontSize": "12px", "color": "#666", "lineHeight": "1.4"},
                            children=[
                                html.Div("• 객체 이동: 배치된 사각형을 마우스로 드래그"),
                                html.Div("• 새 객체 배치: '추가 모드' 누른 뒤, 도면에서 원하는 위치 클릭"),
                                html.Div("• ESC 키: 배치 완료(추가/이동 잠금)"),
                            ],
                        ),
                    ],
                ),

                # 우측 그래프
                html.Div(
                    style={"flex": "2", "minWidth": "520px", "border": "1px solid #ddd", "padding": "12px", "borderRadius": "8px"},
                    children=[
                        EventListener(
                            id="key-listener",
                            events=[{"event": "keydown", "props": ["key"]}],
                            children=html.Div(
                                children=[
                                    dcc.Graph(
                                        id="graph",
                                        figure=make_figure(DEFAULT_SITE["w"], DEFAULT_SITE["h"], [], finished=False),
                                        config={"editable": True, "scrollZoom": True},
                                        style={"height": "720px"},
                                    )
                                ]
                            ),
                        ),
                    ],
                ),
            ],
        ),

        # 상태 저장
        dcc.Store(id="store-site", data=DEFAULT_SITE),
        dcc.Store(id="store-items", data=[]),
        dcc.Store(id="store-add-armed", data=False),
        dcc.Store(id="store-finished", data=False),
        dcc.Store(id="store-pending-spec", data=None),
    ],
)


# -----------------------------
# 단일 콜백(중복 Output 에러 방지)
# -----------------------------
@app.callback(
    Output("store-site", "data"),
    Output("store-items", "data"),
    Output("store-add-armed", "data"),
    Output("store-pending-spec", "data"),
    Output("store-finished", "data"),
    Output("graph", "figure"),
    Output("add-status", "children"),
    Output("finish-status", "children"),
    Input("btn-make-site", "n_clicks"),
    Input("btn-arm-add", "n_clicks"),
    Input("graph", "clickData"),
    Input("graph", "relayoutData"),
    Input("key-listener", "event"),
    Input("btn-finish", "n_clicks"),
    Input("btn-unfinish", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    State("site-w", "value"),
    State("site-h", "value"),
    State("kind", "value"),
    State("item-w", "value"),
    State("item-h", "value"),
    State("store-site", "data"),
    State("store-items", "data"),
    State("store-add-armed", "data"),
    State("store-pending-spec", "data"),
    State("store-finished", "data"),
    prevent_initial_call=True,
)
def unified_callback(
    _make_site, _arm_add, clickData, relayoutData, key_event, _finish, _unfinish, _reset,
    site_w_in, site_h_in, kind_in, item_w_in, item_h_in,
    site, items, armed, pending, finished
):
    trig = ctx.triggered_id

    add_msg = no_update
    fin_msg = no_update

    # 방어적 기본값
    if site is None:
        site = dict(DEFAULT_SITE)
    if items is None:
        items = []
    if armed is None:
        armed = False
    if pending is None:
        pending = None
    if finished is None:
        finished = False

    # 1) 전체 초기화
    if trig == "btn-reset":
        items = []
        armed = False
        pending = None
        finished = False
        fig = make_figure(site["w"], site["h"], items, finished=False)
        return site, items, armed, pending, finished, fig, "초기화 완료.", ""

    # 2) ESC 키로 배치 완료
    if trig == "key-listener":
        if key_event and key_event.get("key") == "Escape":
            finished = True
            armed = False
            pending = None
            fig = make_figure(site["w"], site["h"], items, finished=True)
            return site, items, armed, pending, finished, fig, "배치 완료 상태입니다.", "✅ ESC: 배치 완료(잠금)."
        raise PreventUpdate

    # 3) 완료 / 완료해제 버튼
    if trig in ("btn-finish", "btn-unfinish"):
        finished = (trig == "btn-finish")
        if finished:
            armed = False
            pending = None
            fin_msg = "✅ 배치 완료(추가/이동 잠금)."
        else:
            fin_msg = "✏️ 완료 해제(편집 재개)."
        fig = make_figure(site["w"], site["h"], items, finished=finished)
        return site, items, armed, pending, finished, fig, add_msg, fin_msg

    # 4) 부지 생성/갱신
    if trig == "btn-make-site":
        if not site_w_in or not site_h_in:
            raise PreventUpdate
        site = {"w": float(site_w_in), "h": float(site_h_in)}
        fig = make_figure(site["w"], site["h"], items, finished=finished)
        return site, items, armed, pending, finished, fig, add_msg, fin_msg

    # 5) 추가 모드 Arm
    if trig == "btn-arm-add":
        if finished:
            armed = False
            pending = None
            fig = make_figure(site["w"], site["h"], items, finished=True)
            return site, items, armed, pending, finished, fig, "배치 완료 상태입니다. 완료 해제 후 추가하세요.", fin_msg

        if not item_w_in or not item_h_in:
            fig = make_figure(site["w"], site["h"], items, finished=False)
            return site, items, False, None, finished, fig, "가로/세로 값을 입력하세요.", fin_msg

        k, kname, *_ = KIND_META[kind_in]
        armed = True
        pending = {"kind": k, "w": float(item_w_in), "h": float(item_h_in)}
        fig = make_figure(site["w"], site["h"], items, finished=False)
        return site, items, armed, pending, finished, fig, f"추가 모드 ON: 다음 클릭 위치에 [{kname}] 배치", fin_msg

    # 6) 그래프 클릭으로 배치
    # Dash에서 clickData/relayoutData 모두 graph에서 들어오므로, 값 존재로 구분
    if trig == "graph" and clickData is not None:
        if finished:
            raise PreventUpdate
        if not armed or not pending:
            raise PreventUpdate
        if "points" not in clickData or not clickData["points"]:
            raise PreventUpdate

        x = float(clickData["points"][0]["x"])
        y = float(clickData["points"][0]["y"])

        new_item = Item(
            id=str(uuid.uuid4())[:8],
            kind=pending["kind"],
            w=pending["w"],
            h=pending["h"],
            x=x,
            y=y,
        )
        items = items + [asdict(new_item)]
        armed = False
        pending = None
        fig = make_figure(site["w"], site["h"], items, finished=False)
        return site, items, armed, pending, finished, fig, "추가 완료. 또 추가하려면 '추가 모드'를 다시 누르세요.", fin_msg

    # 7) 드래그 이동/리사이즈 반영
    if trig == "graph" and relayoutData is not None:
        if finished:
            raise PreventUpdate
        items = apply_relayout_to_items(relayoutData, items)
        fig = make_figure(site["w"], site["h"], items, finished=False)
        return site, items, armed, pending, finished, fig, add_msg, fin_msg

    raise PreventUpdate


if __name__ == "__main__":
    # Dash 3.x: run_server 대신 run 사용
    app.run(debug=True, host="127.0.0.1", port=8050)
