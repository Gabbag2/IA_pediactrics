"""Live-replay dashboard for a pre-computed DemoBundle.

Four synchronised panels, driven by a single frame index advanced by a
``dcc.Interval`` callback:

    ┌───────────────────────────┬───────────────────────────┐
    │ Scrolling 18-ch EEG trace │ 3D latent space           │
    │                           │   • ghost cloud (train)   │
    │                           │   • trail  + current pt   │
    │                           │   • centroid spheres      │
    ├───────────────────────────┼───────────────────────────┤
    │ Scalp topography          │ Risk / TTS / Criticality  │
    │   + channel × band heat   │ gauges                    │
    └───────────────────────────┴───────────────────────────┘

Run:
    python app/dashboard.py --bundle cache/demo/chb01_s0.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.demo_bundle import DemoBundle, load_bundle  # noqa: E402
from src.topography import bipolar_positions, head_outline  # noqa: E402


STATE_COLORS = {0: "#6baed6", 1: "#fd8d3c", 2: "#e31a1c"}
STATE_NAMES = {0: "interictal", 1: "pre-ictal", 2: "ictal"}


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _ghost_figure(b: DemoBundle, max_points: int = 6000) -> go.Figure:
    """Build the 3D ghost-cloud scatter + placeholder traces for trail / cursor.

    The dashboard only needs to update the *traces that move*, so we build the
    static layers (ghost, centroids) once and append trail + cursor as
    updatable traces.
    """
    rng = np.random.default_rng(0)
    n = min(max_points, b.ghost_Z.shape[0])
    idx = rng.choice(b.ghost_Z.shape[0], size=n, replace=False)
    Z = b.ghost_Z[idx, :3]
    st = b.ghost_state[idx]

    fig = go.Figure()
    for s in (0, 1, 2):
        m = st == s
        if not m.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=Z[m, 0], y=Z[m, 1], z=Z[m, 2],
            mode="markers",
            marker=dict(size=1.5, color=STATE_COLORS[s], opacity=0.25),
            name=f"ghost · {STATE_NAMES[s]}",
            hoverinfo="skip",
        ))
    # centroid spheres via a single big marker (fast; no Mesh3d cost)
    for name, c, color in [
        ("centroid · interictal", b.centroid_interictal, STATE_COLORS[0]),
        ("centroid · pre-ictal",  b.centroid_preictal,  STATE_COLORS[1]),
        ("centroid · ictal",      b.centroid_ictal,     STATE_COLORS[2]),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[c[0]], y=[c[1]], z=[c[2]],
            mode="markers",
            marker=dict(size=22, color=color, opacity=0.25,
                        line=dict(width=1, color="white")),
            name=name, hoverinfo="name",
        ))
    # trail (placeholder, updated per frame)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(width=5, color="white"),
        name="trail", hoverinfo="skip",
    ))
    # cursor (placeholder)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="markers",
        marker=dict(size=10, color="#ffffff",
                    line=dict(width=2, color="black")),
        name="now", hoverinfo="name",
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title="z₀", yaxis_title="z₁", zaxis_title="z₂",
            bgcolor="#0b1220",
            xaxis=dict(color="#cfd8dc", gridcolor="#2a3440"),
            yaxis=dict(color="#cfd8dc", gridcolor="#2a3440"),
            zaxis=dict(color="#cfd8dc", gridcolor="#2a3440"),
        ),
        paper_bgcolor="#0b1220",
        font=dict(color="#e0e0e0"),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=-0.05, font=dict(size=9)),
        title="latent trajectory (dims 0–2)",
        uirevision="latent_static",
    )
    return fig


def _eeg_figure(b: DemoBundle, window_s: float = 10.0) -> go.Figure:
    """Placeholder figure for the scrolling EEG trace. Data filled per frame."""
    n_ch = len(b.eeg_channels)
    fig = go.Figure()
    for i, ch in enumerate(b.eeg_channels):
        fig.add_trace(go.Scattergl(
            x=[], y=[], mode="lines", name=ch,
            line=dict(width=1, color="#9bd1ff"),
            hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        title=f"EEG · {window_s:.0f} s window",
        paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="time (s)", color="#cfd8dc", gridcolor="#1d2a3a"),
        yaxis=dict(showticklabels=True, tickvals=list(range(n_ch)),
                   ticktext=b.eeg_channels,
                   color="#cfd8dc", gridcolor="#1d2a3a"),
        margin=dict(l=60, r=10, t=30, b=30),
        uirevision="eeg",
    )
    return fig


def _topo_figure(b: DemoBundle) -> go.Figure:
    """Top-down scalp view with 18 bipolar electrodes; colour fills per frame."""
    outline = head_outline()
    pos = bipolar_positions(b.eeg_channels)

    fig = go.Figure()
    # head outline
    c = outline["circle"]
    fig.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], mode="lines",
                             line=dict(color="#e0e0e0", width=2),
                             hoverinfo="skip", showlegend=False))
    for key in ("nose", "left_ear", "right_ear"):
        p = outline[key]
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines",
                                 line=dict(color="#e0e0e0", width=2),
                                 hoverinfo="skip", showlegend=False))
    # electrodes — placeholder, updated per frame
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1],
        mode="markers+text",
        marker=dict(size=24, color=[0.0] * len(b.eeg_channels),
                    colorscale="Inferno", cmin=0.0, cmax=1.0,
                    showscale=False,
                    line=dict(width=1, color="#ffffff")),
        text=b.eeg_channels,
        textposition="middle center",
        textfont=dict(size=7, color="#ffffff"),
        hovertext=b.eeg_channels, hoverinfo="text",
        name="electrodes",
    ))
    fig.update_layout(
        title="scalp · attribution",
        paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
        font=dict(color="#e0e0e0"),
        xaxis=dict(range=[-1.3, 1.3], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-1.3, 1.3], visible=False),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        uirevision="topo",
    )
    return fig


def _heatmap_figure(b: DemoBundle) -> go.Figure:
    """Channel × band attribution heatmap (static layout, data per frame)."""
    fig = go.Figure(data=go.Heatmap(
        z=np.zeros((len(b.channels), len(b.bands))),
        x=list(b.bands), y=list(b.channels),
        colorscale="Inferno", zmin=0.0, zmax=1.0,
        showscale=True, colorbar=dict(title="|∂z/∂x|", tickfont=dict(size=9)),
    ))
    fig.update_layout(
        title="channel × band",
        paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
        font=dict(color="#e0e0e0"),
        xaxis=dict(color="#cfd8dc"),
        yaxis=dict(color="#cfd8dc", autorange="reversed",
                   tickfont=dict(size=9)),
        margin=dict(l=70, r=10, t=30, b=30),
        uirevision="heat",
    )
    return fig


def _gauges_figure(b: DemoBundle) -> go.Figure:
    """Three gauges: risk (0-100), time-to-seizure (min), criticality (0-5)."""
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=0,
        title=dict(text="seizure risk", font=dict(size=13)),
        domain=dict(row=0, column=0),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color="#e31a1c"),
            steps=[
                dict(range=[0, 40], color="#1e3a5f"),
                dict(range=[40, 70], color="#b35900"),
                dict(range=[70, 100], color="#7a1c1c"),
            ],
        ),
        number=dict(suffix="", font=dict(size=26)),
    ))
    fig.add_trace(go.Indicator(
        mode="number",
        value=0,
        number=dict(suffix=" min", valueformat=".1f", font=dict(size=34)),
        title=dict(text="time to seizure (est.)", font=dict(size=13)),
        domain=dict(row=0, column=1),
    ))
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=0,
        title=dict(text="criticality index", font=dict(size=13)),
        domain=dict(row=0, column=2),
        gauge=dict(
            axis=dict(range=[0, 5]),
            bar=dict(color="#fd8d3c"),
            steps=[
                dict(range=[0, 1], color="#1e3a5f"),
                dict(range=[1, 3], color="#b35900"),
                dict(range=[3, 5], color="#7a1c1c"),
            ],
        ),
        number=dict(font=dict(size=26), valueformat=".2f"),
    ))
    fig.update_layout(
        grid=dict(rows=1, columns=3, pattern="independent"),
        paper_bgcolor="#0b1220",
        font=dict(color="#e0e0e0"),
        margin=dict(l=20, r=20, t=40, b=20),
        uirevision="gauges",
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app(b: DemoBundle) -> Dash:
    app = Dash(__name__, title=f"seizure replay · {b.patient}")

    initial_ghost = _ghost_figure(b)
    initial_eeg = _eeg_figure(b)
    initial_topo = _topo_figure(b)
    initial_heat = _heatmap_figure(b)
    initial_gauge = _gauges_figure(b)

    # normalise per-frame ch×band for display (global min/max across the replay)
    heat_max = float(np.nanmax(b.frame_ch_band)) or 1.0

    panel_style = {
        "background": "#0b1220",
        "border": "1px solid #1d2a3a",
        "padding": "4px",
        "borderRadius": "6px",
    }
    app.layout = html.Div(
        style={"background": "#070b12", "minHeight": "100vh",
               "color": "#e0e0e0", "fontFamily": "Inter, system-ui, sans-serif",
               "padding": "10px"},
        children=[
            html.Div(style={"display": "flex", "alignItems": "center",
                            "justifyContent": "space-between",
                            "padding": "6px 12px"},
                     children=[
                html.Div([
                    html.H2(f"{b.patient}  ·  seizure #{b.seizure_idx}",
                            style={"margin": 0, "color": "#e0e0e0"}),
                    html.Div(id="clock",
                             style={"fontFamily": "JetBrains Mono, monospace",
                                    "color": "#9bd1ff", "fontSize": "14px"}),
                ]),
                html.Div([
                    html.Button("▶ Play", id="play-btn", n_clicks=0,
                                style={"marginRight": "10px", "padding": "6px 14px"}),
                    html.Button("⟲ Reset", id="reset-btn", n_clicks=0,
                                style={"marginRight": "10px", "padding": "6px 14px"}),
                    html.Span("speed", style={"marginRight": "6px"}),
                    dcc.Slider(id="speed", min=10, max=120, step=5, value=60,
                               marks={10: "10×", 60: "60×", 120: "120×"},
                               tooltip={"placement": "bottom"}),
                ], style={"width": "420px"}),
            ]),
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gridTemplateRows": "1fr 1fr",
                            "gap": "10px", "height": "calc(100vh - 110px)"},
                     children=[
                html.Div(dcc.Graph(id="eeg-graph", figure=initial_eeg,
                                   style={"height": "100%"},
                                   config={"displayModeBar": False}),
                         style=panel_style),
                html.Div(dcc.Graph(id="latent-graph", figure=initial_ghost,
                                   style={"height": "100%"},
                                   config={"displayModeBar": False}),
                         style=panel_style),
                html.Div(style={**panel_style, "display": "grid",
                                "gridTemplateColumns": "1fr 1fr"},
                         children=[
                    dcc.Graph(id="topo-graph", figure=initial_topo,
                              style={"height": "100%"},
                              config={"displayModeBar": False}),
                    dcc.Graph(id="heat-graph", figure=initial_heat,
                              style={"height": "100%"},
                              config={"displayModeBar": False}),
                ]),
                html.Div(dcc.Graph(id="gauge-graph", figure=initial_gauge,
                                   style={"height": "100%"},
                                   config={"displayModeBar": False}),
                         style=panel_style),
            ]),
            dcc.Interval(id="tick", interval=100, disabled=True),
            dcc.Store(id="frame-idx", data=0),
            dcc.Store(id="playing", data=False),
        ],
    )

    n_frames = len(b.frame_t)
    eeg_window_s = 10.0
    eeg_samples_per_window = int(eeg_window_s * b.eeg_sfreq)

    # ---- tick: advance frame ------------------------------------------------
    @app.callback(
        Output("frame-idx", "data"),
        Input("tick", "n_intervals"),
        State("frame-idx", "data"),
        State("speed", "value"),
        prevent_initial_call=True,
    )
    def _tick(_n, idx, speed):
        # Advance ``speed`` × real-time. Interval is 100 ms, step is 2.5 s,
        # so at speed=60 we advance 60×0.1 / 2.5 = 2.4 frames per tick.
        step = max(1, int(round(speed * 0.1 / b.step_s)))
        new_idx = (idx + step) % n_frames
        return new_idx

    # ---- play / pause / reset ----------------------------------------------
    @app.callback(
        Output("tick", "disabled"),
        Output("play-btn", "children"),
        Output("playing", "data"),
        Output("frame-idx", "data", allow_duplicate=True),
        Input("play-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        State("playing", "data"),
        prevent_initial_call=True,
    )
    def _controls(_play, _reset, playing):
        from dash import callback_context as ctx
        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
        if trig == "reset-btn":
            return True, "▶ Play", False, 0
        # toggle play
        new = not playing
        return (not new), ("⏸ Pause" if new else "▶ Play"), new, no_update

    # ---- figure updates on frame change ------------------------------------
    @app.callback(
        Output("eeg-graph", "figure"),
        Output("latent-graph", "figure"),
        Output("topo-graph", "figure"),
        Output("heat-graph", "figure"),
        Output("gauge-graph", "figure"),
        Output("clock", "children"),
        Input("frame-idx", "data"),
        State("eeg-graph", "figure"),
        State("latent-graph", "figure"),
        State("topo-graph", "figure"),
        State("heat-graph", "figure"),
        State("gauge-graph", "figure"),
    )
    def _render(idx, eeg_fig, lat_fig, topo_fig, heat_fig, gauge_fig):
        idx = int(idx)
        t = float(b.frame_t[idx])                 # seconds rel to onset
        state = int(b.frame_state[idx])

        # ------------- EEG scrolling window --------------------------------
        eeg_time_center = t + b.onset_s           # seconds from eeg start
        i_center = int(round(eeg_time_center * b.eeg_sfreq))
        half = eeg_samples_per_window // 2
        i0 = max(0, i_center - half)
        i1 = min(b.eeg.shape[1], i0 + eeg_samples_per_window)
        i0 = max(0, i1 - eeg_samples_per_window)
        x_axis = np.arange(i0, i1) / b.eeg_sfreq - b.onset_s  # x=0 at onset
        # stack channels vertically; scale each channel to ~0.8 spacing
        scale = 1.0 / (np.std(b.eeg[:, i0:i1], axis=1, keepdims=True) + 1e-6)
        stacked = b.eeg[:, i0:i1] * scale * 0.4   # ≤ ±0.4 per channel
        for ci in range(len(b.eeg_channels)):
            eeg_fig["data"][ci]["x"] = x_axis.tolist()
            eeg_fig["data"][ci]["y"] = (stacked[ci] + ci).tolist()
        # onset marker
        eeg_fig["layout"]["shapes"] = [dict(
            type="line", x0=0, x1=0, yref="paper", y0=0, y1=1,
            line=dict(color="#e31a1c", width=1, dash="dash"),
        ), dict(
            type="line", x0=t, x1=t, yref="paper", y0=0, y1=1,
            line=dict(color="#9bd1ff", width=1),
        )]

        # ------------- 3D latent trail + cursor ---------------------------
        trail_lo = max(0, idx - 60)
        trail = b.frame_Z[trail_lo:idx + 1, :3]
        lat_fig["data"][-2]["x"] = trail[:, 0].tolist()
        lat_fig["data"][-2]["y"] = trail[:, 1].tolist()
        lat_fig["data"][-2]["z"] = trail[:, 2].tolist()
        lat_fig["data"][-2]["line"]["color"] = STATE_COLORS[state]
        cur = b.frame_Z[idx, :3]
        lat_fig["data"][-1]["x"] = [float(cur[0])]
        lat_fig["data"][-1]["y"] = [float(cur[1])]
        lat_fig["data"][-1]["z"] = [float(cur[2])]
        lat_fig["data"][-1]["marker"]["color"] = STATE_COLORS[state]

        # ------------- topography + heatmap -------------------------------
        ch_band = b.frame_ch_band[idx]
        ch_sum = ch_band.sum(axis=1)
        ch_sum_norm = ch_sum / (heat_max * len(b.bands) + 1e-9)
        topo_fig["data"][-1]["marker"]["color"] = ch_sum_norm.tolist()
        topo_fig["data"][-1]["marker"]["cmax"] = float(np.clip(ch_sum_norm.max(), 0.1, 1.0))

        heat_fig["data"][0]["z"] = ch_band.tolist()
        heat_fig["data"][0]["zmax"] = heat_max

        # ------------- gauges ---------------------------------------------
        risk = float(b.frame_risk[idx])
        tts = float(b.frame_tts[idx])
        crit = float(b.frame_criticality[idx])
        gauge_fig["data"][0]["value"] = risk
        gauge_fig["data"][1]["value"] = tts if np.isfinite(tts) else 0.0
        gauge_fig["data"][1]["title"]["text"] = (
            "time to seizure (est.)" if np.isfinite(tts)
            else "time to seizure  —  n/a"
        )
        gauge_fig["data"][2]["value"] = crit

        # ------------- clock ----------------------------------------------
        sign = "−" if t < 0 else "+"
        clock = (f"t = {sign}{abs(t):6.1f} s   ·   state: {STATE_NAMES[state]}"
                 f"   ·   frame {idx + 1}/{n_frames}")

        return eeg_fig, lat_fig, topo_fig, heat_fig, gauge_fig, clock

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True,
                    help=".npz produced by scripts/05_build_demo.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    app = build_app(bundle)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
