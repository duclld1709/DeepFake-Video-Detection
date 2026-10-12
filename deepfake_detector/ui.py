"""Gradio demo UI. Run with ``python app.py`` or ``deepfake-detect serve-ui``."""

from __future__ import annotations

import base64
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deepfake_detector.inference import DeepfakeAnalyzer, NoFaceDetectedError, VideoTooLongError  # noqa: E402
from deepfake_detector.schemas import AnalysisResult, Verdict  # noqa: E402
from deepfake_detector.video import VideoReadError  # noqa: E402

SERIES = "#2a78d6"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

_VERDICT_TEXT = {
    Verdict.FAKE: "⚠️ **Likely manipulated**",
    Verdict.REAL: "✅ **No manipulation detected**",
    Verdict.INCONCLUSIVE: "❔ **Inconclusive**",
}


def summary_markdown(result: AnalysisResult) -> str:
    lines = [
        f"## {_VERDICT_TEXT[result.verdict]}",
        f"**P(fake): {result.fake_probability:.1%}** (threshold {result.threshold:.0%}) · "
        f"max clip {result.max_clip_fake_probability:.1%} · face coverage {result.face_coverage:.0%}",
        "",
        "| Clip | Time | Frames | P(fake) |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {c.index} | {c.start_s:.1f}s – {c.end_s:.1f}s | {c.frames_used} | {c.fake_probability:.1%} |"
        for c in result.clips
    ]
    t = result.timings
    lines += [
        "",
        f"<sub>Model {result.model.version} · {result.model.temporal_mode} · faces: {result.model.face_backend} · "
        f"{result.model.device} · faces {t.faces_ms:.0f} ms · inference {t.inference_ms:.0f} ms · "
        f"Grad-CAM {t.explain_ms:.0f} ms · total {t.total_ms:.0f} ms</sub>",
    ]
    if result.warnings:
        lines += ["", "**Notes**"] + [f"- {w}" for w in result.warnings]
    lines += [
        "",
        "<sub>This is a research model with known limits (see the model card). "
        "Treat the score as one signal, not proof.</sub>",
    ]
    return "\n".join(lines)


def timeline_figure(result: AnalysisResult):
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for c in result.clips:
        ax.axvspan(c.start_s, c.end_s, color=GRID, alpha=0.6, lw=0)
    scored = [f for f in result.frames if f.fake_probability is not None]
    if scored:
        ax.plot([f.timestamp_s for f in scored], [f.fake_probability for f in scored], color=SERIES, lw=2)
        title = "Per-frame P(fake) over time (shaded = analysed clips)"
    else:
        xs = [(c.start_s + c.end_s) / 2 for c in result.clips]
        ax.plot(xs, [c.fake_probability for c in result.clips], color=SERIES, lw=2, marker="o", ms=8)
        title = "Per-clip P(fake) over time"
    ax.axhline(result.threshold, color=INK_2, lw=1, ls="--")
    ax.text(ax.get_xlim()[1], result.threshold, " threshold", va="center", ha="left", color=INK_2, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)", color=INK_2)
    ax.set_ylabel("P(fake)", color=INK_2)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    fig.tight_layout()
    return fig


def gallery_items(result: AnalysisResult) -> list[tuple[np.ndarray, str]]:
    items = []
    for e in result.explanations:
        buf = np.frombuffer(base64.b64decode(e.overlay_png_base64), dtype=np.uint8)
        image = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        score = "" if e.fake_probability is None else f" · {e.fake_probability:.0%}"
        items.append((image, f"{e.timestamp_s:.2f}s{score}"))
    return items


def build_ui(analyzer: DeepfakeAnalyzer):
    import gradio as gr

    def run(video_path, num_clips, explain):
        if not video_path:
            raise gr.Error("Please upload a video.")
        try:
            result = analyzer.analyze(video_path, num_clips=int(num_clips), explain_frames=int(explain))
        except (NoFaceDetectedError, VideoTooLongError, VideoReadError) as exc:
            raise gr.Error(str(exc)) from exc
        fig = timeline_figure(result)
        payload = result.model_dump(mode="json", exclude={"explanations"})
        return summary_markdown(result), fig, gallery_items(result), json.dumps(payload, indent=2)

    with gr.Blocks(title="Deepfake Video Detection") as demo:
        gr.Markdown(
            "# 🕵️ Deepfake Video Detection\n"
            "Upload a video with one clearly visible face. The model scores several clips across the video, "
            "shows how suspicion changes over time and highlights the facial regions that drove the decision."
        )
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Video", sources=["upload"])
                num_clips = gr.Slider(1, 6, value=analyzer.config.num_clips, step=1, label="Clips to analyse")
                explain = gr.Slider(0, 12, value=analyzer.config.gradcam_frames, step=1, label="Grad-CAM frames")
                button = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                plot = gr.Plot(show_label=False)
                gallery = gr.Gallery(label="Most suspicious frames (Grad-CAM)", columns=6, height=240)
                with gr.Accordion("Raw JSON", open=False):
                    raw = gr.Code(language="json")
        button.click(run, [video, num_clips, explain], [summary, plot, gallery, raw])
    return demo
