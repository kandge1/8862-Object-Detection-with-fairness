import argparse
import csv
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


SIZES_ORDER = ["Nano", "Small", "Medium", "Large", "X-Large"]
SIZES_TICKS = ["nano", "small", "medium", "large", "x-large"]


def _normalize_series(model_name: str, series: str) -> str:
    mn = (model_name or "").lower()
    s = (series or "").lower()

    if mn.startswith("yolov8") or s == "yolov8":
        return "YOLO8"
    if mn.startswith("yolo12") or s.startswith("yolo12"):
        return "YOLO12"
    if mn.startswith("yolo26") or s == "yolo26":
        return "YOLO26"

    if s:
        return s.upper()
    return (model_name or "UNKNOWN").upper()


def load_results(csv_path: str):
    """
    Returns: results[benchmark][series][size] = mAP (float)
    """
    results = defaultdict(lambda: defaultdict(dict))

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            benchmark = (row.get("benchmark") or "").strip()
            model_name = (row.get("model_name") or "").strip()
            series = _normalize_series(model_name, (row.get("series") or "").strip())
            size = (row.get("size") or "").strip()
            m = row.get("mAP")
            if not benchmark or not size or m is None or m == "":
                continue

            try:
                m = float(m)
            except ValueError:
                continue

            results[benchmark][series][size] = m

    return results


def _svg_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _write_svg_line_chart(
    *,
    title: str,
    x_labels: list[str],
    series_to_y: dict[str, list[float | None]],
    out_path: str,
):
    width, height = 1000, 600
    margin_left, margin_right, margin_top, margin_bottom = 85, 35, 70, 85

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_y = []
    for ys in series_to_y.values():
        for v in ys:
            if v is not None:
                all_y.append(v)

    if not all_y:
        y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = min(all_y), max(all_y)
        pad = max(0.02, (y_max - y_min) * 0.08)
        y_min = max(0.0, y_min - pad)
        y_max = min(1.0, y_max + pad)
        if abs(y_max - y_min) < 1e-9:
            y_min = max(0.0, y_min - 0.05)
            y_max = min(1.0, y_max + 0.05)

    def x_to_px(i: int) -> float:
        if len(x_labels) == 1:
            return margin_left + plot_w / 2
        return margin_left + (plot_w * i) / (len(x_labels) - 1)

    def y_to_px(y: float) -> float:
        t = (y - y_min) / (y_max - y_min)
        return margin_top + (1.0 - t) * plot_h

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')

    parts.append(
        f'<text x="{width/2:.1f}" y="38" text-anchor="middle" font-size="22" font-family="Segoe UI, Arial">'
        f"{_svg_escape(title)}</text>"
    )

    x0, y0 = margin_left, margin_top + plot_h
    x1, y1 = margin_left + plot_w, margin_top
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111" stroke-width="2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111" stroke-width="2"/>')

    for i, lbl in enumerate(x_labels):
        xp = x_to_px(i)
        parts.append(f'<line x1="{xp:.2f}" y1="{y0}" x2="{xp:.2f}" y2="{y0+6}" stroke="#111" stroke-width="1"/>')
        parts.append(
            f'<text x="{xp:.2f}" y="{y0+28}" text-anchor="middle" font-size="14" font-family="Segoe UI, Arial">'
            f"{_svg_escape(lbl)}</text>"
        )

    for t in range(6):
        yv = y_min + (y_max - y_min) * (t / 5.0)
        yp = y_to_px(yv)
        parts.append(f'<line x1="{x0}" y1="{yp:.2f}" x2="{x1}" y2="{yp:.2f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(
            f'<text x="{x0-10}" y="{yp+5:.2f}" text-anchor="end" font-size="13" font-family="Segoe UI, Arial">'
            f"{yv:.3f}</text>"
        )

    parts.append(
        f'<text x="{width/2:.1f}" y="{height-25}" text-anchor="middle" font-size="16" font-family="Segoe UI, Arial">'
        f"Model size</text>"
    )
    parts.append(
        f'<text x="22" y="{height/2:.1f}" text-anchor="middle" font-size="16" font-family="Segoe UI, Arial" '
        f'transform="rotate(-90 22 {height/2:.1f})">mAP</text>'
    )

    for idx, (series, ys) in enumerate(series_to_y.items()):
        color = colors[idx % len(colors)]
        pts = []
        for i, v in enumerate(ys):
            if v is None:
                pts.append(None)
            else:
                pts.append((x_to_px(i), y_to_px(float(v))))

        seg = []
        for p in pts:
            if p is None:
                if seg:
                    parts.append(
                        f'<polyline fill="none" stroke="{color}" stroke-width="3" points="'
                        + " ".join(f"{a:.2f},{b:.2f}" for a, b in seg)
                        + '"/>'
                    )
                    seg = []
                continue
            seg.append(p)
        if seg:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" points="'
                + " ".join(f"{a:.2f},{b:.2f}" for a, b in seg)
                + '"/>'
            )

        for p in pts:
            if p is None:
                continue
            parts.append(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="4.5" fill="{color}" />')

    legend_x, legend_y = width - margin_right - 220, margin_top - 12
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="220" height="70" fill="white" stroke="#ccc"/>')
    for idx, series in enumerate(series_to_y.keys()):
        color = colors[idx % len(colors)]
        y = legend_y + 22 + idx * 18
        parts.append(f'<line x1="{legend_x+12}" y1="{y}" x2="{legend_x+42}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x+27}" cy="{y}" r="4" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x+52}" y="{y+5}" font-size="14" font-family="Segoe UI, Arial">'
            f"{_svg_escape(series)}</text>"
        )

    parts.append("</svg>")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(parts))


def plot_benchmark(results_for_benchmark: dict, benchmark: str, out_dir: str, show: bool):
    series_order = ["YOLO8", "YOLO12", "YOLO26"]

    title_benchmark = f"{benchmark}Det" if not benchmark.endswith("Det") else benchmark
    os.makedirs(out_dir, exist_ok=True)

    series_to_y = {}
    for series in series_order:
        size_to_map = results_for_benchmark.get(series, {})
        series_to_y[series] = [size_to_map.get(sz, None) for sz in SIZES_ORDER]

    if plt is None:
        out_path = os.path.join(out_dir, f"{benchmark}_mAP_by_size.svg")
        _write_svg_line_chart(
            title=f"{title_benchmark} against YOLO",
            x_labels=SIZES_TICKS,
            series_to_y=series_to_y,
            out_path=out_path,
        )
        return out_path

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = list(range(len(SIZES_ORDER)))

    for series, ys in series_to_y.items():
        ax.plot(x, ys, marker="o", linewidth=2, label=series)

    ax.set_title(f"{title_benchmark} against YOLO")
    ax.set_xlabel("Model size")
    ax.set_ylabel("mAP")
    ax.set_xticks(x, SIZES_TICKS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{benchmark}_mAP_by_size.png")
    fig.savefig(out_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot MOT17/MOT20 mAP results by YOLO series and model size.")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "benchmark_results.csv"),
        help="Path to benchmark_results.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(__file__),
        help="Directory to save PNG plots",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    results = load_results(args.csv)

    wanted = ["MOT17", "MOT20"]
    for b in wanted:
        if b not in results:
            raise SystemExit(f"Missing benchmark '{b}' in CSV: {args.csv}")

    saved = []
    for b in wanted:
        saved.append(plot_benchmark(results[b], b, args.out_dir, args.show))

    print("Saved plots:")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
