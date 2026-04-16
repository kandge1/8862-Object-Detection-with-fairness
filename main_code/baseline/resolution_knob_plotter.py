import argparse
import csv
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


SIZE_ORDER = ["Nano", "Small", "Medium", "Large", "X-Large"]
SIZE_TICKS = ["n", "s", "m", "l", "x"]


def _normalize_size(size: str) -> str:
    s = (size or "").strip().lower()
    if s in {"nano", "n"}:
        return "Nano"
    if s in {"small", "s"}:
        return "Small"
    if s in {"medium", "m"}:
        return "Medium"
    if s in {"large", "l"}:
        return "Large"
    if s in {"x-large", "xlarge", "xl", "x"}:
        return "X-Large"
    return size


def load_resolution_sweep(csv_path: str, benchmark: str = "MOT17"):
    """
    Returns:
      map_data[size][resolution] = mAP
      time_data[size][resolution] = benchmark_time_sec
      resolutions_sorted = sorted list of resolutions found
    """
    map_data = defaultdict(dict)
    time_data = defaultdict(dict)
    resolutions = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("benchmark") or "").strip() != benchmark:
                continue

            raw_resolution = row.get("resolution")
            raw_map = row.get("mAP")
            raw_time = row.get("benchmark_time_sec")
            if not raw_resolution or raw_map in (None, ""):
                continue

            try:
                resolution = int(float(raw_resolution))
                map_value = float(raw_map)
                time_value = float(raw_time) if raw_time not in (None, "") else None
            except ValueError:
                continue

            size = _normalize_size((row.get("size") or "").strip())
            if size not in SIZE_ORDER:
                continue

            map_data[size][resolution] = map_value
            if time_value is not None:
                time_data[size][resolution] = time_value
            resolutions.add(resolution)

    resolutions_sorted = sorted(resolutions)
    return map_data, time_data, resolutions_sorted


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
    out_path: str,
    x_values: list[int],
    series_to_y: dict[str, list[float | None]],
    title: str,
    y_label: str,
):
    width, height = 1100, 650
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 70, 90

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
        y_min = y_min - pad
        y_max = y_max + pad
        if abs(y_max - y_min) < 1e-9:
            y_min = y_min - 0.05
            y_max = y_max + 0.05

    xmin, xmax = min(x_values), max(x_values)

    def x_to_px(xv: int) -> float:
        if xmax == xmin:
            return margin_left + plot_w / 2
        return margin_left + ((xv - xmin) / (xmax - xmin)) * plot_w

    def y_to_px(y: float) -> float:
        t = (y - y_min) / (y_max - y_min)
        return margin_top + (1.0 - t) * plot_h

    colors = {
        "n": "#1f77b4",
        "s": "#ff7f0e",
        "m": "#2ca02c",
        "l": "#d62728",
        "x": "#9467bd",
    }

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width/2:.1f}" y="38" text-anchor="middle" font-size="22" font-family="Segoe UI, Arial">{_svg_escape(title)}</text>'
    )

    x0, y0 = margin_left, margin_top + plot_h
    x1, y1 = margin_left + plot_w, margin_top
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111" stroke-width="2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111" stroke-width="2"/>')

    for xv in x_values:
        xp = x_to_px(xv)
        parts.append(f'<line x1="{xp:.2f}" y1="{y0}" x2="{xp:.2f}" y2="{y0+6}" stroke="#111" stroke-width="1"/>')
        parts.append(
            f'<text x="{xp:.2f}" y="{y0+28}" text-anchor="middle" font-size="14" font-family="Segoe UI, Arial">{xv}</text>'
        )

    for t in range(6):
        yv = y_min + (y_max - y_min) * (t / 5.0)
        yp = y_to_px(yv)
        parts.append(f'<line x1="{x0}" y1="{yp:.2f}" x2="{x1}" y2="{yp:.2f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(
            f'<text x="{x0-10}" y="{yp+5:.2f}" text-anchor="end" font-size="13" font-family="Segoe UI, Arial">{yv:.3f}</text>'
        )

    parts.append(
        f'<text x="{width/2:.1f}" y="{height-25}" text-anchor="middle" font-size="16" font-family="Segoe UI, Arial">Resolution (imgsz)</text>'
    )
    parts.append(
        f'<text x="22" y="{height/2:.1f}" text-anchor="middle" font-size="16" font-family="Segoe UI, Arial" transform="rotate(-90 22 {height/2:.1f})">{_svg_escape(y_label)}</text>'
    )

    for size_tick, ys in series_to_y.items():
        color = colors.get(size_tick, "#333333")
        points = []
        for idx, yv in enumerate(ys):
            if yv is None:
                continue
            points.append((x_to_px(x_values[idx]), y_to_px(float(yv))))

        if points:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" points="'
                + " ".join(f"{a:.2f},{b:.2f}" for a, b in points)
                + '"/>'
            )

        for p in points:
            parts.append(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="4.5" fill="{color}" />')

    legend_x, legend_y = width - margin_right - 170, margin_top - 10
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="170" height="120" fill="white" stroke="#ccc"/>')
    for idx, size_tick in enumerate(series_to_y.keys()):
        color = colors.get(size_tick, "#333333")
        y = legend_y + 22 + idx * 20
        parts.append(f'<line x1="{legend_x+12}" y1="{y}" x2="{legend_x+42}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x+27}" cy="{y}" r="4" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x+52}" y="{y+5}" font-size="14" font-family="Segoe UI, Arial">YOLOv8{_svg_escape(size_tick)}</text>'
        )

    parts.append("</svg>")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(parts))


def _build_series(data: dict, resolutions: list[int]):
    return {
        "n": [data.get("Nano", {}).get(r, None) for r in resolutions],
        "s": [data.get("Small", {}).get(r, None) for r in resolutions],
        "m": [data.get("Medium", {}).get(r, None) for r in resolutions],
        "l": [data.get("Large", {}).get(r, None) for r in resolutions],
        "x": [data.get("X-Large", {}).get(r, None) for r in resolutions],
    }


def plot_resolution_vs_metric(
    data: dict,
    resolutions: list[int],
    out_path: str,
    show: bool,
    *,
    title: str,
    y_label: str,
):
    series = _build_series(data, resolutions)

    if plt is None:
        svg_path = os.path.splitext(out_path)[0] + ".svg"
        _write_svg_line_chart(
            out_path=svg_path,
            x_values=resolutions,
            series_to_y=series,
            title=title,
            y_label=y_label,
        )
        return svg_path

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, ys in series.items():
        ax.plot(resolutions, ys, marker="o", linewidth=2, label=f"YOLOv8{label}")

    ax.set_title(title)
    ax.set_xlabel("Resolution (imgsz)")
    ax.set_ylabel(y_label)
    ax.set_xticks(resolutions)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot mAP vs resolution for YOLOv8 n/s/m/l/x on MOT17.")
    parser.add_argument(
        "--csv",
        default=os.path.join(
            os.path.dirname(__file__),
            "resolution_sweep_outputs",
            "resolution_sweep_results.csv",
        ),
        help="Path to resolution sweep CSV",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__),
            "resolution_sweep_outputs",
            "mot17det_map_vs_resolution.png",
        ),
        help="Output plot path (.png; falls back to .svg if matplotlib is unavailable)",
    )
    parser.add_argument(
        "--time-out",
        default=os.path.join(
            os.path.dirname(__file__),
            "resolution_sweep_outputs",
            "mot17det_time_vs_resolution.png",
        ),
        help="Output path for benchmark-time plot (.png; falls back to .svg if matplotlib is unavailable)",
    )
    parser.add_argument("--benchmark", default="MOT17", help="Benchmark name to filter (default: MOT17)")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    map_data, time_data, resolutions = load_resolution_sweep(args.csv, benchmark=args.benchmark)
    if not resolutions:
        raise SystemExit(f"No rows found for benchmark '{args.benchmark}' in CSV: {args.csv}")

    map_plot_path = plot_resolution_vs_metric(
        map_data,
        resolutions,
        args.out,
        args.show,
        title=f"{args.benchmark}Det: mAP vs Resolution (YOLOv8 n/s/m/l/x)",
        y_label="mAP",
    )
    print(f"Saved mAP plot: {map_plot_path}")

    if any(time_data.get(size, {}) for size in SIZE_ORDER):
        time_plot_path = plot_resolution_vs_metric(
            time_data,
            resolutions,
            args.time_out,
            args.show,
            title=f"{args.benchmark}Det: Benchmark Time vs Resolution (YOLOv8 n/s/m/l/x)",
            y_label="Benchmark Time (sec)",
        )
        print(f"Saved benchmark-time plot: {time_plot_path}")
    else:
        print("No benchmark_time_sec values found in CSV; skipped benchmark-time plot.")


if __name__ == "__main__":
    main()
