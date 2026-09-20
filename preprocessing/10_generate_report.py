"""
10_generate_report.py
----------------------
Aggregates all outputs from the preprocessing pipeline steps and
generates a single, comprehensive HTML report summarizing:

  • Dataset exploration (Step 01)
  • Corrupted images removed (Step 02)
  • Duplicates removed (Step 03)
  • Resize parameters (Step 04)
  • Normalization statistics (Step 05)
  • Augmentation results (Step 06)
  • Dataset split (Step 07)
  • Visualizations (Step 08)
  • TF Dataset configuration (Step 09)

The report is self-contained HTML with embedded images.
"""

import sys
import io
import json
import base64
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "preprocessing" / "outputs"
REPORT_DIR = BASE_DIR / "preprocessing" / "outputs" / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def image_to_b64(path: Path) -> str | None:
    """Encode an image file to base64 for embedding in HTML."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = path.suffix.lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        return f"data:image/{mime};base64,{data}"
    except Exception:
        return None


def embed_image(path: Path, caption: str = "", width: str = "100%") -> str:
    b64 = image_to_b64(path)
    if b64 is None:
        return f'<p class="missing">⚠ Image not found: {path.name}</p>'
    return f"""
    <figure>
        <img src="{b64}" alt="{caption}" style="width:{width}; border-radius:8px;" />
        <figcaption>{caption}</figcaption>
    </figure>"""


def stat_card(label: str, value: str, color: str = "#3b82f6") -> str:
    return f"""
    <div class="stat-card" style="border-left: 4px solid {color};">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>"""


def section_header(step: str, title: str, icon: str = "📌") -> str:
    return f"""
    <div class="section-header">
        <span class="step-badge">{step}</span>
        <h2>{icon} {title}</h2>
    </div>"""

# ─────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────

def generate_html() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Load all summaries ─────────────────────────────────
    explore   = load_json(OUTPUTS_DIR / "01_exploration"   / "exploration_summary.json")
    corrupted = load_json(OUTPUTS_DIR / "02_corrupted"     / "removal_summary.json")
    dedup     = load_json(OUTPUTS_DIR / "03_duplicates"    / "dedup_summary.json")
    resize    = load_json(OUTPUTS_DIR / "04_resize"        / "resize_summary.json")
    norm      = load_json(OUTPUTS_DIR / "05_normalize"     / "normalization_config.json")
    augment   = load_json(OUTPUTS_DIR / "06_augmentation"  / "augmentation_summary.json")
    split     = load_json(OUTPUTS_DIR / "07_split"         / "split_summary.json")
    tf_ds     = load_json(OUTPUTS_DIR / "09_tf_dataset"    / "tf_dataset_summary.json")

    # ── Summary cards ──────────────────────────────────────
    total_orig = explore.get("total_images", "–")
    total_after_aug = augment.get("total_final", "–")
    n_classes = explore.get("num_classes", "–")
    split_total = split.get("total", {})
    train_cnt = split_total.get("train", "–")
    val_cnt   = split_total.get("val", "–")
    test_cnt  = split_total.get("test", "–")
    dataset_mean = norm.get("dataset_stats", {}).get("mean", ["–", "–", "–"])

    # ── Per-class table ────────────────────────────────────
    class_rows = ""
    for cls, cnt in explore.get("classes", {}).items():
        aug_info = augment.get("class_breakdown", {}).get(cls, {})
        split_info = split.get("class_breakdown", {}).get(cls, {})
        class_rows += f"""
        <tr>
            <td>{cls}</td>
            <td class="num">{cnt}</td>
            <td class="num">{aug_info.get("augmented", 0)}</td>
            <td class="num">{aug_info.get("total", cnt)}</td>
            <td class="num">{split_info.get("train", "–")}</td>
            <td class="num">{split_info.get("val", "–")}</td>
            <td class="num">{split_info.get("test", "–")}</td>
        </tr>"""

    # ── Image embeds ───────────────────────────────────────
    img_dist = embed_image(OUTPUTS_DIR / "01_exploration"  / "class_distribution.png",
                           "Original class distribution")
    img_aug  = embed_image(OUTPUTS_DIR / "06_augmentation" / "augmentation_distribution.png",
                           "Class distribution after augmentation")
    img_split = embed_image(OUTPUTS_DIR / "07_split"       / "split_distribution.png",
                            "Train / Val / Test split per class")
    img_samples = embed_image(OUTPUTS_DIR / "08_visualize" / "sample_image_grid.png",
                              "Sample images per class")
    img_avg     = embed_image(OUTPUTS_DIR / "08_visualize" / "average_images_per_class.png",
                              "Average (prototype) image per class")
    img_hist    = embed_image(OUTPUTS_DIR / "08_visualize" / "pixel_intensity_histograms.png",
                              "RGB pixel intensity histograms per split")
    img_norm    = embed_image(OUTPUTS_DIR / "05_normalize" / "normalization_stats.png",
                              "Dataset vs ImageNet normalization statistics")
    img_resize  = embed_image(OUTPUTS_DIR / "04_resize"    / "resize_distribution.png",
                              "Image size distribution before/after resize")
    img_aug_samp = embed_image(OUTPUTS_DIR / "06_augmentation" / "augmented_samples.png",
                               "Sample augmented images")

    mean_str = " | ".join([f"{v:.4f}" for v in dataset_mean]) if isinstance(dataset_mean, list) else str(dataset_mean)
    dataset_std = norm.get("dataset_stats", {}).get("std", ["–","–","–"])
    std_str = " | ".join([f"{v:.4f}" for v in dataset_std]) if isinstance(dataset_std, list) else str(dataset_std)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Preprocessing Report — Skin Disease Detection</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    line-height: 1.6;
  }}
  .header {{
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 60%);
    padding: 40px 48px;
    border-bottom: 1px solid #1e3a5f;
  }}
  .header h1 {{ font-size: 2rem; color: #f1f5f9; margin-bottom: 6px; }}
  .header p  {{ color: #94a3b8; font-size: 0.95rem; }}
  .badge {{
    display: inline-block;
    background: #3b82f6;
    color: white;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    margin-right: 8px;
    vertical-align: middle;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin: 24px 0;
  }}
  .stat-card {{
    background: #1e293b;
    border-radius: 10px;
    padding: 18px 20px;
    border-left: 4px solid #3b82f6;
  }}
  .stat-value {{ font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }}
  .stat-label {{ font-size: 0.78rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .section {{
    background: #1e293b;
    border-radius: 12px;
    padding: 28px 32px;
    margin: 24px 0;
    border: 1px solid #334155;
  }}
  .section-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 20px;
    border-bottom: 1px solid #334155;
    padding-bottom: 14px;
  }}
  .section-header h2 {{ font-size: 1.15rem; color: #f1f5f9; }}
  .step-badge {{
    background: #0f172a;
    color: #3b82f6;
    border: 1px solid #3b82f6;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 600;
    white-space: nowrap;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-top: 12px;
  }}
  th {{
    background: #0f172a;
    color: #94a3b8;
    padding: 10px 12px;
    text-align: left;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
  }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #1e293b; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:hover td {{ background: #334155; }}
  figure {{ margin: 16px 0; text-align: center; }}
  figcaption {{ font-size: 0.78rem; color: #64748b; margin-top: 6px; }}
  img {{ max-width: 100%; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 12px; }}
  .info-item {{
    background: #0f172a;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
  }}
  .info-item .key {{ color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 4px; }}
  .info-item .val {{ color: #f1f5f9; font-weight: 600; }}
  .missing {{ color: #f59e0b; font-style: italic; font-size: 0.85rem; }}
  code {{
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 10px 14px;
    display: block;
    font-size: 0.82rem;
    color: #7dd3fc;
    margin-top: 10px;
    white-space: pre-wrap;
  }}
  .footer {{
    text-align: center;
    color: #475569;
    font-size: 0.8rem;
    padding: 32px 0;
    border-top: 1px solid #1e293b;
    margin-top: 40px;
  }}
  @media (max-width: 600px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>🔬 Preprocessing Pipeline Report</h1>
  <p>
    <span class="badge">AI Skin Disease Detection</span>
    Generated: {now}
  </p>
</div>

<div class="container">

  <!-- ── Executive Summary ── -->
  <div class="cards">
    {stat_card("Original Images", f"{total_orig:,}" if isinstance(total_orig, int) else str(total_orig), "#3b82f6")}
    {stat_card("After Augmentation", f"{total_after_aug:,}" if isinstance(total_after_aug, int) else str(total_after_aug), "#10b981")}
    {stat_card("Classes", str(n_classes), "#8b5cf6")}
    {stat_card("Train Images", f"{train_cnt:,}" if isinstance(train_cnt, int) else str(train_cnt), "#f59e0b")}
    {stat_card("Val Images",   f"{val_cnt:,}"   if isinstance(val_cnt, int)   else str(val_cnt),   "#06b6d4")}
    {stat_card("Test Images",  f"{test_cnt:,}"  if isinstance(test_cnt, int)  else str(test_cnt),  "#f43f5e")}
  </div>

  <!-- ── Step 01: Exploration ── -->
  <div class="section">
    {section_header("Step 01", "Dataset Exploration", "🗂")}
    <div class="info-grid">
      <div class="info-item"><div class="key">Total Images</div><div class="val">{explore.get("total_images", "–"):,}</div></div>
      <div class="info-item"><div class="key">Readable Images</div><div class="val">{explore.get("readable_images", "–"):,}</div></div>
      <div class="info-item"><div class="key">Corrupted</div><div class="val">{explore.get("corrupted_images", "–")}</div></div>
      <div class="info-item"><div class="key">Number of Classes</div><div class="val">{explore.get("num_classes", "–")}</div></div>
      <div class="info-item"><div class="key">Mean Width</div><div class="val">{explore.get("image_size", {}).get("mean_width", "–"):.0f} px</div></div>
      <div class="info-item"><div class="key">Mean Height</div><div class="val">{explore.get("image_size", {}).get("mean_height", "–"):.0f} px</div></div>
    </div>
    {img_dist}
  </div>

  <!-- ── Step 02 + 03: Cleaning ── -->
  <div class="section">
    {section_header("Steps 02–03", "Data Cleaning", "🧹")}
    <div class="two-col">
      <div>
        <h3 style="color:#94a3b8; font-size:0.9rem; margin-bottom:10px;">Corrupted Images (Step 02)</h3>
        <div class="info-grid">
          <div class="info-item"><div class="key">Scanned</div><div class="val">{corrupted.get("total_scanned", "–"):,}</div></div>
          <div class="info-item"><div class="key">Removed</div><div class="val">{corrupted.get("corrupted_removed", "–")}</div></div>
          <div class="info-item"><div class="key">Healthy</div><div class="val">{corrupted.get("healthy", "–"):,}</div></div>
        </div>
      </div>
      <div>
        <h3 style="color:#94a3b8; font-size:0.9rem; margin-bottom:10px;">Duplicate Images (Step 03)</h3>
        <div class="info-grid">
          <div class="info-item"><div class="key">Scanned</div><div class="val">{dedup.get("total_scanned", "–"):,}</div></div>
          <div class="info-item"><div class="key">Removed</div><div class="val">{dedup.get("duplicates_removed", "–")}</div></div>
          <div class="info-item"><div class="key">Hash Threshold</div><div class="val">{dedup.get("hash_threshold", "–")}</div></div>
          <div class="info-item"><div class="key">Unique Remaining</div><div class="val">{dedup.get("unique_remaining", "–"):,}</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── Step 04: Resize ── -->
  <div class="section">
    {section_header("Step 04", "Image Resizing", "📐")}
    <div class="info-grid">
      <div class="info-item"><div class="key">Target Size</div><div class="val">{resize.get("target_width","–")} × {resize.get("target_height","–")} px</div></div>
      <div class="info-item"><div class="key">Strategy</div><div class="val">{resize.get("resize_strategy","–")}</div></div>
      <div class="info-item"><div class="key">Output Format</div><div class="val">{resize.get("output_format","–")}</div></div>
      <div class="info-item"><div class="key">Successfully Resized</div><div class="val">{resize.get("success","–"):,}</div></div>
      <div class="info-item"><div class="key">Errors</div><div class="val">{resize.get("errors","–")}</div></div>
    </div>
    {img_resize}
  </div>

  <!-- ── Step 05: Normalization ── -->
  <div class="section">
    {section_header("Step 05", "Normalization Statistics", "📊")}
    <div class="info-grid">
      <div class="info-item"><div class="key">Channel Mean (R|G|B)</div><div class="val">{mean_str}</div></div>
      <div class="info-item"><div class="key">Channel Std (R|G|B)</div><div class="val">{std_str}</div></div>
      <div class="info-item"><div class="key">Sample Size</div><div class="val">{norm.get("sample_size","–"):,}</div></div>
      <div class="info-item"><div class="key">Recommendation</div><div class="val">{norm.get("recommendation","–")}</div></div>
    </div>
    <code>
# TensorFlow / Keras
tf.keras.layers.Normalization(mean={dataset_mean}, variance={[round(s**2,6) for s in (dataset_std if isinstance(dataset_std, list) else [0,0,0])]})

# PyTorch
transforms.Normalize(mean={dataset_mean}, std={dataset_std})
    </code>
    {img_norm}
  </div>

  <!-- ── Step 06: Augmentation ── -->
  <div class="section">
    {section_header("Step 06", "Data Augmentation", "🔀")}
    <div class="info-grid">
      <div class="info-item"><div class="key">Target per Class</div><div class="val">{augment.get("target_per_class","–"):,}</div></div>
      <div class="info-item"><div class="key">Original Total</div><div class="val">{augment.get("total_original","–"):,}</div></div>
      <div class="info-item"><div class="key">Augmented Total</div><div class="val">{augment.get("total_augmented","–"):,}</div></div>
      <div class="info-item"><div class="key">Final Total</div><div class="val">{augment.get("total_final","–"):,}</div></div>
    </div>
    <div class="two-col" style="margin-top:16px;">
      {img_aug}
      {img_aug_samp}
    </div>
  </div>

  <!-- ── Step 07: Split ── -->
  <div class="section">
    {section_header("Step 07", "Dataset Split", "✂")}
    <div class="info-grid">
      <div class="info-item"><div class="key">Train</div><div class="val">{train_cnt:,} images ({split.get("split_ratios",{}).get("train",0):.0%})</div></div>
      <div class="info-item"><div class="key">Validation</div><div class="val">{val_cnt:,} images ({split.get("split_ratios",{}).get("val",0):.0%})</div></div>
      <div class="info-item"><div class="key">Test</div><div class="val">{test_cnt:,} images ({split.get("split_ratios",{}).get("test",0):.0%})</div></div>
      <div class="info-item"><div class="key">Random Seed</div><div class="val">{split.get("random_seed","–")}</div></div>
    </div>
    {img_split}

    <!-- Per-class table -->
    <table>
      <thead>
        <tr>
          <th>Class</th><th>Original</th><th>Augmented</th><th>Total</th>
          <th>Train</th><th>Val</th><th>Test</th>
        </tr>
      </thead>
      <tbody>
        {class_rows}
      </tbody>
    </table>
  </div>

  <!-- ── Step 08: Visualizations ── -->
  <div class="section">
    {section_header("Step 08", "Dataset Visualizations", "🖼")}
    {img_samples}
    {img_avg}
    {img_hist}
  </div>

  <!-- ── Step 09: TF Dataset ── -->
  <div class="section">
    {section_header("Step 09", "TensorFlow Dataset Pipeline", "⚡")}
    <div class="info-grid">
      <div class="info-item"><div class="key">Image Size</div><div class="val">{tf_ds.get("image_size",["–","–","–"])[0]} × {tf_ds.get("image_size",["–","–","–"])[1]} × {tf_ds.get("image_size",["–","–","–"])[2]}</div></div>
      <div class="info-item"><div class="key">Batch Size</div><div class="val">{tf_ds.get("batch_size","–")}</div></div>
      <div class="info-item"><div class="key">Augmentation</div><div class="val">{"Enabled" if tf_ds.get("augmentation_enabled") else "Disabled"}</div></div>
      <div class="info-item"><div class="key">Splits Built</div><div class="val">{", ".join(tf_ds.get("splits_built",[]))}</div></div>
    </div>
    <code>
# Load and use the preprocessed dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/train",
    image_size=({tf_ds.get("image_size",["224","224"])[0]}, {tf_ds.get("image_size",["224","224"])[1]}),
    batch_size={tf_ds.get("batch_size",32)},
    label_mode="categorical",
)
    </code>
  </div>

</div>

<div class="footer">
  Preprocessing Pipeline Report — AI-Based Skin Disease Detection &amp; Preliminary Diagnosis System<br>
  Generated on {now}
</div>

</body>
</html>"""

    return html

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print_section("GENERATE PREPROCESSING REPORT — Skin Disease Detection")
    print(f"Reading outputs from : {OUTPUTS_DIR}")
    print(f"Report saved to      : {REPORT_DIR}")

    html_content = generate_html()

    report_path = REPORT_DIR / "preprocessing_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print_section("Report Generation Complete ✓")
    print(f"  📄 Report: {report_path}")
    print(f"  Open in your browser to view the full report.\n")


if __name__ == "__main__":
    main()
