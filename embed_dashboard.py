import base64
import os

imgs = {
    'demo_adaptation_spot6_patch000.png': 'outputs/visualizations/demo_adaptation_spot6_patch000.png',
    'demo_adaptation_sentinel1_sar_patch000.png': 'outputs/visualizations/demo_adaptation_sentinel1_sar_patch000.png',
    'multi_domain_translation_matrix.png': 'outputs/visualizations/multi_domain_translation_matrix.png',
    'segmentation_eval_sentinel2_to_spot6.png': 'outputs/visualizations/segmentation_eval_sentinel2_to_spot6.png',
    'segmentation_eval_sentinel2_to_sentinel1_sar.png': 'outputs/visualizations/segmentation_eval_sentinel2_to_sentinel1_sar.png',
    'continuous_style_interpolation.png': 'outputs/visualizations/continuous_style_interpolation.png',
    'multi_angle_view_synthesis.png': 'outputs/visualizations/multi_angle_view_synthesis.png',
    'segmentation_eval_nadir_0_to_oblique_45.png': 'outputs/visualizations/segmentation_eval_nadir_0_to_oblique_45.png',
}

b64_map = {}
for basename, path in imgs.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            b64_map[basename] = base64.b64encode(f.read()).decode()
        print(f"[OK] Loaded {basename} ({len(b64_map[basename])//1024} KB base64)")
    else:
        print(f"[MISSING] {path}")

with open('visualizer_dashboard.html', 'r', encoding='utf-8') as f:
    html = f.read()

for basename, b64 in b64_map.items():
    html = html.replace(
        f'src="outputs/visualizations/{basename}"',
        f'src="data:image/png;base64,{b64}"'
    )

with open('visualizer_dashboard_embedded.html', 'w', encoding='utf-8') as f:
    f.write(html)

size_kb = os.path.getsize('visualizer_dashboard_embedded.html') // 1024
print(f"\n[Done] Wrote visualizer_dashboard_embedded.html ({size_kb} KB) - fully self-contained, no external files needed.")
