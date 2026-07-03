"""
generate_figures.py
Generates interpretability figures using Integrated Gradients
on correctly classified test examples, one figure per class.
"""

import torch
from src.models.convformer import ECGConvFormer
from src.data.dataset import load_dataset, ECGDataset
from src.data.splits import get_inter_patient_split, MITBIH_TEST_INDICES
from src.interpretability.integrated_gradients import (
    compute_attributions,
    select_correctly_classified_examples,
)
from src.interpretability.visualize import (
    plot_attribution,
    plot_class_grid,
    compute_region_attribution_score,
    FIGURES_DIR,
)

device = torch.device("cpu")

# Load model
checkpoint = torch.load("results/checkpoints/convformer_interpatient_v1.pt", map_location=device)
config = checkpoint["config"]
model = ECGConvFormer(
    d_model=config["d_model"], n_heads=config["n_heads"],
    d_ff=config["d_ff"], n_layers=config["n_layers"], dropout=config["dropout"],
)
model.load_state_dict(checkpoint["model_state"])
model.eval()
model.to(device)

# Load test data
beats, labels, rec_ids = load_dataset(db="mit")
_, _, _, _, X_test, y_test = get_inter_patient_split(beats, labels, rec_ids, MITBIH_TEST_INDICES)
test_dataset = ECGDataset(X_test, y_test)

# Find correctly classified examples per class
print("Finding correctly classified examples per class...")
selected = select_correctly_classified_examples(model, test_dataset, device, n_per_class=3)
for cls, indices in selected.items():
    print(f"  Class {cls}: found {len(indices)} examples")

# Generate individual attribution plots and collect region scores
class_names = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
all_region_scores = {c: [] for c in range(5)}

beats_dict, attrs_dict, labels_dict = {}, {}, {}

for cls, indices in selected.items():
    if len(indices) == 0:
        print(f"  Skipping class {cls} — no correctly classified examples found")
        continue

    beats_dict[cls], attrs_dict[cls], labels_dict[cls] = [], [], []

    for i, idx in enumerate(indices):
        beat_tensor, label_tensor = test_dataset[idx]
        beat_batch = beat_tensor.unsqueeze(0)  # (1, 1, 187)

        attribution = compute_attributions(
            model, beat_batch, target_class=cls, device=device, n_steps=50
        )

        beat_np = beat_tensor.squeeze().numpy()

        beats_dict[cls].append(beat_np)
        attrs_dict[cls].append(attribution)
        labels_dict[cls].append((cls, cls))  # true=pred=cls since correctly classified

        region_scores = compute_region_attribution_score(attribution)
        all_region_scores[cls].append(region_scores)

        # Save individual figure for the first example of each class
        if i == 0:
            save_path = FIGURES_DIR / f"attribution_class_{class_names[cls]}.png"
            plot_attribution(
                beat_np, attribution, true_label=cls, pred_label=cls,
                save_path=str(save_path),
            )

# Generate grid figure (all classes, all examples)
print("\nGenerating grid figure...")
grid_path = FIGURES_DIR / "attribution_grid_all_classes.png"
plot_class_grid(beats_dict, attrs_dict, labels_dict, save_path=str(grid_path))

# Print region attribution summary
print("\n── Clinical Region Attribution Summary ──")
for cls, scores_list in all_region_scores.items():
    if len(scores_list) == 0:
        continue
    avg_scores = {}
    for region in scores_list[0].keys():
        avg_scores[region] = sum(s[region] for s in scores_list) / len(scores_list)
    print(f"\nClass {class_names[cls]}:")
    for region, score in avg_scores.items():
        print(f"  {region:<10}: {score:.3f}")

print(f"\nAll figures saved to {FIGURES_DIR}")
