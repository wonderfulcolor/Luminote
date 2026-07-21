from __future__ import annotations

import json
import math

from luminote.affect import PrototypeEstimator
from luminote.config import settings
from luminote.model import LocalModel


def main() -> None:
    data = json.loads(settings.coordinates_path.read_text())
    model = LocalModel(settings.model_name)
    estimator = PrototypeEstimator(settings.prototype_path, settings.prototype_meta_path)
    candidates = list(range(model.layer_count // 2, model.layer_count))
    totals = {layer: {"error": 0.0, "quadrant": 0, "count": 0} for layer in candidates}

    for concept in data["concepts"]:
        expected = (concept["valence"], concept["arousal"])
        for template in data["held_out_templates"]:
            prompt = template.format(concept=concept["name"])
            boundary = model.inspect_boundary([{"role": "user", "content": prompt}])
            for layer in candidates:
                estimate = estimator.estimate(boundary.by_layer[layer], layer)
                totals[layer]["error"] += math.dist(expected, (estimate.valence, estimate.arousal))
                totals[layer]["quadrant"] += int(
                    estimate.valence * expected[0] >= 0 and estimate.arousal * expected[1] >= 0
                )
                totals[layer]["count"] += 1

    rows = []
    for layer, score in totals.items():
        count = score["count"]
        rows.append((score["error"] / count, layer, score["quadrant"] / count))
    rows.sort()
    print("layer  mean_coordinate_error  quadrant_accuracy")
    for error, layer, accuracy in rows:
        print(f"{layer:>5}  {error:>21.4f}  {accuracy:>17.1%}")
    print(f"\nRecommended provisional layer: {rows[0][1]}")


if __name__ == "__main__":
    main()

