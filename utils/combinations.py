import json
from itertools import combinations

# Feature numbers from 1 to 11
feature_ids = list(range(1, 12))

# Generate all combinations (excluding empty set)
all_combinations = [
    list(combo)
    for r in range(1, len(feature_ids) + 1)
    for combo in combinations(feature_ids, r)
]

# Define output path
output_path = "./all_feature_combinations.json"

# Write to JSON file
with open(output_path, "w") as f:
    json.dump(all_combinations, f, indent=2)

output_path
