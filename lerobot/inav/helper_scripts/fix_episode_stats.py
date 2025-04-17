import json
import numpy as np
from pathlib import Path

# Path to the original episodes_stats JSONL file (update this path if needed)
original_file = Path("/home/demo/lerobot-beavr/lerobot/inav/episode_stats.jsonl")
# Path for the fixed output file; you can change the filename as desired
fixed_file = Path("/home/demo/lerobot-beavr/lerobot/inav/episodes_stats_fixed.jsonl")

fixed_lines = []
with open(original_file, "r") as f:
    for line in f:
        # Parse the JSON for an episode entry
        entry = json.loads(line)
        # Check if the "observation.images" stats are present
        image_stats = entry.get("stats", {}).get("observation.images", None)
        if image_stats:
            # Process each statistic except "count"
            for stat_key in ["min", "max", "mean", "std"]:
                # Convert the value to a numpy array
                arr = np.array(image_stats[stat_key])
                # If the current shape is (1, 3, 1) then swap the first two axes to get (3, 1, 1)
                if arr.shape == (1, 3, 1):
                    arr = np.transpose(arr, (1, 0, 2))
                    image_stats[stat_key] = arr.tolist()
        fixed_lines.append(entry)

# Write out all fixed episodes to the new JSONL file
with open(fixed_file, "w") as f:
    for entry in fixed_lines:
        json.dump(entry, f)
        f.write("\n")

print(f"Fixed episodes_stats file written to: {fixed_file.resolve()}")
