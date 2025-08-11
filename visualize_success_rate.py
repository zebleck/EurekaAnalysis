import os
import json
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EurekaRewards", "results")

# List all JSON files in the results directory
files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]

if not files:
    print("No JSON result files found in the directory.")
    exit(1)

print("Available result files:")
for idx, fname in enumerate(files):
    print(f"{idx + 1}: {fname}")

# Prompt user to select a file
while True:
    try:
        selection = int(input(f"Select a file [1-{len(files)}]: "))
        if 1 <= selection <= len(files):
            selected_file = files[selection - 1]
            break
        else:
            print("Invalid selection. Try again.")
    except ValueError:
        print("Please enter a valid number.")

selected_path = os.path.join(RESULTS_DIR, selected_file)

# Load the selected JSON file
with open(selected_path, 'r') as f:
    data = json.load(f)

# Extract step numbers and success rates from checkpoints
steps = []
success_rates = []
for entry in data.get('results', []):
    if entry.get('type') == 'checkpoint':
        steps.append(entry.get('step_number', 0))
        success_rates.append(entry.get('success_rate', 0))

if not steps:
    print("No checkpoint data found in the selected file.")
    exit(1)

# Plot success rate over time
plt.figure(figsize=(8, 5))
plt.plot(steps, success_rates, marker='o')
plt.title('Success Rate Over Time')
plt.xlabel('Step Number')
plt.ylabel('Success Rate')
plt.grid(True)
plt.tight_layout()
plt.show() 