import os
import json
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EurekaRewards", "results")

def visualize_results():
    """
    Lists JSON files in the script's directory, prompts the user to select one,
    and then generates a dual-axis plot showing success rate and mean reward
    components over time.
    """
    try:
        # Get the directory where the script is located
        results_dir = RESULTS_DIR
        
        # List all JSON files in the results directory
        files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

        if not files:
            print("No JSON result files found in the directory.")
            return

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

        selected_path = os.path.join(results_dir, selected_file)

        # Load the selected JSON file
        with open(selected_path, 'r') as f:
            data = json.load(f)

        # First, find all unique reward component names across all checkpoints
        all_component_names = set()
        for entry in data.get('results', []):
            if entry.get('type') == 'checkpoint':
                for component in entry.get('reward_components', []):
                    all_component_names.add(component['name'])
        
        # Initialize data structures
        steps = []
        success_rates = []
        reward_data = {name: [] for name in all_component_names}

        # Extract data from checkpoints
        for entry in data.get('results', []):
            if entry.get('type') == 'checkpoint':
                steps.append(entry.get('step_number', 0))
                success_rates.append(entry.get('success_rate', 0))

                current_step_rewards = {
                    comp['name']: comp['stats']['mean'] 
                    for comp in entry.get('reward_components', [])
                }

                # For each component, append its mean reward, or 0 if not present in this step
                for name in all_component_names:
                    reward_data[name].append(current_step_rewards.get(name, 0))

        if not steps:
            print("No checkpoint data found in the selected file.")
            return

        # Plotting
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot Success Rate on the first y-axis
        color = 'tab:green'
        ax1.set_xlabel('Step Number')
        ax1.set_ylabel('Success Rate', color=color)
        ax1.plot(steps, success_rates, color=color, marker='o', linestyle='-', label='Success Rate')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, which='major', linestyle='--', linewidth=0.5, axis='y')


        # Create a second y-axis for the reward components
        ax2 = ax1.twinx()
        ax2.set_ylabel('Mean Reward')

        # Plot each reward component
        for name, values in reward_data.items():
            ax2.plot(steps, values, marker='x', linestyle='--', label=name)
        
        ax2.tick_params(axis='y')

        # Title and legend
        plt.title(f'Success Rate and Mean Rewards Over Time\n({selected_file})')
        
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        fig.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: The directory for the script was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    visualize_results() 