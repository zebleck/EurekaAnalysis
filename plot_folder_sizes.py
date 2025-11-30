import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

trainingruns_path = r'C:\Users\Kontor\AppData\LocalLow\DefaultCompany\AR4\EurekaRewards'

folder_sizes = []
folder_names = []

for folder in os.listdir(trainingruns_path):
    folder_path = os.path.join(trainingruns_path, folder)
    if os.path.isdir(folder_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except:
                    pass
        folder_sizes.append(total_size / (1024 * 1024))  # Convert to MB
        folder_names.append(folder)

# Sort by date (folder name)
sorted_data = sorted(zip(folder_names, folder_sizes))
folder_names_sorted = [x[0] for x in sorted_data]
folder_sizes_sorted = [max(x[1], 0.1) for x in sorted_data]  # Avoid log(0)

plt.figure(figsize=(20, 10))
plt.bar(range(len(folder_sizes_sorted)), folder_sizes_sorted, color='steelblue', width=0.8)
plt.yscale('log')
plt.ylabel('Size (MB) - Log Scale', fontsize=12)
plt.xlabel('Training Run', fontsize=12)
plt.title('Training Runs Folder Sizes', fontsize=14)

# Clear y-axis ticks
plt.yticks([0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1000'])
plt.grid(axis='y', alpha=0.3, which='both')

# X-axis: show all folder names
plt.xticks(range(len(folder_names_sorted)), folder_names_sorted,
           rotation=90, ha='center', fontsize=6)

plt.tight_layout()
plt.savefig(r'C:\Users\Kontor\AppData\LocalLow\DefaultCompany\AR4\folder_sizes.png', dpi=150)
plt.show()

print(f'Total folders: {len(folder_sizes_sorted)}')
print(f'Total size: {sum(folder_sizes_sorted):.1f} MB')
