import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

folder_path = '../checkpoints'  
x_column = 'step'                
y_column = 'val_ppl'         
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

if not csv_files:
    print(f"No CSV files found in {folder_path}!")


plt.figure(figsize=(12, 7))


for file in csv_files:
    filename = os.path.basename(file)
    
    try:
       
        df = pd.read_csv(file)
       
        if x_column in df.columns and y_column in df.columns:
           
            plt.plot(df[x_column], df[y_column], label=filename, marker='.', alpha=0.8)
        else:
            print(f"Skipping {filename}: Missing '{x_column}' or '{y_column}'.")
            
    except Exception as e:
        print(f"Could not read {filename}. Error: {e}")


plt.title(f'Comparison of {y_column} vs {x_column} across training runs', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.legend(title='CSV Files', bbox_to_anchor=(1.05, 1), loc='upper left')


plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

plt.show()