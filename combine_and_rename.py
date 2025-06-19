import os
import shutil
import glob

def combine_datasets(test_dir='../test', train_dir='../train', output_dir='../combined_data'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files from both directories
    test_files = glob.glob(os.path.join(test_dir, '*.npy'))
    train_files = glob.glob(os.path.join(train_dir, '*.npy'))
    
    # Process and copy files from test directory
    for file_path in test_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_dir, filename)
        shutil.copy2(file_path, dest_path)
        print(f"Copied: {filename}")
    
    # Process and copy files from train directory, renaming if necessary
    for file_path in train_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_dir, filename)
        
        # If file already exists, modify the name
        if os.path.exists(dest_path):
            # Split the filename to insert _1 before the letter
            # Example: acoustic_diff_0_Y.npy -> acoustic_diff_0_1_Y.npy
            base_parts = filename.rsplit('_', 1)  # Split at last underscore
            new_filename = f"{base_parts[0]}_1_{base_parts[1]}"
            dest_path = os.path.join(output_dir, new_filename)
            print(f"Renaming duplicate: {filename} -> {new_filename}")
        
        shutil.copy2(file_path, dest_path)
        print(f"Copied: {os.path.basename(dest_path)}")

if __name__ == "__main__":
    combine_datasets()
    print("Dataset combination complete!") 