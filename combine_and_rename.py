import os
import shutil
import glob

def combine_datasets(test_dir='../test', train_dir='../train', glasses_dir='../glasses_test_1', output_dir='../combined_data'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files from all directories
    test_files = glob.glob(os.path.join(test_dir, '*.npy'))
    train_files = glob.glob(os.path.join(train_dir, '*.npy'))
    glasses_files = glob.glob(os.path.join(glasses_dir, '*.npy'))
    
    print(f"Found {len(test_files)} files in test directory")
    print(f"Found {len(train_files)} files in train directory")
    print(f"Found {len(glasses_files)} files in glasses_test_1 directory")
    
    # Process and copy files from test directory first
    for file_path in test_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_dir, filename)
        shutil.copy2(file_path, dest_path)
        print(f"Copied from test: {filename}")
    
    # Process and copy files from train directory, renaming if necessary
    for file_path in train_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_dir, filename)
        
        # If file already exists, find the next available number
        if os.path.exists(dest_path):
            base_parts = filename.rsplit('_', 1)  # Split at last underscore
            counter = 1
            while True:
                new_filename = f"{base_parts[0]}_{counter}_{base_parts[1]}"
                new_dest_path = os.path.join(output_dir, new_filename)
                if not os.path.exists(new_dest_path):
                    dest_path = new_dest_path
                    print(f"Renaming duplicate from train: {filename} -> {new_filename}")
                    break
                counter += 1
        
        shutil.copy2(file_path, dest_path)
        print(f"Copied from train: {os.path.basename(dest_path)}")

    # Process and copy files from glasses directory, renaming if necessary
    for file_path in glasses_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_dir, filename)
        
        # If file already exists, find the next available number
        if os.path.exists(dest_path):
            base_parts = filename.rsplit('_', 1)  # Split at last underscore
            counter = 1
            while True:
                new_filename = f"{base_parts[0]}_{counter}_{base_parts[1]}"
                new_dest_path = os.path.join(output_dir, new_filename)
                if not os.path.exists(new_dest_path):
                    dest_path = new_dest_path
                    print(f"Renaming duplicate from glasses: {filename} -> {new_filename}")
                    break
                counter += 1
        
        shutil.copy2(file_path, dest_path)
        print(f"Copied from glasses: {os.path.basename(dest_path)}")

    # Count total files in combined directory
    combined_files = glob.glob(os.path.join(output_dir, '*.npy'))
    print("\nSummary:")
    print(f"Total files in combined_data: {len(combined_files)}")
    print(f"Original files: {len(test_files)} (test) + {len(train_files)} (train) + {len(glasses_files)} (glasses) = {len(test_files) + len(train_files) + len(glasses_files)}")
    print(f"Number of renamed files: {len(combined_files) - len(test_files)}")

if __name__ == "__main__":
    combine_datasets()
    print("Dataset combination complete!") 