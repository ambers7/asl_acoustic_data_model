import os
import shutil
import glob

def combine_datasets(dir1='/data/asl_fingerspelling/glasses_test_repeat2', dir2='/data/asl_fingerspelling/glasses_test_repeat8', output_dir='/data/asl_fingerspelling/glasses_test_2_combined'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files from both directories
    files1 = glob.glob(os.path.join(dir1, '*.npy'))
    files2 = glob.glob(os.path.join(dir2, '*.npy'))
    
    print(f"Found {len(files1)} files in {dir1} directory")
    print(f"Found {len(files2)} files in {dir2} directory")
    
    # Process and copy files from the first directory
    for file_path in files1:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_dir, filename)
        shutil.copy2(file_path, dest_path)
        print(f"Copied from {dir1}: {filename}")
    
    # Process and copy files from the second directory, renaming if necessary
    for file_path in files2:
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
                    print(f"Renaming duplicate from {dir2}: {filename} -> {new_filename}")
                    break
                counter += 1
        
        shutil.copy2(file_path, dest_path)
        print(f"Copied from {dir2}: {os.path.basename(dest_path)}")

    # Count total files in combined directory
    combined_files = glob.glob(os.path.join(output_dir, '*.npy'))
    print("\nSummary:")
    print(f"Total files in {output_dir}: {len(combined_files)}")
    print(f"Original files: {len(files1)} ({dir1}) + {len(files2)} ({dir2}) = {len(files1) + len(files2)}")
    print(f"Number of renamed files: {len(combined_files) - len(files1)}")

if __name__ == "__main__":
    combine_datasets()
    print("Dataset combination complete!") 