#!/usr/bin/env python3
"""
Combined script to:
1. Download a dataset from HuggingFace
2. Rename camera observations from ACT format to new format:
   - observation.images.base -> observation.images.cam_high
   - observation.images.wrist -> observation.images.cam_left_wrist
3. Upload the renamed dataset back to HuggingFace
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
import shutil
import json


def rename_camera_keys_in_dataset(source_repo_id, target_repo_id, camera_mappings, local_dir="./temp_dataset"):
    """
    Download LeRobot dataset, rename camera keys, and upload to new repo.
    
    Args:
        source_repo_id: Source dataset repo ID
        target_repo_id: Target dataset repo ID
        camera_mappings: Dict mapping old camera names to new camera names
        local_dir: Temporary directory for processing
    """
    
    # Create local directory
    local_path = Path(local_dir)
    if local_path.exists():
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/3] Downloading dataset from {source_repo_id}...")
    
    # Download the dataset repository
    snapshot_download(
        repo_id=source_repo_id,
        repo_type="dataset",
        local_dir=str(local_path),
    )
    
    print("✓ Downloaded dataset files")
    
    print(f"\n[2/3] Renaming camera observations...")
    
    # Read and modify meta/info.json
    info_path = local_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Rename keys in features
        if 'features' in info:
            for old_key, new_key in camera_mappings.items():
                old_feature_key = f"observation.images.{old_key}"
                new_feature_key = f"observation.images.{new_key}"
                
                if old_feature_key in info['features']:
                    info['features'][new_feature_key] = info['features'].pop(old_feature_key)
                    print(f"  Renamed feature: {old_feature_key} -> {new_feature_key}")
        
        # Write back the modified info
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print("✓ Updated meta/info.json")
    
    # Rename image directories
    data_path = local_path / "data"
    if data_path.exists():
        for old_key, new_key in camera_mappings.items():
            old_dir = data_path / f"observation.images.{old_key}"
            new_dir = data_path / f"observation.images.{new_key}"
            
            if old_dir.exists():
                old_dir.rename(new_dir)
                print(f"  Renamed directory: {old_dir.name} -> {new_dir.name}")
    
    # Update parquet files
    parquet_files = list(data_path.glob("*.parquet")) if data_path.exists() else []
    
    if parquet_files:
        try:
            import pyarrow.parquet as pq
            
            for parquet_file in parquet_files:
                # Read parquet file
                table = pq.read_table(parquet_file)
                
                # Get column names
                columns = table.column_names
                
                # Create mapping for column renaming
                column_mapping = {}
                for old_key, new_key in camera_mappings.items():
                    old_col = f"observation.images.{old_key}"
                    new_col = f"observation.images.{new_key}"
                    if old_col in columns:
                        column_mapping[old_col] = new_col
                
                # Rename columns if needed
                if column_mapping:
                    new_columns = []
                    for col_name in columns:
                        new_columns.append(column_mapping.get(col_name, col_name))
                    
                    table = table.rename_columns(new_columns)
                    
                    # Write back
                    pq.write_table(table, parquet_file)
                    print(f"  Updated parquet file: {parquet_file.name}")
        except ImportError:
            print("  Warning: pyarrow not available, skipping parquet file updates")
            print("  (Parquet files may need manual column renaming)")
    
    print("✓ Camera renaming complete!")
    return local_path


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset from HuggingFace, rename camera observations, and upload"
    )
    parser.add_argument(
        "--source-repo-id",
        type=str,
        default="KHandsome/train50-1",
        help="Source dataset repository ID on HuggingFace (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--target-repo-id",
        type=str,
        default="KHandsome/pi0-train50-1",
        help="Target dataset repository ID on HuggingFace for the renamed dataset"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./temp_dataset",
        help="Temporary local directory to store the dataset"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded dataset private"
    )
    parser.add_argument(
        "--overwrite-target",
        action="store_true",
        help="Overwrite the target repository if it exists (default: False, will fail if repo exists)"
    )
    
    args = parser.parse_args()
    
    # Define camera name mappings
    camera_mappings = {
        "base": "cam_high",
        "wrist": "cam_left_wrist"
    }
    
    print("=" * 70)
    print("Dataset Camera Renaming and Upload Tool")
    print("=" * 70)
    print(f"Source repository: {args.source_repo_id}")
    print(f"Target repository: {args.target_repo_id}")
    print(f"Camera mappings: {camera_mappings}")
    print("=" * 70)
    
    # Process and rename the dataset
    try:
        local_path = rename_camera_keys_in_dataset(
            source_repo_id=args.source_repo_id,
            target_repo_id=args.target_repo_id,
            camera_mappings=camera_mappings,
            local_dir=args.local_dir
        )
        
        print("\n[3/3] Uploading renamed dataset to HuggingFace...")
        
        # Initialize HuggingFace API
        api = HfApi()
        
        # Create repository
        try:
            api.create_repo(
                repo_id=args.target_repo_id,
                repo_type="dataset",
                private=args.private,
                exist_ok=args.overwrite_target
            )
            print(f"✓ Created repository: {args.target_repo_id}")
        except Exception as e:
            if args.overwrite_target:
                print(f"  Repository already exists, will overwrite")
            else:
                print(f"✗ Error: {e}")
                return
        
        # Upload the entire dataset directory
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=args.target_repo_id,
            repo_type="dataset",
        )
        
        print(f"✓ Successfully uploaded renamed dataset!")
        print(f"  Access it at: https://huggingface.co/datasets/{args.target_repo_id}")
        
        # Clean up local directory
        if local_path.exists():
            print(f"\n  Cleaning up temporary directory: {local_path}")
            shutil.rmtree(local_path)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("✓ All steps completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
