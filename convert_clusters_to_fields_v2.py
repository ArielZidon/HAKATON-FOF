"""
Convert cluster_samples_random_clusters.csv to separate field files

This script extracts ALL coordinates from friend_pos_list and foe_pos_list
and creates individual field files with:
- friend coordinates → class 0
- foe coordinates → class 1
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path


def convert_clusters_to_fields(input_file='coordinates/cluster_samples_50_with_time-ish.csv',
                               output_dir='coordinates',
                               start_field_num=2):
    """
    Convert cluster CSV to separate field files.

    Each row contains:
    - friend_pos_list: list of [lat, lon] pairs for friends (class 0)
    - foe_pos_list: list of [lat, lon] pairs for foes (class 1)

    We'll create one field file per row, combining both friend and foe coordinates.
    """

    print("="*70)
    print("CONVERTING CLUSTER DATA TO FIELD FILES (V2)")
    print("="*70)

    # Read the cluster data
    df = pd.read_csv(input_file)
    print(f"\nLoaded {len(df)} rows from {input_file}")
    print(f"Columns: {list(df.columns)}")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # We start from field2 (field1 already exists)
    field_number = start_field_num
    created_files = []

    # Process each row as a separate field
    for idx, row in df.iterrows():
        if pd.isna(row['friend_pos_list']) and pd.isna(row['foe_pos_list']):
            continue

        all_points = []

        # Parse friend positions
        if not pd.isna(row['friend_pos_list']):
            try:
                friend_positions = ast.literal_eval(row['friend_pos_list'])
                for pos in friend_positions:
                    # pos is [latitude, longitude]
                    all_points.append({
                        'latitude': pos[0],
                        'longitude': pos[1],
                        'class': 0  # friend = 0
                    })
            except Exception as e:
                print(f"Error parsing friend_pos_list on row {idx+2}: {e}")

        # Parse foe positions
        if not pd.isna(row['foe_pos_list']):
            try:
                foe_positions = ast.literal_eval(row['foe_pos_list'])
                for pos in foe_positions:
                    # pos is [latitude, longitude]
                    all_points.append({
                        'latitude': pos[0],
                        'longitude': pos[1],
                        'class': 1  # foe = 1
                    })
            except Exception as e:
                print(f"Error parsing foe_pos_list on row {idx+2}: {e}")

        if not all_points:
            print(f"Warning: No points found on row {idx+2}, skipping")
            continue

        # Create field file for this row
        field_name = f"field{field_number}.csv"
        field_path = output_path / field_name

        # Create DataFrame
        points_df = pd.DataFrame(all_points)

        # Save to CSV
        points_df.to_csv(field_path, index=False)
        created_files.append(field_name)

        # Count classes
        n_friends = len([p for p in all_points if p['class'] == 0])
        n_foes = len([p for p in all_points if p['class'] == 1])

        print(f"Created {field_name}: {len(all_points)} points "
              f"({n_friends} friends, {n_foes} foes) from row {idx+2}")

        field_number += 1

    print(f"\n{'='*70}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"Created {len(created_files)} field files (field{start_field_num} - field{field_number-1})")
    print(f"Total fields including field1: {len(created_files) + 1}")

    # Verify total count
    all_field_files = sorted(output_path.glob('field*.csv'))
    print(f"\nTotal field files in {output_dir}: {len(all_field_files)}")

    if len(all_field_files) == 22:
        print("✓ Successfully created 22 field files!")
    else:
        print(f"⚠ Expected 22 files, found {len(all_field_files)}")

    return created_files


def verify_field_files(output_dir='coordinates'):
    """Verify all field files have the correct format."""

    print("\n" + "="*70)
    print("VERIFYING FIELD FILES")
    print("="*70)

    output_path = Path(output_dir)
    field_files = sorted(output_path.glob('field*.csv'), key=lambda x: int(x.stem[5:]))

    print(f"\nFound {len(field_files)} field files")

    total_points = 0
    total_friends = 0
    total_foes = 0

    for field_file in field_files:
        df = pd.read_csv(field_file)
        n_friends = len(df[df['class'] == 0])
        n_foes = len(df[df['class'] == 1])

        total_points += len(df)
        total_friends += n_friends
        total_foes += n_foes

        print(f"{field_file.name:15s}: {len(df):3d} points "
              f"({n_friends:2d} friends, {n_foes:2d} foes)")

    print(f"\n{'='*70}")
    print(f"TOTAL STATISTICS")
    print(f"{'='*70}")
    print(f"Total files:   {len(field_files)}")
    print(f"Total points:  {total_points}")
    print(f"  Friends (0): {total_friends} ({total_friends/total_points*100:.1f}%)")
    print(f"  Foes (1):    {total_foes} ({total_foes/total_points*100:.1f}%)")


if __name__ == "__main__":
    # First, remove old field files (except field1)
    print("Cleaning up old field2-field22 files...")
    output_path = Path('coordinates')
    for i in range(2, 23):
        old_file = output_path / f"field{i}.csv"
        if old_file.exists():
            old_file.unlink()
            print(f"  Removed {old_file.name}")

    print()

    # Convert cluster data to field files
    created_files = convert_clusters_to_fields()

    # Verify the created files
    verify_field_files()

    print("\n✅ All done! You now have field files with all friend and foe coordinates")
    print("\nYou can use these files for cross-validation with:")
    print("  python test_crossval.py --grid-files coordinates/field*.csv")
