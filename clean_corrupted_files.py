#!/usr/bin/env python3
"""
Script to clean corrupted files (.tmp and other invalid images) from dataset split files.
This creates backup files before modifying the originals.
"""

import os
import sys
from PIL import Image
from pathlib import Path

def clean_split_file(dataset_path, split_file, dry_run=True):
    """
    Remove corrupted file entries from a split file.

    Args:
        dataset_path: Path to the dataset directory
        split_file: Name of the split file (e.g., 'driver_train.txt')
        dry_run: If True, only report what would be removed without modifying files
    """
    split_path = os.path.join(dataset_path, split_file)

    if not os.path.exists(split_path):
        print(f"Error: {split_path} does not exist")
        return

    valid_lines = []
    corrupted_lines = []

    print(f"\nProcessing {split_file}...")

    with open(split_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            original_line = line
            line = line.strip()

            if not line:
                valid_lines.append(original_line)
                continue

            parts = line.split('\t')
            if len(parts) >= 3:
                img_path = parts[1]
                label = int(parts[2])
                full_path = os.path.join(dataset_path, img_path)

                # Check if this is a .tmp file
                if img_path.endswith('.tmp'):
                    corrupted_lines.append((line_num, img_path, label, "File ends with .tmp"))
                    continue

                # Check if file exists
                if not os.path.exists(full_path):
                    corrupted_lines.append((line_num, img_path, label, "File does not exist"))
                    continue

                # Verify image can be loaded
                try:
                    with Image.open(full_path) as img:
                        img.verify()
                    # Image is valid, keep this line
                    valid_lines.append(original_line)
                except Exception as e:
                    corrupted_lines.append((line_num, img_path, label, f"Cannot load image: {str(e)[:50]}"))
            else:
                # Keep lines that don't match expected format (might be headers)
                valid_lines.append(original_line)

    # Report findings
    print(f"  Total lines: {len(valid_lines) + len(corrupted_lines)}")
    print(f"  Valid lines: {len(valid_lines)}")
    print(f"  Corrupted lines: {len(corrupted_lines)}")

    if corrupted_lines:
        print(f"\n  Corrupted entries found:")
        for line_num, img_path, label, reason in corrupted_lines[:10]:
            print(f"    Line {line_num}: {img_path} (class {label}) - {reason}")
        if len(corrupted_lines) > 10:
            print(f"    ... and {len(corrupted_lines) - 10} more")

    if not dry_run and corrupted_lines:
        # Create backup
        backup_path = split_path + '.backup'
        print(f"\n  Creating backup: {backup_path}")
        with open(split_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())

        # Write cleaned file
        print(f"  Writing cleaned file: {split_path}")
        with open(split_path, 'w') as f:
            f.writelines(valid_lines)

        print(f"  ✓ Successfully cleaned {split_file}")
    elif dry_run and corrupted_lines:
        print(f"\n  [DRY RUN] Would remove {len(corrupted_lines)} corrupted entries")

    return len(corrupted_lines)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Clean corrupted files from dataset split files')
    parser.add_argument('--dataset-path', default='//extra/ThesisProject/Dataset/Cam2',
                       help='Path to dataset directory (default: //extra/ThesisProject/Dataset/Cam2)')
    parser.add_argument('--execute', action='store_true',
                       help='Actually modify files (default is dry-run mode)')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dry_run = not args.execute

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No files will be modified")
        print("Use --execute to actually clean the files")
        print("=" * 60)
    else:
        print("=" * 60)
        print("EXECUTE MODE - Files will be modified (backups will be created)")
        print("=" * 60)

    split_files = ['driver_train.txt', 'driver_val.txt', 'driver_test.txt']
    total_corrupted = 0

    for split_file in split_files:
        count = clean_split_file(dataset_path, split_file, dry_run)
        if count:
            total_corrupted += count

    print(f"\n{'=' * 60}")
    print(f"Summary: Found {total_corrupted} corrupted entries total")

    if dry_run and total_corrupted > 0:
        print("\nTo actually clean the files, run:")
        print(f"  python3 {sys.argv[0]} --execute")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
