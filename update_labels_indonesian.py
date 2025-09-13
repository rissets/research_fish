#!/usr/bin/env python3
"""
Script untuk mengupdate label dataset ikan ke bahasa Indonesia
Author: AI Assistant
Date: 2025-09-13
"""

import yaml
import shutil
from pathlib import Path

def update_dataset_labels_to_indonesian(dataset_path):
    """Update dataset labels dari bahasa Inggris ke bahasa Indonesia"""
    
    dataset_path = Path(dataset_path)
    data_yaml_path = dataset_path / "data.yaml"
    
    if not data_yaml_path.exists():
        print(f"❌ File data.yaml tidak ditemukan: {data_yaml_path}")
        return False
    
    # Mapping nama ikan ke bahasa Indonesia
    fish_name_mapping = {
        'Black Sea Sprat': 'Ikan Sprat Laut Hitam',
        'Gilt-Head Bream': 'Ikan Bawal Emas',
        'Hourse Mackerel': 'Ikan Kembung Kuda',
        'Red Mullet': 'Ikan Kuniran Merah',
        'Red Sea Bream': 'Ikan Bawal Merah',
        'Sea Bass': 'Ikan Kakap Laut',
        'Shrimp': 'Udang',
        'Striped Red Mullet': 'Ikan Kuniran Bergaris',
        'Trout': 'Ikan Trout',
        'Anchovy': 'Ikan Teri',
        'Sardine': 'Ikan Sarden',
        'Mackerel': 'Ikan Kembung',
        'Tuna': 'Ikan Tuna',
        'Salmon': 'Ikan Salmon',
        'Cod': 'Ikan Kod',
        'Herring': 'Ikan Hering',
        'Plaice': 'Ikan Lidah Buntal',
        'Sole': 'Ikan Lidah',
        'Whiting': 'Ikan Whiting',
        'Haddock': 'Ikan Haddock',
        'Pollock': 'Ikan Pollock',
        'Flounder': 'Ikan Sebelah',
        'Turbot': 'Ikan Turbot',
        'Halibut': 'Ikan Halibut',
        'Monkfish': 'Ikan Setan Laut',
        'John Dory': 'Ikan John Dory',
        'Gurnard': 'Ikan Gurnard',
        'Brill': 'Ikan Brill',
        'Dab': 'Ikan Dab',
        'Lemon Sole': 'Ikan Lidah Lemon',
        'Dover Sole': 'Ikan Lidah Dover',
        'Bass': 'Ikan Bass',
        'Catfish': 'Ikan Lele',
        'Carp': 'Ikan Mas',
        'Snapper': 'Ikan Kakap',
        'Grouper': 'Ikan Kerapu',
        'Barracuda': 'Ikan Alu-alu',
        'Pomfret': 'Ikan Bawal',
        'Skipjack': 'Ikan Cakalang',
        'Yellowfin': 'Ikan Tuna Sirip Kuning',
        'Mahi-mahi': 'Ikan Lemadang',
        'Swordfish': 'Ikan Todak'
    }
    
    # Backup original file
    backup_path = data_yaml_path.with_suffix('.yaml.backup')
    if not backup_path.exists():
        shutil.copy2(data_yaml_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
    
    # Load current config
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"📊 Current dataset config:")
    print(f"   Classes: {config.get('nc', 'Unknown')}")
    print(f"   Current names: {config.get('names', [])}")
    
    # Convert names to Indonesian
    original_names = config.get('names', [])
    indonesian_names = []
    
    print(f"\n🔄 Converting to Indonesian:")
    for i, english_name in enumerate(original_names):
        indonesian_name = fish_name_mapping.get(english_name, english_name)
        indonesian_names.append(indonesian_name)
        print(f"  {i}: {english_name} → {indonesian_name}")
    
    # Update config
    config['names'] = indonesian_names
    
    # Save updated config
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✅ Dataset labels updated to Indonesian!")
    print(f"   File: {data_yaml_path}")
    print(f"   Classes: {len(indonesian_names)}")
    print(f"   New names: {indonesian_names[:5]}..." if len(indonesian_names) > 5 else f"   New names: {indonesian_names}")
    
    return True

def main():
    import sys
    
    print("🐟 Update Dataset Labels ke Bahasa Indonesia")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Cari dataset yang ada
        possible_paths = [
            "./datasets/FishImgDataset_yolo",
            "./datasets/fish_large_scale",
            "./datasets/fish_sample"
        ]
        
        print("Dataset yang tersedia:")
        available_datasets = []
        for i, path in enumerate(possible_paths, 1):
            if Path(path).exists():
                available_datasets.append(path)
                print(f"  {i}. {path}")
        
        if not available_datasets:
            print("❌ Tidak ada dataset yang ditemukan")
            print("Gunakan: python update_labels_indonesian.py <path_to_dataset>")
            return
        
        try:
            choice = int(input(f"\nPilih dataset (1-{len(available_datasets)}): ")) - 1
            dataset_path = available_datasets[choice]
        except (ValueError, IndexError):
            print("❌ Pilihan tidak valid")
            return
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset tidak ditemukan: {dataset_path}")
        return
    
    success = update_dataset_labels_to_indonesian(dataset_path)
    
    if success:
        print(f"\n🎉 Update selesai!")
        print(f"Dataset siap untuk training dengan label bahasa Indonesia.")
        print(f"\nContoh training:")
        print(f"python train_fish_segmentation.py --data {dataset_path}/data.yaml --epochs 100")
    else:
        print(f"\n❌ Update gagal!")

if __name__ == "__main__":
    main()
