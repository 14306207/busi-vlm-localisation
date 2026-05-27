from pathlib import Path
import subprocess
import os
from dotenv import load_dotenv
from imagededup.methods import CNN
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

busi_dataset = 'sabahesaraki/breast-ultrasound-images-dataset'
busi_folder = 'Dataset_BUSI_with_GT'

# Download the BUSI dataset from Kaggle.
def get_busi(output_path=None):
    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / '.env'

    load_dotenv(env_path, override=True)

    username = os.getenv('kaggle_username')
    api_key = os.getenv('kaggle_api_key')

    if not username or not api_key:
        raise ValueError('missing kaggle credentials')

    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = api_key

    if output_path is None:
        output_path = project_root / 'dataset' / 'raw'
    else:
        output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    busi_raw = output_path / busi_folder

    if busi_raw.exists():
        return busi_raw

    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', busi_dataset,
        '-p', str(output_path),
        '--unzip'
    ], check=True)

    if not busi_raw.exists():
        raise FileNotFoundError(f'busi folder not found: {busi_raw}')

    return busi_raw

'''
Referred to imagededup for CNN-based duplicate and near-duplicate detection of BUSI images.
See: https://github.com/idealo/imagededup
'''

def remove_duplicates(source_dir, output_dir, threshold=0.96):
    cnn = CNN()
    stats = {}
    
    for category in ['benign', 'malignant', 'normal']:
        src_path = Path(source_dir) / category

        if not src_path.exists():
            continue
        
        # CNN scans images only, masks follow their corresponding images.
        images = [f for f in src_path.iterdir() if f.is_file() and f.suffix == '.png' and '_mask' not in f.name]
        
        if not images:
            continue
            
        duplicates = set(cnn.find_duplicates_to_remove(
            image_dir=str(src_path),
            min_similarity_threshold=threshold
        ))
        
        image_names = {img.name for img in images}
        duplicate_images = duplicates & image_names
        
        dst_path = Path(output_dir) / category
        dst_path.mkdir(parents=True, exist_ok=True)
        
        kept = 0

        for img in images:
            if img.name not in duplicate_images:
                shutil.copy2(img, dst_path / img.name)
                
                mask_name = img.stem + '_mask.png'
                mask_path = src_path / mask_name
                if mask_path.exists():
                    shutil.copy2(mask_path, dst_path / mask_name)
                
                kept += 1
        
        stats[category] = {'removed': len(duplicate_images), 'kept': kept}
    
    return stats

# Edge-touching lesion detection for BUSI masks. Manually checks if lesion pixels touch mask boundaries using NumPy.
def has_edge_touching_lesion(mask_path):
    # Load mask as numpy array.
    mask = np.array(Image.open(mask_path).convert('L'))

    # Check if any edge pixels are non-zero (lesion present).
    top = np.any(mask[0, :] > 0)
    bottom = np.any(mask[-1, :] > 0)
    left = np.any(mask[:, 0] > 0)
    right = np.any(mask[:, -1] > 0)
    
    return top or bottom or left or right    

# BUSI multi-lesion mask combination for segmentation tasks. Combines split masks (e.g. image_mask.png + image_mask_1.png) into single ground truth.
def combine_busi_masks(mask_dir, output_dir):
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {'multi_lesion': 0, 'single_lesion': 0, 'mismatched': 0}
    
    # Get all base masks.
    base_masks = [f for f in mask_dir.glob('*_mask.png') if not f.stem.endswith('_1')]
    
    for base_mask in base_masks:
        base_name = base_mask.stem.replace('_mask', '')
        
        # Load all masks for this image at once.
        mask_paths = [base_mask] + list(mask_dir.glob(f"{base_name}_mask_*.png"))
        masks = [np.array(Image.open(p).convert('L')) for p in mask_paths]
        
        # Validate sizes.
        if len(set(m.shape for m in masks)) > 1:
            stats['mismatched'] += 1
            continue
        
        # Combine using vectorized reduce.
        combined = np.maximum.reduce(masks) if len(masks) > 1 else masks[0]
        
        Image.fromarray(combined).save(output_dir / base_mask.name)
        
        if len(masks) > 1:
            stats['multi_lesion'] += 1
        else:
            stats['single_lesion'] += 1
    
    return stats