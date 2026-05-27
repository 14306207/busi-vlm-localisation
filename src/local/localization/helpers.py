from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

'''
BUSSAM is used here for BUSI segmentation/localisation. The segmentation target is background vs lesion. 
'''

# Finds the main project folder so paths work from helper .py files not notebooks.
def get_repo_root():
    return next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())

def _sanitize_busi_name(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_')

def _link_or_copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return 'linked'
    except OSError:
        shutil.copyfile(source, destination)
        return 'copied'

def _bussam_output_path(bussam_root, output_path):
    output_dir = Path(output_path)
    if output_dir.is_absolute():
        return output_dir
    return bussam_root / output_dir

# Points to the cloned BUSSAM repo inside the external folder.
def get_bussam_root():
    return get_repo_root() / 'external' / 'BUSSAM'

# Used to clone the BUSSAM repository for training. Note: you can also do 'git clone https://github.com/bscs12/BUSSAM.git' direclty.
def clone_bussam():
    bussam = get_bussam_root()
    print(f'clone: exists = {bussam.exists()} path = {bussam}')

    if bussam.exists():
        return bussam
    
    bussam.parent.mkdir(parents = True, exist_ok = True)

    subprocess.run(['git', 'clone', 'https://github.com/bscs12/BUSSAM.git', str(bussam)], check = True)

    return bussam

# Downloads the original SAM ViT-B checkpoint that BUSSAM needs before training.
def download_sam_checkpoint():
    bussam = get_bussam_root()
    checkpoint = bussam / 'checkpoints' / 'sam_vit_b_01ec64.pth'

    print(f'sam checkpoint: exists = {checkpoint.exists()} path = {checkpoint}')

    if checkpoint.exists():
        return checkpoint
    
    checkpoint.parent.mkdir(parents = True, exist_ok = True)
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', checkpoint)

    return checkpoint

# Copies the BUSI split images and masks into the folder structure BUSSAM expects.
def prepare_bussam_busi_dataset():
    root = get_repo_root()
    bussam = get_bussam_root()

    split_root = root / 'dataset' / 'split'
    out_img = bussam / 'datasets' / 'BUSI' / 'img'
    out_label = bussam / 'datasets' / 'BUSI' / 'label'
    out_main = bussam / 'datasets' / 'MainPatient'
    manifest_path = out_main / 'BUSI_manifest.json'

    out_img.mkdir(parents=True, exist_ok=True)
    out_label.mkdir(parents=True, exist_ok=True)
    out_main.mkdir(parents=True, exist_ok=True)

    classes = ['benign', 'malignant']

    print(f'dataset: source = {split_root} classes = {classes}')

    split_lists = {'train': [], 'val': [], 'test': []}
    file_plan = []
    skipped_missing_masks = []

    for split in split_lists:
        for cls in classes:
            cls_dir = split_root / split / cls

            if not cls_dir.exists():
                continue

            for image_path in sorted(cls_dir.glob('*.png')):
                if '_mask' in image_path.stem:
                    continue

                mask_path = image_path.with_name(f'{image_path.stem}_mask{image_path.suffix}')

                if not mask_path.exists():
                    skipped_missing_masks.append(str(image_path.relative_to(split_root)))
                    continue

                out_stem = _sanitize_busi_name(f'{cls}__{image_path.stem}')
                out_file = f'{out_stem}.png'

                file_plan.append({
                    'image_src': str(image_path.relative_to(root)).replace('\\', '/'),
                    'mask_src': str(mask_path.relative_to(root)).replace('\\', '/'),
                    'image_size': image_path.stat().st_size,
                    'image_mtime_ns': image_path.stat().st_mtime_ns,
                    'mask_size': mask_path.stat().st_size,
                    'mask_mtime_ns': mask_path.stat().st_mtime_ns,
                    'output_name': out_file,
                })

                if split == 'test':
                    split_lists[split].append(f'BUSI/{out_stem}')
                else:
                    split_lists[split].append(f'1/BUSI/{out_stem}')

    manifest = {
        'classes': classes,
        'counts': {split: len(names) for split, names in split_lists.items()},
        'files': file_plan,
    }

    split_file_paths = {split: out_main / f'BUSI_{split}.txt' for split in split_lists}
    required_outputs = [
        *(out_img / item['output_name'] for item in file_plan),
        *(out_label / item['output_name'] for item in file_plan),
        *split_file_paths.values(),
        out_main / 'class.json',
    ]

    if manifest_path.exists():
        current_manifest = json.loads(manifest_path.read_text())
        if current_manifest == manifest and all(path.exists() for path in required_outputs):
            print(
                'dataset: already prepared '
                f'train/val/test={len(split_lists["train"])}/{len(split_lists["val"])}/{len(split_lists["test"])}'
            )
            return {
                'img_dir': out_img,
                'label_dir': out_label,
                'split_files': split_file_paths,
                'manifest_path': manifest_path,
                'counts': manifest['counts'],
            }

    for png_file in out_img.glob('*.png'):
        png_file.unlink()

    for png_file in out_label.glob('*.png'):
        png_file.unlink()

    link_mode = None

    for item in file_plan:
        image_src = root / item['image_src']
        mask_src = root / item['mask_src']
        image_mode = _link_or_copy_file(image_src, out_img / item['output_name'])
        mask_mode = _link_or_copy_file(mask_src, out_label / item['output_name'])
        link_mode = image_mode if link_mode is None else link_mode

        if image_mode != mask_mode:
            link_mode = 'mixed'

    for split, names in split_lists.items():
        split_file_paths[split].write_text('\n'.join(names) + ('\n' if names else ''))

    (out_main / 'class.json').write_text(json.dumps({'BUSI': 2}) + '\n')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    print(
        f'dataset: train/val/test='
        f'{len(split_lists["train"])}/'
        f'{len(split_lists["val"])}/'
        f'{len(split_lists["test"])}'
    )

    print(f'dataset: files {link_mode or "prepared"} for BUSSAM')

    if skipped_missing_masks:
        print(f'dataset: skipped {len(skipped_missing_masks)} images without masks')

    return {
        'img_dir': out_img,
        'label_dir': out_label,
        'split_files': split_file_paths,
        'manifest_path': manifest_path,
        'counts': manifest['counts'],
    }

# I had to manually patch the config file because BUSSAM reads epochs and output paths from config.py.
def patch_bussam_config(epochs=100, output_dir='outputs/'):
    bussam = get_bussam_root()
    config = bussam / 'utils' / 'config.py'

    print(f'config: epochs={epochs} output={output_dir}')

    text = config.read_text()

    text = text.replace('pre_trained = True', 'pre_trained = False')
    text = re.sub(r'epochs\s*=\s*\d+', f'epochs = {epochs}', text)
    text = re.sub(r'output_path\s*=\s*[\'"].*?[\'"]', f'output_path = \'{output_dir}\'', text)

    config.write_text(text)

    return config

def patch_bussam_repo_imports():
    bussam = get_bussam_root()

    model_dict = bussam / 'models' / 'model_dict.py'
    text = model_dict.read_text()

    text = text.replace(
        'from models.segment_anything_bussam.build_sam_us import bussam_model_registry',
        'from models.segment_anything_samus.build_sam_us import bussam_model_registry'
    )

    model_dict.write_text(text)

    modeling_init = bussam / 'models' / 'segment_anything_samus' / 'modeling' / '__init__.py'
    text = modeling_init.read_text()

    text = text.replace(
        'from .bussam import Bussam',
        'from .samus import Bussam'
    )
    modeling_init.write_text(text)

    print('BUSSAM repo imports patched')

    return {
        'model_dict': model_dict,
        'modeling_init': modeling_init,
    }

# Runs BUSSAM training on the BUSI segmentation task.
def train_bussam(batch_size=4, base_lr=0.0005):
    bussam = get_bussam_root()

    patch_bussam_repo_imports()


    if str(bussam) not in sys.path:
        sys.path.insert(0, str(bussam))

    import importlib
    import random

    import numpy as np
    import torch
    import torch.optim as optim
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    bussam_config = importlib.import_module('utils.config')

    importlib.reload(bussam_config)

    evaluation = importlib.import_module('utils.evaluation')
    get_model = importlib.import_module('models.model_dict').get_model
    data_us = importlib.import_module('utils.data_us')
    get_criterion = importlib.import_module('utils.loss_functions.sam_loss').get_criterion
    get_click_prompt = importlib.import_module('utils.generate_prompts').get_click_prompt

    class Args:
        modelname = 'BUSSAM'
        encoder_input_size = 256
        low_image_size = 128
        task = 'BUSI'
        vit_name = 'vit_b'
        sam_ckpt = 'checkpoints/sam_vit_b_01ec64.pth'
        n_gpu = 1
        warmup = True
        warmup_period = 250

    args = Args()
    args.sam_ckpt = str((bussam / args.sam_ckpt).resolve())
    opt = bussam_config.get_config(args.task)
    opt.data_path = str(_bussam_output_path(bussam, opt.data_path).resolve())
    opt.batch_size = batch_size * args.n_gpu
    device = torch.device(opt.device)

    seed_value = 1234
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True

    timestr = time.strftime('%m%d%H%M%S')
    save_dir = _bussam_output_path(bussam, opt.output_path) / f'{args.modelname}_{timestr}'
    checkpoint_dir = save_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)

    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path, map_location=device)
        new_state_dict = {}

        for key, value in checkpoint.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    tf_train = data_us.JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0.0,
        p_rota=0.5,
        p_scale=0.5,
        p_gaussn=0.0,
        p_contr=0.5,
        p_gama=0.5,
        p_distor=0.0,
        color_jitter_params=None,
        long_mask=True,
    )

    tf_val = data_us.JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0,
        color_jitter_params=None,
        long_mask=True,
    )

    train_dataset = data_us.ImageToImage2D(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
    val_dataset = data_us.ImageToImage2D(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)

    pin_memory = torch.cuda.is_available() and device.type == 'cuda'
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    if args.warmup:
        lr_start = base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_start,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    print('Total_params:{}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    history = []
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice = float('-inf')
    best_checkpoint = None

    for epoch in range(opt.epochs):
        model.train()
        train_losses = 0.0
        train_batches = 0

        progress_bar = tqdm(
            trainloader,
            desc=f"Epoch {epoch + 1}/{opt.epochs}",
            leave=True
        )

        for batch_idx, datapack in enumerate(progress_bar):
            train_batches = batch_idx + 1
            imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype=torch.float32, device=opt.device)

            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)

            pred = model(imgs, pt, bbox)
            train_loss = criterion(pred, masks)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses += train_loss.item()

            if args.warmup and iter_num < args.warmup_period:
                lr_value = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_value

            elif args.warmup:
                shift_iter = iter_num - args.warmup_period
                lr_value = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_value

            else:
                lr_value = optimizer.param_groups[0]['lr']

            running_loss = train_losses / train_batches
            progress_bar.set_postfix(
                loss=f"{running_loss:.4f}",
                lr=f"{lr_value:.2e}"
            )

            iter_num += 1

        train_loss_epoch = train_losses / max(train_batches, 1)

        opt.mode = 'train'
        _, val_dice, _, val_loss = evaluation.get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        opt.mode = 'test'
        _, _, val_iou, val_acc, val_se, val_sp, *_ = evaluation.get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        opt.mode = 'train'

        epoch_metrics = {
            'epoch': epoch + 1,
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'train_loss': float(train_loss_epoch),
            'val_loss': float(val_loss),
            'val_dice': float(np.mean(val_dice[1:]) if hasattr(val_dice, '__len__') else val_dice),
            'val_iou': float(np.mean(val_iou[1:]) if hasattr(val_iou, '__len__') else val_iou),
            'val_accuracy': float(np.mean(val_acc[1:]) if hasattr(val_acc, '__len__') else val_acc),
            'val_sensitivity': float(np.mean(val_se[1:]) if hasattr(val_se, '__len__') else val_se),
            'val_specificity': float(np.mean(val_sp[1:]) if hasattr(val_sp, '__len__') else val_sp),
        }

        history.append(epoch_metrics)

        print(
            f"epoch {epoch + 1:03d}/{opt.epochs} - "
            f"train_loss={epoch_metrics['train_loss']:.4f} - "
            f"val_loss={epoch_metrics['val_loss']:.4f} - "
            f"val_dice={epoch_metrics['val_dice']:.2f} - "
            f"val_acc={epoch_metrics['val_accuracy']:.2f} - "
            f"lr={epoch_metrics['learning_rate']:.6f}",
            flush=True
        )

        if epoch_metrics['val_dice'] > best_dice:
            best_dice = epoch_metrics['val_dice']
            checkpoint_timestr = time.strftime('%m%d%H%M')
            best_checkpoint = checkpoint_dir / f"{args.modelname}_{checkpoint_timestr}_{epoch + 1}_{best_dice}.pth"
            torch.save(model.state_dict(), best_checkpoint, _use_new_zipfile_serialization=False)

        if (epoch + 1) % opt.save_freq == 0 or (epoch + 1) == opt.epochs:
            torch.save(model.state_dict(), checkpoint_dir / f'{args.modelname}_{epoch + 1}.pth', _use_new_zipfile_serialization=False)

    history_path = save_dir / 'history.json'
    history_path.write_text(json.dumps(history, indent=2) + '\n')

    return {
        'save_dir': save_dir,
        'checkpoint_dir': checkpoint_dir,
        'history_path': history_path,
        'best_checkpoint': best_checkpoint,
        'history': history,
    }

# Finds the newest trained checkpoint after BUSSAM has saved model weights.
def find_latest_checkpoint():
    bussam = get_bussam_root()
    checkpoints = sorted((bussam / 'outputs').glob('**/*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)

    return checkpoints[0] if checkpoints else None

def patch_bussam_test_config(checkpoint):
    bussam = get_bussam_root()
    config = bussam / 'utils' / 'config.py'

    text = config.read_text()
    checkpoint = str(checkpoint).replace('\\', '/')

    text = re.sub(r'load_path\s*=\s*[\'"].*?[\'"]', f'load_path = \'{checkpoint}\'', text)
    text = re.sub(r'visual\s*=\s*False', 'visual = True', text)

    config.write_text(text)

    return config

# Runs BUSSAM testing after training.
def test_bussam(batch_size=4):
    bussam = get_bussam_root()

    subprocess.run([
        sys.executable, 'test.py',
        '--task', 'BUSI',
        '--modelname', 'BUSSAM',
        '--encoder_input_size', '256',
        '--low_image_size', '128',
        '--vit_name', 'vit_b',
        '--sam_ckpt', 'checkpoints/sam_vit_b_01ec64.pth',
        '--batch_size', str(batch_size),
        '--n_gpu', '1',
    ], cwd=bussam, check=True)
