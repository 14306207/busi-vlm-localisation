'''
Referred to PEFT and BiomedCLIP-LoRA examples. This file applies LoRA to the BiomedCLIP vision encoder only. Note: BiomedCLIP uses a timm ViT trunk (so we target attn.qkv and attn.proj). To-Do - add optional text encoder LoRA.
See: https://github.com/huggingface/peft
See: https://github.com/LightersWang/BiomedCLIP-LoRA
See: https://github.com/jinggqu/NextGen-UIA
'''

from peft import LoraConfig, get_peft_model
import torch.nn as nn

# Freeze all model paramters.
def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

# Return PEFT target module names for the BiomedCLIP vision encoder.
def get_target_modules(model, num_layers=None):
    if not (hasattr(model, 'visual') and hasattr(model.visual, 'trunk') and hasattr(model.visual.trunk, 'blocks')):
        raise ValueError('vision blocks not found.')
    
    trunk = model.visual.trunk
    blocks = trunk.blocks
    num_blocks = len(blocks)

    if num_layers is None: 
        layers_to_inject = num_blocks
    else: 
        layers_to_inject = min(num_layers, num_blocks)

    target_modules = []

    for i in range(layers_to_inject):
        block = blocks[i]

        if not hasattr(block, 'attn'):
            raise ValueError(f'block {i} missing attn.')

        if not hasattr(block.attn, 'qkv'):
            raise ValueError(f'block {i} missing attn.qkv.')

        if not hasattr(block.attn, 'proj'):
            raise ValueError(f'block {i} missing attn.proj.')

        if not isinstance(block.attn.qkv, nn.Linear):
            raise ValueError(f'block {i} attn.qkv is not nn.Linear.')

        if not isinstance(block.attn.proj, nn.Linear):
            raise ValueError(f'block {i} attn.proj is not nn.Linear.')

        target_modules.append(f'visual.trunk.blocks.{i}.attn.qkv')
        target_modules.append(f'visual.trunk.blocks.{i}.attn.proj')

    if not target_modules:
        raise ValueError('no LoRA targets found.')

    return target_modules

# Apply LoRA to the BiomedCLIP vision encoder.
def apply_lora(args, model, num_layers=None, lora_rank=16, lora_alpha=32, lora_dropout=0.1):
    freeze_model(model)
    
    if args.encoder == 'vision':
        target_modules = get_target_modules(model=model, num_layers=num_layers)

        peft_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules,  bias='none')
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    else: 
        raise ValueError(f'unsupported encoder {args.encoder} for LoRA injection.')
    
    return model, target_modules

def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total