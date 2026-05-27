import torch
import open_clip

# Loads the BiomedCLIP model.
def load_biomedclip(device="cuda"):
    model_id = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_id, device=device)
    tokenizer = open_clip.get_tokenizer(model_id)
    model = model.float()
    return model, preprocess_train, preprocess_val, tokenizer

def make_biomedclip_loader(device="cuda"):
    def loader():
        model, _, preprocess_val, _ = load_biomedclip(device=device)
        return model, preprocess_val

    return loader