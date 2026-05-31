import clip

'''
We chose OpenAI CLIP as a generic, non-medical VLM baseline for comparison with the medical-domain VLMs. 
Initially, we experimented with a few OpenAI CLIP variants including ViT-L/14@336px.
See: https://github.com/openai/CLIP
'''

# Loads the OpenAI CLIP model.
def load_openai_clip(model_name='ViT-B/16', device='cuda'):
    model, preprocess = clip.load(model_name, device=device)
    model.eval().float()
    tokenizer = clip.tokenize
    return model, preprocess, tokenizer

def make_openai_clip_loader(model_name='ViT-B/16', device='cuda'):
    def loader():
        model, preprocess = clip.load(model_name, device=device)
        model.eval().float()
        return model, preprocess

    return loader