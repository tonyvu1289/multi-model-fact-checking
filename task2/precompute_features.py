import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor, ViTModel, BeitImageProcessor, BeitModel, DeiTImageProcessor, DeiTModel
from PIL import Image

from read_data import get_dataset


def get_vision_processor_and_model(vision_pt):
    if vision_pt == 'vit':
        return ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k'), ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    if vision_pt == 'beit':
        return BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k'), BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    if vision_pt == 'deit':
        return DeiTImageProcessor.from_pretrained('facebook/deit-base-distilled-patch16-224'), DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
    # fallback
    return ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k'), ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')


def mean_pool(last_hidden_state, attention_mask=None):
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1)


def encode_texts(tokenizer, model, texts, device):
    embeddings = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for t in texts:
            if str(t) == 'nan':
                embeddings.append(torch.zeros(model.config.hidden_size))
                continue
            inputs = tokenizer(t, truncation=True, padding=True, return_tensors='pt')
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            out = model(**inputs)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                emb = out.pooler_output.squeeze(0).cpu()
            else:
                emb = mean_pool(out.last_hidden_state, inputs.get('attention_mask')).squeeze(0).cpu()
            embeddings.append(emb)
    return embeddings


def encode_images(processor, model, image_paths, device):
    model = model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for p in image_paths:
            try:
                img = Image.open(p).convert('RGB')
            except Exception:
                embeddings.append(torch.zeros(model.config.hidden_size))
                continue
            inputs = processor(images=img, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                emb = out.pooler_output.squeeze(0).cpu()
            else:
                emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            embeddings.append(emb)
    return embeddings


def build_label_tensor(label):
    label2idx = {
        'refuted': 2,
        'NEI': 1,
        'supported': 0
    }
    idx = label2idx[label]
    v = torch.zeros(3, dtype=torch.float)
    v[idx] = 1.0
    return v


def process_split(split_data, tokenizer, text_model, vision_processor, vision_model, device, out_path):
    encoded = []
    for sample in tqdm(split_data, desc=f'Encoding {out_path}'):
        claim, text_evidence, image_evidence, label, claim_id = sample
        claim_emb = encode_texts(tokenizer, text_model, [claim], device)[0]
        text_embs = encode_texts(tokenizer, text_model, list(text_evidence), device)
        image_embs = encode_images(vision_processor, vision_model, list(image_evidence), device)

        item = {
            'claim_id': claim_id,
            'claim_embedding': claim_emb,
            'text_evidence_embeddings': text_embs,
            'image_evidence_embeddings': image_embs,
            'label': build_label_tensor(label)
        }
        encoded.append(item)

    torch.save(encoded, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/kaggle/input/mocheg1/mocheg')
    parser.add_argument('--claim_pt', type=str, default='roberta-base')
    parser.add_argument('--vision_pt', type=str, default='vit')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_dir', type=str, default='precomputed')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading dataset...')
    train, val, test = get_dataset(args.data_path)

    print('Loading text encoder...')
    tokenizer = AutoTokenizer.from_pretrained(args.claim_pt)
    text_model = AutoModel.from_pretrained(args.claim_pt)

    print('Loading vision encoder...')
    vision_processor, vision_model = get_vision_processor_and_model(args.vision_pt)

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    process_split(train, tokenizer, text_model, vision_processor, vision_model, device, os.path.join(args.out_dir, 'train_precomputed.pt'))
    process_split(val, tokenizer, text_model, vision_processor, vision_model, device, os.path.join(args.out_dir, 'val_precomputed.pt'))
    process_split(test, tokenizer, text_model, vision_processor, vision_model, device, os.path.join(args.out_dir, 'test_precomputed.pt'))

    print('Done. Precomputed features saved to', args.out_dir)


if __name__ == '__main__':
    main()
