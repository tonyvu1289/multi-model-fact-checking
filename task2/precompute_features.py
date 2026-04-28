import os
import argparse
from tqdm import tqdm, trange
import random
import torch
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor, ViTModel, BeitImageProcessor, BeitModel, DeiTImageProcessor, DeiTModel
from PIL import Image

from read_data import get_dataset
from train import ClaimVerificationDataset, MultiModalClassification, make_batch

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
    model = model.to(device)
    model.eval()
    hidden_size = model.config.hidden_size
    embeddings = [torch.zeros(hidden_size) for _ in texts]

    valid_indices = [i for i, t in enumerate(texts) if str(t) != 'nan']
    if not valid_indices:
        return embeddings

    valid_texts = [texts[i] for i in valid_indices]
    with torch.inference_mode():
        inputs = tokenizer(valid_texts, truncation=True, padding=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)

        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            batch_embs = out.pooler_output.detach().cpu()
        else:
            batch_embs = mean_pool(out.last_hidden_state, inputs.get('attention_mask')).detach().cpu()

    for j, idx in enumerate(valid_indices):
        embeddings[idx] = batch_embs[j]
    return embeddings


def encode_images(processor, model, image_paths, device):
    model = model.to(device)
    model.eval()
    hidden_size = model.config.hidden_size
    embeddings = [torch.zeros(hidden_size) for _ in image_paths]

    valid_indices = []
    valid_images = []
    for i, p in enumerate(image_paths):
        try:
            valid_images.append(Image.open(p).convert('RGB'))
            valid_indices.append(i)
        except Exception:
            continue

    if not valid_indices:
        return embeddings

    with torch.inference_mode():
        inputs = processor(images=valid_images, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)

        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            batch_embs = out.pooler_output.detach().cpu()
        else:
            batch_embs = out.last_hidden_state.mean(dim=1).detach().cpu()

    for j, idx in enumerate(valid_indices):
        embeddings[idx] = batch_embs[j]

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

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val', default=True, action='store_true')
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--claim_pt', type=str, default="roberta-base")
    parser.add_argument('--vision_pt', type=str, default="vit")
    parser.add_argument('--long_pt', type=str, default="longformer")
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--n_gpu', type=int, default=None)
    parser.add_argument('--sample', default=False,action='store_true')
    parser.add_argument('--precompute_out_dir', type=str, default="./precomputed_features")
    args = parser.parse_args()
    return args
def compute_feature(train_data,output_dir, batch_size, claim_pt="roberta-base", vision_pt='vit',
                long_pt="longformer", device=None):
    model = MultiModalClassification(device, claim_pt, vision_pt, long_pt)
    model = model.to(device)
    os.makedirs(output_dir, exist_ok=True)

    X, y, _ = make_batch(train_data, batch_size=batch_size)
    # model.train()
    model.eval()
    for i in trange(len(X)):
        batch_x = X[i]
        with torch.inference_mode():
            Hc, Ht, Hm = model.compute_Hc_Ht_Hm(batch_x)

        Hc_cpu = Hc.detach().cpu()
        Ht_cpu = Ht.detach().cpu()
        Hm_cpu = Hm.detach().cpu()

        # Save only the row that belongs to each claim id.
        for j, x in enumerate(batch_x):
            claim_id = x['claim_id']

            hc_item = Hc_cpu[j] if Hc_cpu.dim() > 1 else Hc_cpu
            ht_item = Ht_cpu[j] if Ht_cpu.dim() > 1 else Ht_cpu
            hm_item = Hm_cpu[j] if Hm_cpu.dim() > 1 else Hm_cpu

            torch.save({
                'Hc': hc_item,
                'Ht': ht_item,
                'Hm': hm_item,
            }, os.path.join(output_dir, f'{claim_id}.pt'))
    # return best_model, loss_vals, claim_pt


def main():
    args = parser_args()
    output_dir = args.precompute_out_dir
    os.makedirs(output_dir, exist_ok=True)

    # create precomputed paths
    train_precomputed_path = os.path.join(output_dir, 'train')
    val_precomputed_path = os.path.join(output_dir, 'val')
    test_precomputed_path = os.path.join(output_dir, 'test')

    os.makedirs(train_precomputed_path, exist_ok=True)
    os.makedirs(val_precomputed_path, exist_ok=True)
    os.makedirs(test_precomputed_path, exist_ok=True)

    train, val, test = get_dataset(args.path)
    
    # for debugging, run on small subset
    if args.sample:
        train = random.sample(train, 100)
        val = random.sample(val, 100)
        test = random.sample(test, 100)

    # create datasets
    train_claim = ClaimVerificationDataset(train)
    dev_claim = ClaimVerificationDataset(val)
    test_claim = ClaimVerificationDataset(test)

    #setting stuff 
    if args.n_gpu:
        device = torch.device('cuda:{}'.format(args.n_gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # compute feature for each split
    
    print("Computing features for train split...")
    compute_feature(train_claim,train_precomputed_path, batch_size=args.batch_size,
                                        device=device,
                                        claim_pt=args.claim_pt, vision_pt=args.vision_pt, long_pt=args.long_pt)

    print("Computing features for validation split...")
    compute_feature(dev_claim,val_precomputed_path, batch_size=args.batch_size,
                                        device=device,
                                        claim_pt=args.claim_pt, vision_pt=args.vision_pt, long_pt=args.long_pt)

    print("Computing features for test split...")
    compute_feature(test_claim,test_precomputed_path, batch_size=args.batch_size,
                                        device=device,
                                        claim_pt=args.claim_pt, vision_pt=args.vision_pt, long_pt=args.long_pt)
if __name__ == '__main__':
    main()
