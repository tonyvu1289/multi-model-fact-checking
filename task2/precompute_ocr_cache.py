import argparse
import json
import os

from tqdm import tqdm

from read_data import get_dataset
from model import OCRTextEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--vision_pt', type=str, default='ocr_easyocr')
    parser.add_argument('--output', type=str, default='task2/ocr_cache.json')
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val', 'test'])
    parser.add_argument('--sample_limit', type=int, default=None)
    return parser.parse_args()


def select_split(args, train, val, test):
    if args.split == 'train':
        data = train
    elif args.split == 'val':
        data = val
    elif args.split == 'test':
        data = test
    else:
        data = train + val + test

    if args.sample_limit is not None:
        data = data[: max(1, int(args.sample_limit))]
    return data


def collect_image_paths(samples):
    image_paths = set()
    for sample in samples:
        evidences = sample[2]
        for path in evidences.tolist():
            if isinstance(path, str) and path:
                image_paths.add(path)
    return sorted(image_paths)


def main():
    args = parse_args()

    train, val, test = get_dataset(args.path)
    samples = select_split(args, train, val, test)
    image_paths = collect_image_paths(samples)

    print('Total unique image evidences:', len(image_paths))
    ocr_encoder = OCRTextEncoder(args.vision_pt)

    cache = {}
    for image_path in tqdm(image_paths):
        text = ''
        if os.path.exists(image_path):
            try:
                text = ocr_encoder.extract_text(image_path)
            except Exception:
                text = ''

        cache[image_path] = text
        cache[os.path.normpath(image_path)] = text

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

    print('OCR cache written:', args.output)
    print('Cache entries:', len(cache))


if __name__ == '__main__':
    main()
