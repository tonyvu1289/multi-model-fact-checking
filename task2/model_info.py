import argparse

import torch

from model import MultiModalClassification
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    train_param = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        train_param += params
    print(table)

    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params

    print(f"Total Trainable Params: {train_param}")
    print(f"Total Params: {total_params}")


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--claim_pt', type=str, default="roberta-base")
    parser.add_argument(
        '--vision_pt',
        type=str,
        default="ocr_easyocr",
        choices=["ocr_easyocr", "ocr_paddleocr", "easyocr", "paddleocr", "ocr", "ocr_mobilenet_paddle"],
    )
    parser.add_argument('--long_pt', type=str, default="longformer")
    parser.add_argument('--n_gpu', type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    if args.n_gpu:
        device = torch.device('cuda:{}'.format(args.n_gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalClassification(device, args.claim_pt, args.vision_pt, args.long_pt)
    count_parameters(model)
