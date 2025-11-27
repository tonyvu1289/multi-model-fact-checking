import argparse
import pandas as pd
import torch
import datetime

from sklearn.metrics import f1_score

from read_data import get_dataset
from train import ClaimVerificationDataset, train_model, predict, train_resume
import helper


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--val', default=True, action='store_true')
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--claim_pt', type=str, default="roberta-base")
    parser.add_argument('--vision_pt', type=str, default="vit")
    parser.add_argument('--long_pt', type=str, default="longformer")
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--n_gpu', type=int, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    train, val, test = get_dataset(args.path)
    train_claim = ClaimVerificationDataset(train)
    dev_claim = ClaimVerificationDataset(val)
    test_claim = ClaimVerificationDataset(test)

    if args.n_gpu:
        device = torch.device('cuda:{}'.format(args.n_gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.test:
        model = torch.load(args.model_path, map_location=device)
    else:
        if args.checkpoint_path is not None:
            print("Resuming training from checkpoint: {}".format(args.checkpoint_path))
            claim_pt, vision_pt, long_pt, chkpoint = helper.process_checkpoint_path(args.checkpoint_path)
            model, loss, name_pt = train_resume(
                train_claim,
                chkpoint,
                is_val=args.val,
                val_data=dev_claim,
                device=device,
                claim_pt=claim_pt,
                vision_pt=vision_pt,
                long_pt=long_pt,
            )
        else:
            model, loss, name_pt = train_model(train_claim, batch_size=args.batch_size,
                                         epoch=args.epoch, is_val=args.val, val_data=dev_claim, device=device,
                                         claim_pt=args.claim_pt, vision_pt=args.vision_pt, long_pt=args.long_pt)

        # torch.save(model, 'claim_verification_{}.pt'.format(
        #     str(datetime.datetime.now().strftime("%d-%m_%H-%M"))))

    gt, prd, ids = predict(test_claim, model, args.batch_size, device=device)
    print("Test result micro: {}\n".format(f1_score(gt, prd, average='micro')))
    print("Test result macro: {}\n".format(f1_score(gt, prd, average='macro')))

    output_df = pd.DataFrame({'claim_id': ids, 'predict': prd, 'ground_truth': gt})
    output_df.to_csv('predict_test.csv')

    gtd, prdd, idsd = predict(dev_claim, model, args.batch_size, device=device)
    print("Dev result micro: {}\n".format(f1_score(gtd, prdd, average='micro')))
    print("Dev result macro: {}\n".format(f1_score(gtd, prdd, average='macro')))

    output_dfd = pd.DataFrame({'claim_id': idsd, 'predict': prdd, 'ground_truth': gtd})
    output_dfd.to_csv('predict_dev.csv')
