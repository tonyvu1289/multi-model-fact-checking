import torch
import os


def process_checkpoint_path(path):
    # path format: model_verification_{claim_pt}_{long_pt}_{vision_pt}_{timestamp}
    base_name = os.path.basename(path)
    parts = base_name.split('_')
    claim_pt = parts[2]
    long_pt = parts[3]
    vision_pt = parts[4]

    # in {path}/checkpoint/checkpoint_*.pt, find the latest checkpoint
    chkpoint_dir = os.path.join(path, 'checkpoint')
    chkpoints = [os.path.join(chkpoint_dir, f) for f in os.listdir(chkpoint_dir) if f.endswith('.pt')]
    def get_checkpoint_num(ckpt_path):
        base = os.path.basename(ckpt_path)
        num_str = base.split('_')[-1].replace('.pt', '')
        return int(num_str)
    chkpoint = max(chkpoints, key=get_checkpoint_num)
    chkpoint = torch.load(chkpoint, map_location='cpu')
    return claim_pt, vision_pt, long_pt, chkpoint
    
if __name__ == '__main__':
    path = '/home/duy/project/thesis_master/multimodal-fact-checking/model_dump/model_verification_roberta-base_longformer_ocr_easyocr_30-03_12-00'
    claim_pt, vision_pt, long_pt, chkpoint = process_checkpoint_path(path)
    print(claim_pt, vision_pt, long_pt)
    print(chkpoint.keys())
    print(type(chkpoint['model_state_dict']))
