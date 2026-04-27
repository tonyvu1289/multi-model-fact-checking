import torch
import torch.nn.functional as F
import torch.nn as nn
from model import MultiModalClassification
import copy


class PrecomputedMultiModalClassification(nn.Module):
    """Wrapper model that reuses the attention/conv/classifier parts of MultiModalClassification
    but accepts precomputed embeddings in each sample under keys:
      - 'claim_embedding' (tensor or array)
      - 'text_evidence_embeddings' (list of tensors/arrays)
      - 'image_evidence_embeddings' (list of tensors/arrays)

    This keeps original encoders untouched and trains only the fusion/classifier parts.
    """

    def __init__(self, device, claim_pt="roberta-base", vision_pt='vit', long_pt="longformer"):
        super(PrecomputedMultiModalClassification, self).__init__()
        # instantiate base model to get layer definitions
        self.base = MultiModalClassification(device, claim_pt, vision_pt, long_pt)
        # we'll use the base's attention, convs, fc layers
        self.text_attention = self.base.text_attention
        self.image_attention = self.base.image_attention
        self.fc1 = self.base.fc1
        self.pool = self.base.pool
        self.softmax = self.base.softmax
        self.leaky_relu = self.base.leaky_relu
        self.fc_claim = self.base.fc_claim
        self.fc_evidence = self.base.fc_evidence
        self.dropout = self.base.dropout
        self.conv = self.base.conv
        self.conv2 = self.base.conv2
        self.conv3 = self.base.conv3
        self._device = device

    def forward(self, claim_features, label=None):
        device = self._device
        Hc = []
        Ht = []
        Hm = []
        Lb = []

        if label is not None:
            for l in label:
                Lb.append(l)

        for claim_feature in claim_features:
            ce = claim_feature['claim_embedding']
            if isinstance(ce, torch.Tensor):
                claim_f = ce.to(device)
                if claim_f.dim() == 1:
                    claim_f = claim_f.unsqueeze(0)
            else:
                claim_f = torch.tensor(ce).to(device).unsqueeze(0)

            te_list = claim_feature.get('text_evidence_embeddings', [])
            if te_list:
                te_tensors = [t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t).to(device) for t in te_list]
                text_feature = torch.stack(te_tensors).mean(dim=0, keepdim=True)
            else:
                text_feature = torch.zeros_like(claim_f)

            ie_list = claim_feature.get('image_evidence_embeddings', [])
            if ie_list:
                ie_tensors = [t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t).to(device) for t in ie_list]
                image_feature = torch.stack(ie_tensors).mean(dim=0, keepdim=True)
            else:
                image_feature = torch.zeros_like(claim_f)

            Hc.append(claim_f)
            Ht.append(text_feature)
            Hm.append(image_feature)

        Hc = torch.cat(Hc)
        Ht = torch.cat(Ht)
        Hm = torch.cat(Hm)

        if Lb:
            Lb = torch.stack(Lb)

        text_evidence_features = Ht
        image_evidence_features = Hm

        attention_claim_text, _ = self.text_attention(Hc, text_evidence_features, text_evidence_features)
        attention_claim_img, _ = self.image_attention(Hc, image_evidence_features, image_evidence_features)

        fused_text = self.leaky_relu(self.fc1(torch.cat([attention_claim_text * Hc, attention_claim_text - Hc], 1)))
        fused_img = self.leaky_relu(self.fc1(torch.cat([attention_claim_img * Hc, attention_claim_img - Hc], 1)))

        claim_out = self.fc_claim(Hc)
        claim_out = self.softmax(claim_out)

        # Conv modules
        c1_t = F.relu(self.conv(fused_text.T).T)
        c2_t = F.relu(self.conv2(fused_text.T).T)
        c3_t = F.relu(self.conv3(fused_text.T).T)
        conv_t = torch.cat([c1_t, c2_t, c3_t], 1).to(device)

        c1_i = F.relu(self.conv(fused_img.T).T)
        c2_i = F.relu(self.conv2(fused_img.T).T)
        c3_i = F.relu(self.conv2(fused_img.T).T)
        conv_i = torch.cat([c1_i, c2_i, c3_i], 1).to(device)

        combine = torch.cat([conv_t, conv_i], 1).to(device)
        combine = self.pool(combine)
        claim_evidence_out = self.fc_evidence(combine)
        claim_evidence_out = self.dropout(claim_evidence_out)
        claim_evidence_out = self.softmax(claim_evidence_out)

        out = torch.mean(torch.stack([claim_out, claim_evidence_out], 0), 0).to(device)

        return out, Lb
