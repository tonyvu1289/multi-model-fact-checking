from transformers import ViTImageProcessor, ViTModel, BigBirdModel, BigBirdTokenizer
from transformers import BeitImageProcessor, BeitModel, DeiTModel, DeiTImageProcessor
import numpy as np
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import LongformerTokenizer, LongformerModel
import torch.nn.functional as F
import os
import json
import warnings
from PIL import Image


class OCRTextEncoder:
    def __init__(self, backend='ocr_easyocr', lang='en', use_gpu=None):
        self.backend = backend.lower()
        self.lang = lang
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        self.engine = None
        self.param_estimate = 0

        if self.backend in ('ocr_easyocr', 'easyocr', 'ocr_crnn_easyocr'):
            try:
                import easyocr
            except ImportError as exc:
                raise ImportError(
                    "EasyOCR backend selected but package is not installed. Install with: pip install easyocr"
                ) from exc
            self.engine = easyocr.Reader([self.lang], gpu=self.use_gpu)
            # EasyOCR detector + CRNN recognizer is significantly smaller than ViT/BEiT/DeiT.
            self.param_estimate = 80_000_000
        elif self.backend in ('ocr_paddleocr', 'paddleocr', 'ocr_mobilenet_paddle'):
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImportError(
                    "PaddleOCR backend selected but package is not installed. Install with: pip install paddleocr"
                ) from exc
            self.engine = PaddleOCR(use_angle_cls=True, lang=self.lang, use_gpu=self.use_gpu, show_log=False)
            # PaddleOCR mobile variants are generally lightweight.
            self.param_estimate = 25_000_000
        else:
            raise ValueError(
                "Unsupported OCR backend '{}'. Use one of: ocr_easyocr, ocr_paddleocr".format(backend)
            )

    def extract_text(self, image_input):
        if self.backend in ('ocr_easyocr', 'easyocr', 'ocr_crnn_easyocr'):
            result = self.engine.readtext(image_input, detail=0, paragraph=True)
            return ' '.join([str(x).strip() for x in result if str(x).strip()]).strip()

        # PaddleOCR output format: [ [ [box, (text, score)], ... ] ]
        result = self.engine.ocr(image_input, cls=True)
        if not result:
            return ''

        lines = result[0] if isinstance(result[0], list) else result
        texts = []
        for line in lines:
            if line and len(line) > 1 and isinstance(line[1], (list, tuple)) and len(line[1]) > 0:
                text = str(line[1][0]).strip()
                if text:
                    texts.append(text)
        return ' '.join(texts).strip()


class MultiModalClassification(nn.Module):
    def _resolve_ocr_backend(self, vision_pt):
        ocr_aliases = {
            'ocr': 'ocr_easyocr',
            'ocr_easyocr': 'ocr_easyocr',
            'easyocr': 'ocr_easyocr',
            'ocr_crnn_easyocr': 'ocr_easyocr',
            'ocr_paddleocr': 'ocr_paddleocr',
            'paddleocr': 'ocr_paddleocr',
            'ocr_mobilenet_paddle': 'ocr_paddleocr',
        }

        key = str(vision_pt).lower()
        if key in ('vit', 'beit', 'deit'):
            warnings.warn(
                "Vision transformer backbones are deprecated in MCVE. "
                "Falling back to OCR encoder backend: ocr_easyocr",
                UserWarning,
            )
            return 'ocr_easyocr'

        if key not in ocr_aliases:
            raise ValueError(
                "Unsupported vision_pt '{}'. Use one of: ocr_easyocr, ocr_paddleocr".format(vision_pt)
            )
        return ocr_aliases[key]

    def text_model(self, pt="roberta-base"):
        processor = AutoTokenizer.from_pretrained(pt)
        model = AutoModel.from_pretrained(pt)
        print(pt)
        return processor, model

    def text_model_long(self, pt="longformer"):
        if pt == 'longformer':
            processor = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        if pt == 'bigbird':
            model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
            processor = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

        model.requires_grad_(False)
        return processor, model

    def _count_torch_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_parameter_budget(self, max_params=400_000_000):
        total_torch_params = self._count_torch_params()
        total_estimated_params = total_torch_params + self._ocr_encoder.param_estimate
        print("Estimated total params (torch + OCR): {:,}".format(total_estimated_params))
        if total_estimated_params > max_params:
            raise ValueError(
                "Model exceeds parameter budget ({} > {}).".format(total_estimated_params, max_params)
            )

    def _extract_ocr_texts(self, image_evidence):
        image_texts = []
        for item in image_evidence:
            cache_key = item if isinstance(item, str) else None
            resolved_keys = []

            if cache_key is not None:
                resolved_keys.append(cache_key)
                resolved_keys.append(os.path.normpath(cache_key))
                resolved_keys = list(dict.fromkeys(resolved_keys))

            if cache_key is not None and any(k in self._ocr_cache for k in resolved_keys):
                found_key = next(k for k in resolved_keys if k in self._ocr_cache)
                text = self._ocr_cache[found_key]
            else:
                try:
                    if isinstance(item, str):
                        if not os.path.exists(item):
                            text = ''
                        else:
                            text = self._ocr_encoder.extract_text(item)
                    elif isinstance(item, Image.Image):
                        text = self._ocr_encoder.extract_text(np.array(item.convert('RGB')))
                    else:
                        text = self._ocr_encoder.extract_text(np.asarray(item))
                except Exception:
                    text = ''

                if cache_key is not None:
                    for key in resolved_keys:
                        self._ocr_cache[key] = text

            if str(text).strip():
                image_texts.append(str(text).strip())

        return image_texts

    def __init__(self, device, claim_pt="roberta-base", vision_pt='ocr_easyocr', long_pt="longformer",
                 ocr_cache_path=None):
        super(MultiModalClassification, self).__init__()
        print("MCVE model")
        self._claim_pt = claim_pt
        self._vision_pt = self._resolve_ocr_backend(vision_pt)
        self._long_pt = long_pt
        self.text_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self.image_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self._text_processor, self._text_model = self.text_model(self._claim_pt)
        self._long_text_processor, self._long_text_model = self.text_model_long(self._long_pt)
        self._ocr_encoder = OCRTextEncoder(self._vision_pt, use_gpu=torch.cuda.is_available())
        self._ocr_cache = {}
        self._ocr_cache_path = ocr_cache_path
        self._device = device

        if self._ocr_cache_path and os.path.exists(self._ocr_cache_path):
            try:
                with open(self._ocr_cache_path, 'r', encoding='utf-8') as f:
                    loaded_cache = json.load(f)
                # Cache format: {"/abs/path/image.jpg": "recognized text"}
                self._ocr_cache.update({str(k): str(v) for k, v in loaded_cache.items()})
                print("Loaded OCR cache entries: {}".format(len(self._ocr_cache)))
            except Exception as exc:
                print("Failed to load OCR cache '{}': {}".format(self._ocr_cache_path, exc))

        self.conv = nn.Conv1d(768, 100, stride=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768, 100, stride=1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(768, 100, stride=1, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(768 * 2, 768)

        self.pool = nn.MaxPool1d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()

        self.fc_claim = nn.Linear(768, 3)
        self.fc_evidence = nn.Linear(300, 3)

        self.dropout = nn.Dropout(0.2)

        self._validate_parameter_budget()

        # ablation test
        # self._fc_evidence_a = nn.Linear(768 * 2, 3)

    def forward(self, claim_features, label=None):
        device = self._device
        self._text_model.to(device)
        self._long_text_model.to(device)

        Hc = []
        Ht = []
        Hm = []
        Lb = []

        if label is not None:
            label = label
            for l in label:
                Lb.append(l)

        for claim_feature in claim_features:
            claim = claim_feature['claim']
            text_evidence = [x for x in claim_feature['text_evidence'] if str(x) != 'nan']
            image_evidence = claim_feature['image_evidence']

            if len(text_evidence) == 0:
                text_evidence.append("")
            if len(image_evidence) == 0:
                image_evidence.append("")

            claim_encoded = self._text_processor(claim, return_tensors="pt", padding=True, truncation=True,
                                                 max_length=100).to(device)
            claim_f = self._text_model(**claim_encoded).last_hidden_state.mean(dim=1).to(device)
            # claim_f = self._text_model(**claim_encoded).pooler_output.to(device)

            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True,
                                                     truncation=True).to(device)
            text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            # text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            image_text_evidence = self._extract_ocr_texts(image_evidence)
            if len(image_text_evidence) == 0:
                image_text_evidence = [""]
            image_encoded = self._long_text_processor(image_text_evidence, return_tensors="pt", padding=True,
                                                      truncation=True).to(device)
            image_feature = self._long_text_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)

            text_feature = torch.mean(text_feature, 0, keepdim=True)
            image_feature = torch.mean(image_feature, 0, keepdim=True)

            # text_feature = torch.sum(text_feature, 0, keepdim=True)
            # image_feature = torch.mean(image_feature, 0, keepdim=True)

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
        # claim_out = self.dropout(claim_out)
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
        # end conv module

        # remove conv
        # combine = torch.cat([fused_text, fused_img], 1).to(device)
        # claim_evidence_out = self._fc_evidence_a(combine)
        # claim_evidence_out = self.softmax(claim_evidence_out)
        # end remove conv

        # Full
        out = torch.mean(torch.stack([claim_out, claim_evidence_out], 0), 0).to(device)

        # remove claim
        # out = claim_evidence_out

        return out, Lb

### Ablation Study
# model with no conv
class MultiModalClassificationNoConv(nn.Module):
    def vision_model(self, type='vit'):
        if type == 'vit':
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        if type == 'beit':
            processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
            model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        if type == 'deit':
            processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
            model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

        model.requires_grad_(False)
        return processor, model

    def text_model(self, pt="roberta-base"):
        processor = AutoTokenizer.from_pretrained(pt)
        model = AutoModel.from_pretrained(pt)
        print(pt)
        return processor, model

    def text_model_long(self, pt="longformer"):
        if pt == 'longformer':
            processor = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        if pt == 'bigbird':
            model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
            processor = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

        model.requires_grad_(False)
        return processor, model

    def __init__(self, device, claim_pt="roberta-base", vision_pt='vit', long_pt="longformer"):
        super(MultiModalClassificationNoConv, self).__init__()
        print("MCVE with no conv")
        self._claim_pt = claim_pt
        self._vision_pt = vision_pt
        self._long_pt = long_pt
        self.text_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self.image_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self._text_processor, self._text_model = self.text_model(self._claim_pt)
        self._long_text_processor, self._long_text_model = self.text_model_long(self._long_pt)
        self._image_processor, self._vision_model = self.vision_model(self._vision_pt)
        self._device = device

        self.conv = nn.Conv1d(768, 100, stride=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768, 100, stride=1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(768, 100, stride=1, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(768 * 2, 768)

        self.pool = nn.MaxPool1d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()

        self.fc_claim = nn.Linear(768, 3)
        self.fc_evidence = nn.Linear(300, 3)

        self.dropout = nn.Dropout(0.2)

        # ablation test
        self._fc_evidence_a = nn.Linear(768 * 2, 3)

    def forward(self, claim_features, label=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._vision_model.to(device)
        self._text_model.to(device)
        self._long_text_model.to(device)

        Hc = []
        Ht = []
        Hm = []
        Lb = []

        if label is not None:
            label = label
            for l in label:
                Lb.append(l)

        for claim_feature in claim_features:
            claim = claim_feature['claim']
            text_evidence = [x for x in claim_feature['text_evidence'] if str(x) != 'nan']
            image_evidence = [Image.open(simg) for simg in claim_feature['image_evidence']]

            if len(text_evidence) == 0:
                text_evidence.append("")
            if len(image_evidence) == 0:
                blank_img = np.zeros((50, 50, 3), np.uint8)
                blank_img.fill(255)
                image_evidence.append(blank_img)

            claim_encoded = self._text_processor(claim, return_tensors="pt", padding=True, truncation=True,
                                                 max_length=100).to(device)
            # claim_f = self._text_model(**claim_encoded).last_hidden_state.mean(dim=1).to(device)
            claim_f = self._text_model(**claim_encoded).pooler_output.to(device)

            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True,
                                                     truncation=True).to(device)
            # text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            image_encoded = self._image_processor(image_evidence, return_tensors="pt").to(device)
            image_feature = self._vision_model(**image_encoded).pooler_output.to(device)

            text_feature = torch.mean(text_feature, 0, keepdim=True)
            image_feature = torch.mean(image_feature, 0, keepdim=True)

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

        combine = torch.cat([fused_text, fused_img], 1).to(device)
        claim_evidence_out = self._fc_evidence_a(combine)
        claim_evidence_out = self.softmax(claim_evidence_out)

        out = torch.mean(torch.stack([claim_out, claim_evidence_out], 0), 0).to(device)

        return out, Lb


class MultiModalClassificationNoClaim(nn.Module):
    def vision_model(self, type='vit'):
        if type == 'vit':
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        if type == 'beit':
            processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
            model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        if type == 'deit':
            processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
            model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

        model.requires_grad_(False)
        return processor, model

    def text_model(self, pt="roberta-base"):
        processor = AutoTokenizer.from_pretrained(pt)
        model = AutoModel.from_pretrained(pt)
        print(pt)
        return processor, model

    def text_model_long(self, pt="longformer"):
        if pt == 'longformer':
            processor = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        if pt == 'bigbird':
            model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
            processor = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

        model.requires_grad_(False)
        return processor, model

    def __init__(self, device, claim_pt="roberta-base", vision_pt='vit', long_pt="longformer"):
        super(MultiModalClassificationNoClaim, self).__init__()
        print("MCVE with no claim")
        self._claim_pt = claim_pt
        self._vision_pt = vision_pt
        self._long_pt = long_pt
        self.text_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self.image_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self._text_processor, self._text_model = self.text_model(self._claim_pt)
        self._long_text_processor, self._long_text_model = self.text_model_long(self._long_pt)
        self._image_processor, self._vision_model = self.vision_model(self._vision_pt)
        self._device = device

        self.conv = nn.Conv1d(768, 100, stride=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768, 100, stride=1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(768, 100, stride=1, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(768 * 2, 768)

        self.pool = nn.MaxPool1d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()

        self.fc_claim = nn.Linear(768, 3)
        self.fc_evidence = nn.Linear(300, 3)

        self.dropout = nn.Dropout(0.2)

        # ablation test
        self._fc_evidence_a = nn.Linear(768 * 2, 3)

    def forward(self, claim_features, label=None):
        device = self._device
        self._vision_model.to(device)
        self._text_model.to(device)
        self._long_text_model.to(device)

        Hc = []
        Ht = []
        Hm = []
        Lb = []

        if label is not None:
            label = label
            for l in label:
                Lb.append(l)

        for claim_feature in claim_features:
            claim = claim_feature['claim']
            text_evidence = [x for x in claim_feature['text_evidence'] if str(x) != 'nan']
            image_evidence = [Image.open(simg) for simg in claim_feature['image_evidence']]

            if len(text_evidence) == 0:
                text_evidence.append("")
            if len(image_evidence) == 0:
                blank_img = np.zeros((50, 50, 3), np.uint8)
                blank_img.fill(255)
                image_evidence.append(blank_img)

            claim_encoded = self._text_processor(claim, return_tensors="pt", padding=True, truncation=True,
                                                 max_length=100).to(device)
            # claim_f = self._text_model(**claim_encoded).last_hidden_state.mean(dim=1).to(device)
            claim_f = self._text_model(**claim_encoded).pooler_output.to(device)

            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True,
                                                     truncation=True).to(device)
            # text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            image_encoded = self._image_processor(image_evidence, return_tensors="pt").to(device)
            # image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
            image_feature = self._vision_model(**image_encoded).pooler_output.to(device)

            text_feature = torch.mean(text_feature, 0, keepdim=True)
            image_feature = torch.mean(image_feature, 0, keepdim=True)

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
        # end conv module

        # remove claim
        out = claim_evidence_out

        return out, Lb


class MultiModalClassificationNoClaimNoCov(nn.Module):
    def vision_model(self, type='vit'):
        if type == 'vit':
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        if type == 'beit':
            processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
            model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        if type == 'deit':
            processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
            model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

        model.requires_grad_(False)
        return processor, model

    def text_model(self, pt="roberta-base"):
        processor = AutoTokenizer.from_pretrained(pt)
        model = AutoModel.from_pretrained(pt)
        print(pt)
        return processor, model

    def text_model_long(self, pt="longformer"):
        if pt == 'longformer':
            processor = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        if pt == 'bigbird':
            model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
            processor = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

        model.requires_grad_(False)
        return processor, model

    def __init__(self, device, claim_pt="roberta-base", vision_pt='vit', long_pt="longformer"):
        super(MultiModalClassificationNoClaimNoCov, self).__init__()
        print("MCVE with no claim and no conv")
        self._claim_pt = claim_pt
        self._vision_pt = vision_pt
        self._long_pt = long_pt
        self.text_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self.image_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self._text_processor, self._text_model = self.text_model(self._claim_pt)
        self._long_text_processor, self._long_text_model = self.text_model_long(self._long_pt)
        self._image_processor, self._vision_model = self.vision_model(self._vision_pt)
        self._device = device

        self.conv = nn.Conv1d(768, 100, stride=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768, 100, stride=1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(768, 100, stride=1, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(768 * 2, 768)

        self.pool = nn.MaxPool1d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()

        self.fc_claim = nn.Linear(768, 3)
        self.fc_evidence = nn.Linear(300, 3)

        self.dropout = nn.Dropout(0.2)

        # ablation test
        self._fc_evidence_a = nn.Linear(768 * 2, 3)

    def forward(self, claim_features, label=None):
        device = self._device
        self._vision_model.to(device)
        self._text_model.to(device)
        self._long_text_model.to(device)

        Hc = []
        Ht = []
        Hm = []
        Lb = []

        if label is not None:
            label = label
            for l in label:
                Lb.append(l)

        for claim_feature in claim_features:
            claim = claim_feature['claim']
            text_evidence = [x for x in claim_feature['text_evidence'] if str(x) != 'nan']
            image_evidence = [Image.open(simg) for simg in claim_feature['image_evidence']]

            if len(text_evidence) == 0:
                text_evidence.append("")
            if len(image_evidence) == 0:
                blank_img = np.zeros((50, 50, 3), np.uint8)
                blank_img.fill(255)
                image_evidence.append(blank_img)

            claim_encoded = self._text_processor(claim, return_tensors="pt", padding=True, truncation=True,
                                                 max_length=100).to(device)
            # claim_f = self._text_model(**claim_encoded).last_hidden_state.mean(dim=1).to(device)
            claim_f = self._text_model(**claim_encoded).pooler_output.to(device)

            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True,
                                                     truncation=True).to(device)
            # text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            image_encoded = self._image_processor(image_evidence, return_tensors="pt").to(device)
            # image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
            image_feature = self._vision_model(**image_encoded).pooler_output.to(device)

            text_feature = torch.mean(text_feature, 0, keepdim=True)
            image_feature = torch.mean(image_feature, 0, keepdim=True)

            # text_feature = torch.sum(text_feature, 0, keepdim=True)
            # image_feature = torch.mean(image_feature, 0, keepdim=True)

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

        # remove conv
        combine = torch.cat([fused_text, fused_img], 1).to(device)
        claim_evidence_out = self._fc_evidence_a(combine)
        claim_evidence_out = self.softmax(claim_evidence_out)
        # end remove conv

        out = claim_evidence_out

        return out, Lb

###  End Ablation Study

## Ablation Study 
# No attention
class MultiModalClassificationNoAttention(nn.Module):
    def vision_model(self, type='vit'):
        if type == 'vit':
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        if type == 'beit':
            processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
            model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        if type == 'deit':
            processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
            model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

        model.requires_grad_(False)
        return processor, model

    def text_model(self, pt="roberta-base"):
        processor = AutoTokenizer.from_pretrained(pt)
        model = AutoModel.from_pretrained(pt)
        print(pt)
        return processor, model

    def text_model_long(self, pt="longformer"):
        if pt == 'longformer':
            processor = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        if pt == 'bigbird':
            model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
            processor = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

        model.requires_grad_(False)
        return processor, model

    def __init__(self, device, claim_pt="roberta-base", vision_pt='vit', long_pt="longformer"):
        super(MultiModalClassificationNoAttention, self).__init__()
        print("MCVE with no attention")
        self._claim_pt = claim_pt
        self._vision_pt = vision_pt
        self._long_pt = long_pt
        # self.text_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        # self.image_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self._text_processor, self._text_model = self.text_model(self._claim_pt)
        self._long_text_processor, self._long_text_model = self.text_model_long(self._long_pt)
        self._image_processor, self._vision_model = self.vision_model(self._vision_pt)
        self._device = device

        self.conv = nn.Conv1d(768, 100, stride=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768, 100, stride=1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(768, 100, stride=1, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(768 * 2, 768)

        self.pool = nn.MaxPool1d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()

        self.fc_claim = nn.Linear(768, 3)
        self.fc_evidence = nn.Linear(300, 3)

        self.dropout = nn.Dropout(0.2)

        # ablation test
        # self._fc_evidence_a = nn.Linear(768 * 2, 3)

    def forward(self, claim_features, label=None):
        device = self._device
        self._vision_model.to(device)
        self._text_model.to(device)
        self._long_text_model.to(device)

        Hc = []
        Ht = []
        Hm = []
        Lb = []

        if label is not None:
            label = label
            for l in label:
                Lb.append(l)

        for claim_feature in claim_features:
            claim = claim_feature['claim']
            text_evidence = [x for x in claim_feature['text_evidence'] if str(x) != 'nan']
            image_evidence = [Image.open(simg) for simg in claim_feature['image_evidence']]

            if len(text_evidence) == 0:
                text_evidence.append("")
            if len(image_evidence) == 0:
                blank_img = np.zeros((50, 50, 3), np.uint8)
                blank_img.fill(255)
                image_evidence.append(blank_img)

            claim_encoded = self._text_processor(claim, return_tensors="pt", padding=True, truncation=True,
                                                 max_length=100).to(device)
            claim_f = self._text_model(**claim_encoded).last_hidden_state.mean(dim=1).to(device)
            # claim_f = self._text_model(**claim_encoded).pooler_output.to(device)

            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True,
                                                     truncation=True).to(device)
            text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            # text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            image_encoded = self._image_processor(image_evidence, return_tensors="pt").to(device)
            image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
            # image_feature = self._vision_model(**image_encoded).pooler_output.to(device)

            text_feature = torch.mean(text_feature, 0, keepdim=True)
            image_feature = torch.mean(image_feature, 0, keepdim=True)

            # text_feature = torch.sum(text_feature, 0, keepdim=True)
            # image_feature = torch.mean(image_feature, 0, keepdim=True)

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

        # attention_claim_text, _ = self.text_attention(Hc, text_evidence_features, text_evidence_features)
        # attention_claim_img, _ = self.image_attention(Hc, image_evidence_features, image_evidence_features)

        fused_text = self.leaky_relu(self.fc1(torch.cat([text_evidence_features * Hc, text_evidence_features - Hc], 1)))
        fused_img = self.leaky_relu(self.fc1(torch.cat([image_evidence_features * Hc, image_evidence_features - Hc], 1)))

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