import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5EncoderModel
from typing import *

from .engine import RepEngineBase

transformers.logging.set_verbosity(transformers.logging.ERROR)


AVAILABLE_MODELS = {
    'esm2_t48_15B_UR50D': 5120,
    'esm2_t36_3B_UR50D': 2560,
    'esm2_t33_650M_UR50D': 1280,
    'esm1b_t33_650M_UR50S': 1280,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t12_35M_UR50D': 480,
    'esm2_t6_8M_UR50D': 320,
    'ESMplusplus_small': 960,
    'ESMplusplus_large': 1152,
    'prot_t5_xxl_uniref50': 1024,
    'prot_t5_xl_half_uniref50-enc': 1024,
    'prot_bert': 1024,
    'ProstT5': 1024,
    'ankh-base': 768,
    'ankh-large': 1536,
    'MoLFormer-XL-both-10pct': 768,
    'ChemBERTa-77M-MLM': 384
}

SYNONYMS = {
    'prot-t5-xl': 'prot_t5_xl_half_uniref50-enc',
    'prot-t5-xxl': 'prot_t5_xxl_uniref50',
    'protbert': 'prot_bert',
    'prost-t5': 'ProstT5',
    'esm2-15b': 'esm2_t48_15B_UR50D',
    'esm2-3b': 'esm2_t36_3B_UR50D',
    'esm2-650m': 'esm2_t33_650M_UR50D',
    'esm1b': 'esm1b_t33_650M_UR50S',
    'esm2-150m': 'esm2_t30_150M_UR50D',
    'esm2-35m': 'esm2_t12_35M_UR50D',
    'esm2-8m': 'esm2_t6_8M_UR50D',
    'esmc-300m': 'ESMplusplus_small',
    'esmc-600m': 'ESMplusplus_large',
    'ankh-base': 'ankh-base',
    'ankh-large': 'ankh-large',
    'molformer-xl': 'MoLFormer-XL-both-10pct',
    'chemberta-2': 'ChemBERTa-77M-MLM',

}


class RepEngineLM(RepEngineBase):
    def __init__(self, model: str, average_pooling: Optional[bool] = True,
                 cls_token: Optional[bool] = False):

        super().__init__(model, average_pooling=average_pooling,
                         cls_token=cls_token)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self._load_model(model)

    def move_to_device(self, device: str):
        self.device = device
        self.model.to(self.device)

    def dim(self) -> int:
        return self.dimension

    def max_len(self) -> int:
        if self.lab == 'facebook':
            return 1022
        else:
            return -1

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _load_model(self, model: str):
        if model not in AVAILABLE_MODELS and SYNONYMS[model.lower()] not in AVAILABLE_MODELS:
            raise NotImplementedError(
                f"Model: {model} not implemented.",
                f"Available models: {', '.join(AVAILABLE_MODELS)}"
            )
        if model not in AVAILABLE_MODELS:
            model = SYNONYMS[model.lower()]
        if 'pro' in model.lower():
            self.lab = 'Rostlab'
        elif 'plusplus' in model.lower():
            self.lab = 'Synthyra'
        elif 'esm' in model.lower():
            self.lab = 'facebook'
        elif 'lobster' in model.lower():
            self.lab = 'asalam91'
        elif 'ankh' in model.lower():
            self.lab = 'ElnaggarLab'
        elif 'molformer' in model.lower():
            self.lab = 'ibm'
        elif 'chemberta' in model.lower():
            self.lab = 'DeepChem'
        if 't5' in model.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(f'Rostlab/{model}',
                                                         do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(f"Rostlab/{model}")
        else:
            self.model = AutoModel.from_pretrained(f'{self.lab}/{model}',
                                                   trust_remote_code=True)
            if 'plusplus' in model.lower():
                self.tokenizer = self.model.tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    f'{self.lab}/{model}', trust_remote_code=True
                )

        self.dimension = AVAILABLE_MODELS[model]
        self.model_name = model
        self.model.to(self.device)

    def _preprocess_batch(self, sequences: List[str]) -> List[List[str]]:
        if self.lab == 'Rostlab':
            sequences = [' '.join([char for char in seq]) for seq in sequences]
        if self.model_name == 'ProstT5':
            sequences = ["<AA2fold> " + seq for seq in sequences]
        sequences = [seq[:self.max_len()] for seq in sequences]
        return sequences

    def _rep_batch(
        self, batch: List[str],
    ) -> List[np.ndarray]:
        inputs = self.tokenizer(batch, add_special_tokens=True,
                            padding="longest", return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            if self.lab == 'ElnaggarLab':
                embd_rpr = self.model(input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    decoder_input_ids=inputs['input_ids']
                                    ).last_hidden_state
            else:
                embd_rpr = self.model(**inputs).last_hidden_state
        output = []
        for idx in range(len(batch)):
            seq_len = len(batch[idx])
            if self.average_pooling:
                output.append(embd_rpr[idx, :seq_len].mean(0).detach().cpu().numpy())
            elif self.cls_token:
                output.append(embd_rpr[idx, 0].detach().cpu().numpy())
            else:
                output.append(embd_rpr[idx, :seq_len].detach().cpu().numpy())
        return output
