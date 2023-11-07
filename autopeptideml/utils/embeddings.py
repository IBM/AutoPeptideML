import torch
import transformers
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5EncoderModel
from tqdm import tqdm

transformers.logging.set_verbosity(transformers.logging.ERROR)

AVAILABLE_MODELS = {
    'esm2_t48_15B_UR50D': 5120,
    'esm2_t36_3B_UR50D': 2560,
    'esm2_t33_650M_UR50D': 1280,
    'esm1b_t33_650M_UR50S': 1280,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t12_35M_UR50D': 480,
    'esm2_t6_8M_UR50D': 320,
    'prot_t5_xxl_uniref50': 1024,
    'prot_t5_xl_half_uniref50-enc': 1024,
    'prot_bert': 1024,
    'ProstT5': 1024
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
    'esm2-8m': 'esm2_t6_8M_UR50D'
}


class RepresentationEngine(torch.nn.Module):    
    def __init__(self, model: str, batch_size: int):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.model = None
        self._load_model(model)

    def add_head(self, head):
        self.add_module('head', head)
        self.head = head

    def compute_representations(self, sequences: list, average_pooling: bool):
        self.model.to(self.device)
        self.model.eval()
        batched_sequences = self._divide_in_batches(sequences, self.batch_size)
        output = []
        for batch in tqdm(batched_sequences):
            output.extend(self.compute_batch(batch, average_pooling))
        return output

    def compute_batch(self, batch: list, average_pooling: bool):
        inputs = self.tokenizer(batch, add_special_tokens=True, padding="longest", return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            embd_rpr = self.model(**inputs).last_hidden_state

        output = []
        for idx in range(len(batch)):
            seq_len = len(batch[idx])
            if average_pooling is True:
                output.append(embd_rpr[idx, :seq_len].mean(0).detach().cpu())
            else:
                output.append(embd_rpr[idx, :seq_len].detach().cpu())
        return output

    def dim(self) -> int:
        return self.dimension

    def forward(self, batch, labels=None):
        inputs = self.tokenizer(batch, add_special_tokens=True, padding="longest", return_tensors="pt")
        output = self.model(**inputs).last_hidden_state

        if self.head is not None:
            output = self.head(output, labels)
        return output

    def max_len(self) -> int:
        if self.lab == 'Rostlab':
            return -1
        elif self.lab == 'facebook':
            return 1022

    def print_trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable =  sum(p.numel() for p in self.parameters() if p.requires_grad)
        percentage = (trainable / total) * 100
        print(f'trainable params: {trainable} || all params: {total} || trainable: {round(percentage, 2)} %')

    def _load_model(self, model: str):
        if model not in AVAILABLE_MODELS and SYNONYMS[model.lower()] not in AVAILABLE_MODELS:
            raise NotImplementedError(f"Model: {model} not implemented. Available models: {', '.join(AVAILABLE_MODELS)}")
        if model not in AVAILABLE_MODELS:
            model = SYNONYMS[model.lower()]
        if 'pro' in model.lower():
            self.lab = 'Rostlab'
        elif 'esm' in model.lower():
            self.lab = 'facebook'  
        if 't5' in model.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(f'Rostlab/{model}', do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(f"Rostlab/{model}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(f'{self.lab}/{model}')
            self.model = AutoModel.from_pretrained(f'{self.lab}/{model}')
        self.dimension = AVAILABLE_MODELS[model]
        self.model_name = model
    
    def _divide_in_batches(self, sequences: list, batch_size: int):
        if self.model_name == 'ProstT5':
            sequences = ["<AA2fold> " + seq for seq in sequences]
        sequences = [seq[:self.max_len()] for seq in sequences]
        return [sequences[i: i+batch_size] for i in range(0, len(sequences), batch_size)]
