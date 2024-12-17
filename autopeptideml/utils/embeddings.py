import torch
import transformers
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5EncoderModel
from tqdm import tqdm
from typing import List, Optional

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
    'ankh-large': 1536
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
    'ankh-large': 'ankh-large'
}


class RepresentationEngine(torch.nn.Module):
    """
    A class for generating sequence representations using pre-trained models, with flexible pooling and device management.

    Attributes:
    ----------
    - device : str
        The device ('cuda', 'mps', or 'cpu') on which the model will be run.
    - batch_size : int
        The size of data batches for processing.
    - model : torch.nn.Module
        The loaded pre-trained model for generating representations.
    - tokenizer : AutoTokenizer or T5Tokenizer
        The tokenizer associated with the pre-trained model.
    - lab : str
        Identifier for the lab or group associated with the model, e.g., 'Rostlab', 'facebook', or 'ElnaggarLab'.
    - dimension : int
        Dimension of the output representations.
    - model_name : str
        Name of the loaded pre-trained model.
    - head : Optional[torch.nn.Module]
        Optional head layer added to the model for specific tasks.

    Parameters:
    ----------
    - model : str
        Name of the pre-trained model to load.
    - batch_size : int
        Batch size for sequence processing.

    Methods:
    -------
    - move_to_device(device: str)
        Sets the device for computation, either 'cpu', 'mps', or 'cuda'.

    - add_head(head: torch.nn.Module)
        Adds an optional head module to the model, which can be used for task-specific outputs.

    - compute_representations(sequences: List[str], average_pooling: bool, cls_token: Optional[bool] = False) -> List[torch.Tensor]
        Generates representations for a list of input sequences. Supports average pooling or CLS token extraction.

    - compute_batch(batch: List[str], average_pooling: bool, cls_token: Optional[bool] = False) -> List[torch.Tensor]
        Processes a batch of sequences and extracts representations according to specified pooling methods.

    - dim() -> int
        Returns the dimension of the representation layer of the model.

    - forward(batch, labels=None)
        Performs a forward pass through the model, with an optional head layer if added.

    - max_len() -> int
        Returns the maximum sequence length allowed for the model, based on lab specifications.

    - print_trainable_parameters()
        Prints the total and trainable parameter counts, as well as the percentage of trainable parameters.

    - _load_model(model: str)
        Internal method to load a pre-trained model and tokenizer based on the specified model name.

    - _divide_in_batches(sequences: List[str], batch_size: int) -> List[List[str]]
        Divides a list of sequences into batches and processes each sequence for compatibility with the model.

    ### Method Details

    - **move_to_device(device: str)**:
        Sets the device (`cpu` or `cuda`) on which the model will run. Updates the `self.device` attribute.

    - **add_head(head: torch.nn.Module)**:
        Adds a task-specific head module to the model, stored in `self.head`. This head will be applied in the `forward()` method for additional processing.

    - **compute_representations(sequences: List[str], average_pooling: bool, cls_token: Optional[bool] = False) -> List[torch.Tensor]**:
        Generates vector representations for a list of input sequences. Either `average_pooling` or `cls_token` can be used to extract representations, but not both simultaneously. Divides the sequences into batches and processes each batch.

    - **compute_batch(batch: List[str], average_pooling: bool, cls_token: Optional[bool] = False) -> List[torch.Tensor]**:
        Processes a single batch of sequences, creating representations using either average pooling or CLS token extraction. Returns a list of tensor representations for each sequence in the batch.

    - **dim() -> int**:
        Returns the dimension of the representation layer based on the loaded model.

    - **forward(batch, labels=None)**:
        Executes a forward pass through the model with the given batch, applying the optional head if present. Returns the output representation tensor.

    - **max_len() -> int**:
        Returns the maximum sequence length allowed by the model based on the `lab` attribute.

    - **print_trainable_parameters()**:
        Calculates and prints the total number of model parameters, the number of trainable parameters, and the percentage of trainable parameters.

    - **_load_model(model: str)**:
        Internal method that loads the specified pre-trained model and tokenizer from either Rostlab, Facebook, or ElnaggarLab. Sets model attributes such as `dimension`, `tokenizer`, and `model`.

    - **_divide_in_batches(sequences: List[str], batch_size: int) -> List[List[str]]**:
        Splits a list of sequences into smaller batches according to the batch size. Adjusts sequences to meet model-specific formatting requirements.

    This class is intended for users who need a flexible way to compute and handle representations from pre-trained models, supporting GPU (CUDA or MPS) or CPU computation.
    """
    def __init__(self, model: str, batch_size: int):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.model = None
        self._load_model(model)

    def move_to_device(self, device: str):
        self.device = device

    def add_head(self, head: torch.nn.Module):
        self.add_module('head', head)
        self.head = head

    def compute_representations(self, sequences: List[str],
                                average_pooling: bool,
                                cls_token: Optional[bool] = False) -> List[torch.Tensor]:
        self.model.to(self.device)
        self.model.eval()
        batched_sequences = self._divide_in_batches(sequences, self.batch_size)
        output = []
        if average_pooling and cls_token:
            average_pooling = False
            print('Warning: Average pooling and CLS token are incompatible. Defaulting to CLS token.')
        for batch in tqdm(batched_sequences):
            output.extend(self.compute_batch(batch, average_pooling, cls_token))
        return output

    def compute_batch(self, batch: List[str],
                      average_pooling: bool,
                      cls_token: Optional[bool] = False) -> List[torch.Tensor]:
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
            if average_pooling:
                output.append(embd_rpr[idx, :seq_len].mean(0).detach().cpu())
            elif cls_token:
                output.append(embd_rpr[idx, 0].detach().cpu())
            else:
                output.append(embd_rpr[idx, :seq_len].detach().cpu())
        return output

    def dim(self) -> int:
        return self.dimension

    def forward(self, batch, labels=None):
        inputs = self.tokenizer(batch, add_special_tokens=True,
                                padding="longest", return_tensors="pt")
        output = self.model(**inputs).last_hidden_state

        if self.head is not None:
            output = self.head(output, labels)
        return output

    def max_len(self) -> int:
        if self.lab == 'Rostlab':
            return -1
        elif self.lab == 'facebook':
            return 1022

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_trainable_parameters(self):
        total = self.get_num_params()
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        percentage = (trainable / total) * 100
        print(f'trainable params: {trainable} || all params: {total} || trainable: {round(percentage, 2)} %')

    def _load_model(self, model: str):
        if model not in AVAILABLE_MODELS and SYNONYMS[model.lower()] not in AVAILABLE_MODELS:
            raise NotImplementedError(f"Model: {model} not implemented. Available models: {', '.join(AVAILABLE_MODELS)}")
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

    def _divide_in_batches(self, sequences: List[str],
                           batch_size: int) -> List[List[str]]:
        if self.lab == 'Rostlab':
            sequences = [' '.join([char for char in seq]) for seq in sequences]
        if self.model_name == 'ProstT5':
            sequences = ["<AA2fold> " + seq for seq in sequences]
        sequences = [seq[:self.max_len()] for seq in sequences]
        return [sequences[i: i+batch_size] for i in range(0, len(sequences), batch_size)]
