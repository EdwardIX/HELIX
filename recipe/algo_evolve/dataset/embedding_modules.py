import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.utils.model_manager import ModelManagerWithCPUOffload
from src.simplify import normalize_yaml, normalize_python_code

def _mean_pooling(last_hidden_state, attention_mask):
    # last_hidden_state : (B, L, H) on *gpu*
    # attention_mask    : (B, L)    on *gpu*
    assert last_hidden_state.dim() == 3 and attention_mask.dim() == 2 and \
           last_hidden_state.size(0) == attention_mask.size(0) and last_hidden_state.size(1) == attention_mask.size(1),\
           f"Invalid shapes: {last_hidden_state.shape}, {attention_mask.shape}"
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    pooled = (last_hidden_state * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
    return pooled  

class EmbeddingModule:
    def __init__(self):
        pass

    def encode(self, responses_to_embed):
        raise NotImplementedError
    
    def load(self):
        pass

    def release(self):
        pass

class TransformerEmbeddingModule(EmbeddingModule):
    def __init__(self, model_name: str, batch_size: int = 1024, max_len: int = 1024):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_len = max_len

        # Identify embedding model type
        if model_name in ('Salesforce/codet5p-110m-embedding', ):
            self.model_type = 'encoder'
        else:
            self.model_type = 'decoder_only'

        # Initialize model and tokenizer
        trust_remote_code = True if model_name in ('Salesforce/codet5p-110m-embedding', ) else False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = ModelManagerWithCPUOffload.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model.eval()
        if self.model_type == 'encoder':
            with torch.no_grad():
                self.embed_dim = self.model(**self.tokenizer("print(1+1)", return_tensors="pt")).size(-1)
        else:
            self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, responses_to_embed: list[str], device=torch.device('cuda')):
        """Batch compute embeddings and return torch.Tensor (N, D)"""
        ModelManagerWithCPUOffload.load_model(self.model_name)
        self.model.eval()

        all_vecs = []
        with torch.no_grad():
            for s in range(0, len(responses_to_embed), self.batch_size):
                batch = responses_to_embed[s : s + self.batch_size]
                tok = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                ).to(device)

                out = self.model(**tok)
                if self.model_type == 'encoder':
                    vec = out # out=BxH
                else:  # decoder_only
                    vec = _mean_pooling(out.last_hidden_state, tok["attention_mask"])  # out=B,L,H
                vec = F.normalize(vec, p=2, dim=-1)  # L2-norm
                all_vecs.append(vec.cpu())

        ModelManagerWithCPUOffload.release_model(self.model_name)
        torch.cuda.empty_cache()
        return torch.cat(all_vecs, dim=0)

    def load(self):
        ModelManagerWithCPUOffload.load_model(self.model_name)

    def release(self):
        ModelManagerWithCPUOffload.release_model(self.model_name)
        torch.cuda.empty_cache()

class PythonEmbeddingModule(TransformerEmbeddingModule):
    def encode(self, responses_to_embed, **kwargs):
        responses_to_embed = [normalize_python_code(resp) for resp in responses_to_embed]
        return super().encode(responses_to_embed, **kwargs)

class YamlEmbeddingModule(TransformerEmbeddingModule):
    def encode(self, responses_to_embed, **kwargs):
        responses_to_embed = [normalize_yaml(resp) for resp in responses_to_embed]
        return super().encode(responses_to_embed, **kwargs)