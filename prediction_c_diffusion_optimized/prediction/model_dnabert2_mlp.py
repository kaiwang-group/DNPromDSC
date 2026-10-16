import sys
from pathlib import Path
import torch
import torch.nn as nn

MODEL_PATH = Path(__file__).resolve().parent / "DNABERT_2"
INPUT_CHANNELS = 128
sys.path.append(str(MODEL_PATH))

try:
    from bert_layers import BertModel
except ImportError:
    print(f"Error: cannot import bert_layers.py from {MODEL_PATH}.")
    raise


class AttnPool1d(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class DNABERT2_Hybrid_Regression(nn.Module):
    def __init__(self, mlp_hidden_size=256, dropout_rate=0.2,
                 use_last4=True, pool_type="attn"):
        super().__init__()
        print(f">> [Model] Loading DNABERT-2 from: {MODEL_PATH} ...")
        self.bert = BertModel.from_pretrained(str(MODEL_PATH))
        self.use_last4, self.pool_type = use_last4, pool_type

        if hasattr(self.bert, "config") and hasattr(self.bert.config, "output_hidden_states"):
            self.bert.config.output_hidden_states = use_last4
        if hasattr(self.bert, "config") and hasattr(self.bert.config, "hidden_size"):
            self.bert_hidden = self.bert.config.hidden_size
        else:
            self.bert_hidden = 768

        self.attn_pool = AttnPool1d(self.bert_hidden) if pool_type == "attn" else None
        self.regressor = nn.Sequential(
            nn.LayerNorm(self.bert_hidden),
            nn.Linear(self.bert_hidden, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, 1)
        )

    def _get_token_reps(self, outputs):
        last_hidden_state = None
        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            last_hidden_state = outputs[0]

        hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        if self.use_last4 and hidden_states is not None and len(hidden_states) >= 4:
            return torch.stack(hidden_states[-4:], dim=0).mean(dim=0)
        if last_hidden_state is None:
            raise RuntimeError("Unable to parse DNABERT2 output.")
        return last_hidden_state

    def _pool(self, token_reps, attention_mask=None):
        if self.pool_type == "cls":
            return token_reps[:, 0, :]
        return self.attn_pool(token_reps)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.use_last4
        )
        token_reps = self._get_token_reps(outputs)
        pooled = self._pool(token_reps, attention_mask=attention_mask)
        return self.regressor(pooled)


if __name__ == "__main__":
    model = DNABERT2_Hybrid_Regression(256, 0.2, True, "attn")
    print(model)
    print("\nRegression head output dimension = 1")