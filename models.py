import torch
import torch.nn as nn

try:
    from torch.nn.functional import scaled_dot_product_attention
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


class Transformer(nn.Module):
    """Simple Transformer encoder for sequence classification.

    Parameters
    ----------
    depth : int
        number of encoder layers
    dim : int
        model dimension
    heads : int
        number of attention heads
    n_tokens : int
        vocabulary size
    seq_len : int
        maximum sequence length
    dropout : float, optional
        dropout probability, by default 0.0
    pool : str, optional
        'cls' uses last token, 'mean' averages sequence, by default 'cls'
    """

    def __init__(self, depth: int, dim: int, heads: int, n_tokens: int, seq_len: int,
                 dropout: float = 0.0, pool: str = 'cls') -> None:
        super().__init__()
        assert pool in {'cls', 'mean'}, "pool must be 'cls' or 'mean'"
        self.pool = pool
        self.token_emb = nn.Embedding(n_tokens, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        
        if FLASH_ATTENTION_AVAILABLE and torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            batch_first=True,
            activation='silu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, n_tokens, bias=False)

        self._init_params()

    def _init_params(self) -> None:
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=1.0 / (self.token_emb.embedding_dim ** 0.5))
        nn.init.xavier_uniform_(self.out.weight)

    # convenience properties -------------------------------------------------
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def shapes(self):
        return {k: tuple(p.shape) for k, p in self.named_parameters()}

    def summary(self):
        print(self)
        print(f'Number of parameters: {self.num_params}')

    # forward ----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (b, n)
        b, n = x.shape
        tok = self.token_emb(x)                        # (b, n, d)
        tok = tok + self.pos_emb[:, :n]
        h = self.encoder(tok)                          # (b, n, d)
        if self.pool == 'mean':
            h = h.mean(dim=1)
        else:
            h = h[:, -1]  # last token
        return self.out(self.norm(h))


if __name__ == '__main__':
    n_tokens = 10
    seq_len = 4
    model = Transformer(depth=2, dim=128, heads=4, n_tokens=n_tokens, seq_len=seq_len)
    model.summary()
    x = torch.randint(0, n_tokens, (8, seq_len))
    y = model(x)
    print(y.shape)
