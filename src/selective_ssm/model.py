import torch
import torch.nn as nn

from .modules import MambaLayer, RMSNorm


class Mamba(nn.Module):
    """
    Full Mamba model for sequence modeling.

    Architecture:
    -> Embedding layer (for discrete inputs) or input projection (for continuous)
    -> Stack of MambaLayer blocks (each with RMSNorm + MambaBlock + residual)
    -> Final normalization
    -> Output projection head

    Args:
        d_model: Model dimension (hidden size)
        n_layers: Number of Mamba layers to stack
        vocab_size: Vocabulary size for embeddings (None for continuous inputs)
        d_state: SSM state dimension (N in paper), default 16
        d_conv: Convolution kernel size, default 4
        expand_factor: Expansion factor (E in paper), default 2
        dt_rank: Rank of Δ projection, default "auto" (sets to d_model // 16)
        dt_min: Minimum value for Δ initialization, default 0.001
        dt_max: Maximum value for Δ initialization, default 0.1
        dt_init: Initialization scheme for Δ, default "random"
        dt_scale: Scaling factor for Δ, default 1.0
        bias: Whether to use bias in linear layers, default False
        conv_bias: Whether to use bias in conv layers, default True
        pscan: Whether to use parallel scan, default True
        use_cuda: Whether to use custom CUDA kernel (if available), default False
        dropout: Dropout rate, default 0.0
        tie_embeddings: Whether to tie input/output embeddings, default False
        pad_vocab_size_multiple: Pad vocab size to multiple of this, default 8
    """

    def __init__(
        self,
        d_model,
        n_layers,
        vocab_size=None,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        bias=False,
        conv_bias=True,
        pscan=True,
        use_cuda=False,
        dropout=0.0,
        tie_embeddings=False,
        pad_vocab_size_multiple=8,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings

        # Helps with tensor core utilization
        if vocab_size is not None:
            if vocab_size % pad_vocab_size_multiple != 0:
                vocab_size = (
                    vocab_size // pad_vocab_size_multiple + 1
                ) * pad_vocab_size_multiple
            self.vocab_size = vocab_size

        if vocab_size is not None:
            # Discrete inputs -> use embedding layer
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            # Continuous inputs -> use linear projection
            self.embedding = None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Stack of Mamba layers
        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand_factor,
                )
                for _ in range(n_layers)
            ]
        )

        # RMSNorm before the output projection
        self.norm_f = RMSNorm(d_model)

        # Output projection head
        if vocab_size is not None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

            # Should reduce parameter count and improve performance?
            if tie_embeddings and self.embedding is not None:
                self.lm_head.weight = self.embedding.weight
        else:
            self.lm_head = None

        # Init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Embeddings: normal distribution
        Linear layers: normal distribution with proper scaling
        SSM parameters: handled within SelectiveSSM
        """
        if isinstance(module, nn.Embedding):
            # Standard embedding initialization
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            # Standard linear layer initialization
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, return_hidden_states=False):
        """
        Forward pass through the Mamba model.

        Args:
            input_ids: Input tensor of shape (batch_size, seq_len) for discrete inputs
                      or (batch_size, seq_len, d_model) for continuous inputs
            return_hidden_states: If True, return hidden states from all layers

        Returns:
            If return_hidden_states is False:
                logits: Output tensor of shape (batch_size, seq_len, vocab_size)
                       or (batch_size, seq_len, d_model)
            If return_hidden_states is True:
                (logits, hidden_states): Tuple of output and list of hidden states
        """
        if self.embedding is not None:
            # Discrete
            x = self.embedding(input_ids)
        else:
            # Continuous
            x = input_ids

        x = self.dropout(x)

        # Pass through Mamba layers
        hidden_states = []
        for layer in self.layers:
            x = layer(x)
            if return_hidden_states:
                hidden_states.append(x)

        x = self.norm_f(x)

        # Output projection
        if self.lm_head is not None:
            # Language modeling => project to vocabulary
            logits = self.lm_head(x)
        else:
            # Else rreturn hidden states
            logits = x

        if return_hidden_states:
            return logits, hidden_states
        return logits

    def generate(
        self,
        input_ids,
        max_length,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id=None,
    ):
        """
        Generate sequences autoregressively.

        Args:
            input_ids: Starting tokens of shape (batch_size, seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, only sample from top p probability mass (nucleus sampling)
            eos_token_id: Token ID to stop generation

        Returns:
            generated: Generated token IDs of shape (batch_size, max_length)
        """
        assert self.lm_head is not None, "Generate only works with language models"

        # batch_size = input_ids.shape[0]
        # device = input_ids.device

        # Start with input_ids
        generated = input_ids.clone()
        # Gen one token at a time
        for _ in range(max_length - input_ids.shape[1]):
            # Full forward pass rn but should be recurrent mode later
            output = self.forward(generated)

            assert isinstance(output, torch.Tensor), "Expected tensor output"
            logits = output
            next_token_logits = logits.select(1, -1)  # (B, V)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k
            if top_k is not None:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # nucleus filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with proba above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            n_params: Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.embedding is not None:
            n_params -= self.embedding.weight.numel()
        return n_params


class MambaConfig:
    """
    Configuration class for Mamba model, uses GPT-3 params.
    Makes it easy to create models with different sizes.
    """

    PRESETS = {
        "mamba-130m": {
            "d_model": 768,
            "n_layers": 24,
            "vocab_size": 50277,  # GPT-2 tokenizer
        },
        "mamba-370m": {
            "d_model": 1024,
            "n_layers": 48,
            "vocab_size": 50277,
        },
        "mamba-790m": {
            "d_model": 1536,
            "n_layers": 48,
            "vocab_size": 50277,
        },
        "mamba-1.4b": {
            "d_model": 2048,
            "n_layers": 48,
            "vocab_size": 50277,
        },
        "mamba-2.8b": {
            "d_model": 2560,
            "n_layers": 64,
            "vocab_size": 50277,
        },
    }

    def __init__(self, preset=None, **kwargs):
        """
        Initialize config from preset or custom parameters.

        Args:
            preset: Name of preset configuration (e.g., "mamba-130m")
            **kwargs: Override any default parameters
        """
        self.d_model = 768
        self.n_layers = 24
        self.vocab_size = 50277
        self.d_state = 16
        self.d_conv = 4
        self.expand_factor = 2
        self.dt_rank = "auto"
        self.dt_min = 0.001
        self.dt_max = 0.1
        self.dt_init = "random"
        self.dt_scale = 1.0
        self.bias = False
        self.conv_bias = True
        self.pscan = True
        self.use_cuda = False
        self.dropout = 0.0
        self.tie_embeddings = False
        self.pad_vocab_size_multiple = 8

        # Load preset if specified
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError(
                    f"Unknown preset '{preset}'. Available: {list(self.PRESETS.keys())}"
                )
            preset_config = self.PRESETS[preset]
            for key, value in preset_config.items():
                setattr(self, key, value)

        # Override with any custom parameters
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown config parameter: {key}")
            setattr(self, key, value)

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)


def create_mamba_model(preset="mamba-130m", **kwargs):
    """
    Create a Mamba model from a preset.

    Args:
        preset: Name of preset configuration (e.g., "mamba-130m")
        **kwargs: Override any config parameters

    Returns:
        model: Mamba model instance

    """
    config = MambaConfig(preset=preset, **kwargs)
    return Mamba(**config.to_dict())
