import pytest
import torch
import torch.nn as nn

from src.selective_ssm import Mamba, MambaConfig, create_mamba_model


class TestMambaModel:
    """Test suite for the full Mamba model."""

    def test_model_construction(self):
        """Test that model can be constructed with various configurations."""
        # Small model for testing
        model = Mamba(
            d_model=64,
            n_layers=2,
            vocab_size=100,
            d_state=8,
        )

        assert isinstance(model, nn.Module)
        assert model.d_model == 64
        assert model.n_layers == 2
        assert model.vocab_size == 104  # Padded to multiple of 8
        assert isinstance(model.layers, nn.ModuleList)
        assert len(list(model.layers)) == 2

    def test_model_forward_discrete(self):
        """Test forward pass with discrete (token) inputs."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        d_model = 64

        model = Mamba(
            d_model=d_model,
            n_layers=2,
            vocab_size=vocab_size,
            d_state=8,
        )

        # Create random token inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = model(input_ids)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, model.vocab_size)

    def test_model_forward_continuous(self):
        """Test forward pass with continuous inputs."""
        batch_size = 2
        seq_len = 10
        d_model = 64

        model = Mamba(
            d_model=d_model,
            n_layers=2,
            vocab_size=None,  # No vocabulary for continuous inputs
            d_state=8,
        )

        # Create random continuous inputs
        inputs = torch.randn(batch_size, seq_len, d_model)

        # Forward pass
        outputs = model(inputs)

        # Check output shape (should be same as input for continuous)
        assert outputs.shape == (batch_size, seq_len, d_model)

    def test_hidden_states_return(self):
        """Test that hidden states can be returned."""
        batch_size = 2
        seq_len = 10
        model = Mamba(
            d_model=64,
            n_layers=3,
            vocab_size=100,
            d_state=8,
        )

        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        # Forward with hidden states
        logits, hidden_states = model(input_ids, return_hidden_states=True)

        # Check we get hidden states from all layers
        assert len(hidden_states) == 3
        for h in hidden_states:
            assert h.shape == (batch_size, seq_len, 64)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = Mamba(
            d_model=32,
            n_layers=2,
            vocab_size=50,
            d_state=4,
        )

        input_ids = torch.randint(0, 50, (2, 5))
        logits = model(input_ids)

        # Compute dummy loss
        target = torch.randint(0, 50, (2, 5))
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )

        # Backpropagate
        loss.backward()

        # Check that parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_tied_embeddings(self):
        """Test that embedding tying works correctly."""
        model = Mamba(
            d_model=64,
            n_layers=2,
            vocab_size=100,
            tie_embeddings=True,
        )

        # Check that weights are shared
        assert model.embedding.weight is model.lm_head.weight

        # Check parameter count is reduced
        n_params_tied = model.get_num_params()

        model_untied = Mamba(
            d_model=64,
            n_layers=2,
            vocab_size=100,
            tie_embeddings=False,
        )
        n_params_untied = model_untied.get_num_params()

        # Tied model should have fewer parameters
        assert n_params_tied < n_params_untied

    def test_generation(self):
        """Test autoregressive generation."""
        model = Mamba(
            d_model=64,
            n_layers=2,
            vocab_size=100,
            d_state=8,
        )
        model.eval()

        # Start with a few tokens
        input_ids = torch.randint(0, 100, (1, 5))

        # Generate more tokens
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=20,
                temperature=1.0,
            )

        # Check output shape
        assert generated.shape[0] == 1
        assert generated.shape[1] == 20
        assert (generated[:, :5] == input_ids).all()  # Should start with input

    def test_parameter_count(self):
        """Test parameter counting function."""
        model = Mamba(
            d_model=64,
            n_layers=2,
            vocab_size=100,
        )

        # Count all parameters
        n_params_all = model.get_num_params(non_embedding=False)

        # Count non-embedding parameters
        n_params_non_emb = model.get_num_params(non_embedding=True)

        # Non-embedding count should be less
        assert n_params_non_emb < n_params_all

        # Verify manually
        expected_all = sum(p.numel() for p in model.parameters())
        assert n_params_all == expected_all


class TestMambaConfig:
    """Test suite for MambaConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MambaConfig()

        assert config.d_model == 768
        assert config.n_layers == 24
        assert config.d_state == 16

    def test_preset_config(self):
        """Test preset configurations."""
        config = MambaConfig(preset="mamba-370m")

        assert config.d_model == 1024
        assert config.n_layers == 48

    def test_override_config(self):
        """Test overriding config parameters."""
        config = MambaConfig(
            preset="mamba-130m",
            d_state=32,  # Override default
            dropout=0.1,
        )

        assert config.d_model == 768  # From preset
        assert config.d_state == 32  # Overridden
        assert config.dropout == 0.1  # Overridden

    def test_config_to_dict(self):
        """Test config to dict conversion."""
        config = MambaConfig(d_model=512, n_layers=12)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["d_model"] == 512
        assert config_dict["n_layers"] == 12

    def test_config_from_dict(self):
        """Test config from dict creation."""
        config_dict = {
            "d_model": 512,
            "n_layers": 12,
            "vocab_size": 50000,
        }
        config = MambaConfig.from_dict(config_dict)

        assert config.d_model == 512
        assert config.n_layers == 12
        assert config.vocab_size == 50000

    def test_invalid_preset(self):
        """Test that invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            MambaConfig(preset="invalid-preset")

    def test_invalid_parameter(self):
        """Test that invalid parameter raises error."""
        with pytest.raises(ValueError, match="Unknown config parameter"):
            MambaConfig(invalid_param=123)


class TestCreateMambaModel:
    """Test suite for model creation helper."""

    def test_create_from_preset(self):
        """Test creating model from preset."""
        model = create_mamba_model("mamba-130m")

        assert isinstance(model, Mamba)
        assert model.d_model == 768
        assert model.n_layers == 24

    def test_create_with_overrides(self):
        """Test creating model with config overrides."""
        model = create_mamba_model(
            "mamba-130m",
            dropout=0.1,
            d_state=32,
        )

        # Check that model was created
        assert isinstance(model, Mamba)
        assert model.d_model == 768  # From preset


class TestModelSizes:
    """Test that model sizes approximately match paper specifications."""

    @pytest.mark.parametrize(
        "preset,expected_params_millions",
        [
            ("mamba-130m", 182),
            pytest.param("mamba-370m", 360, marks=pytest.mark.slow),
            pytest.param("mamba-790m", 760, marks=pytest.mark.slow),
            pytest.param("mamba-1.4b", 1300, marks=pytest.mark.slow),
            pytest.param("mamba-2.8b", 2500, marks=pytest.mark.slow),
        ],
    )
    def test_model_size(self, preset, expected_params_millions):
        model = create_mamba_model(preset)
        n_params = model.get_num_params(non_embedding=True) / 1e6

        tolerance = expected_params_millions * 0.2
        assert abs(n_params - expected_params_millions) < tolerance, (
            f"{preset}: Expected ~{expected_params_millions}"
            f" M params, got {n_params:.1f}M"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
