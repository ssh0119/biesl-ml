"""Compare parameter sizes and check forward/backward/checkpoint compatibility."""
import io
import json
import torch
from src.cnn.model import Phase2CNN1D, count_parameters
from src.cnn.expanded_model import ExpandedPhase2CNN1D


def main():
    torch.set_num_threads(2)
    results = {}
    for name, cls in (("baseline", Phase2CNN1D), ("expanded", ExpandedPhase2CNN1D)):
        torch.manual_seed(42)
        model = cls()
        x = torch.randn(2, 3, 10000)
        logits = model(x)
        assert logits.shape == (2, 5)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            z = model.features(x)
            pooled = torch.cat((model.avg_pool(z), model.max_pool(z)), dim=1).squeeze(-1)
            hidden = model.classifier[:3](pooled)
            assert pooled.shape == (2, 256) and hidden.shape == (2, 96)
            torch.testing.assert_close(model.classifier[3:](hidden), model(x))
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            checkpoint_bytes = buffer.tell()
            buffer.seek(0)
            restored = cls().eval()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            torch.testing.assert_close(restored(x), model(x))
        results[name] = {"parameters": count_parameters(model),
                         "parameter_mib_fp32": sum(p.numel()*p.element_size() for p in model.parameters()) / 2**20,
                         "checkpoint_bytes": checkpoint_bytes,
                         "pooled_features": 256, "embedding_features": 96}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
