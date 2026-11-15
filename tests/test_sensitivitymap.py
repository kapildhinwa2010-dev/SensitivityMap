import pytest
import torch
from sensitivitymap import SensitivityMap, create_sensitivitymap

class TestSensitivityMap:
    def test_init(self):
        m = SensitivityMap(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = SensitivityMap(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_sensitivitymap()
        assert isinstance(m, SensitivityMap)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
