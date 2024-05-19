import torch

class Compressor:
    def __init__(self, device):
        self.device = device

    def _convert_idx_1d_nd(self, index, shape):
        keys = torch.zeros(len(shape), len(index), dtype=torch.int32, device=index.device)
        for i, d in enumerate(shape[::-1]):
            keys[len(shape)-i-1] = torch.remainder(index, d)
            index = torch.div(index, d, rounding_mode='floor').int() 
        return keys

class TopKCompressor(Compressor):
    def __init__(self, *, k=None, percentage=None, device=torch.device('cuda:0')):
        if k is None and percentage is None:
            raise ValueError("At least one of 'k' or 'percentage' must be provided")
        if k and percentage:
            raise ValueError("The 'k' and 'percentage' parameters are mutually exclusive.")
        if percentage and (percentage < 0 or percentage > 1):
            raise ValueError("'percentage' must be a float number between 0 and 1")

        super().__init__(device=device)
        if k:
            self.k = k
        else:
            self.percentage = percentage


    def __call__(self, x):
        if hasattr(self, "k"):
            k = self.k
        else:
            k = max(int(self.percentage * x.numel()), 1)

        if x.is_sparse:
            _, indices = torch.topk(x.abs()._values(), min(k, x._nnz()))
            values = x._values()[indices]
            keys = x._indices()[:, indices]
        else:
            _, indices = torch.topk(x.abs().view(-1), min(k, x.numel()))
            values = x.view(-1)[indices]
            keys = self._convert_idx_1d_nd(indices, x.shape)

        return torch.sparse_coo_tensor(keys, values, size=x.shape)

class RandomKCompressor(Compressor):
    def __init__(self, *, k=None, percentage=None, device=torch.device('cuda:0')):
        if k is None and percentage is None:
            raise ValueError("At least one of 'k' or 'percentage' must be provided")
        if k and percentage:
            raise ValueError("The 'k' and 'percentage' parameters are mutually exclusive.")
        if percentage and (percentage < 0 or percentage > 1):
            raise ValueError("'percentage' must be a float number between 0 and 1")

        super().__init__(device=device)
        self.device = device
        if k:
            self.k = k
        else:
            self.percentage = percentage 
    
    def __call__(self, x):
        if hasattr(self, "k"):
            k = self.k
        else:
            k = max(int(self.percentage * x.numel()), 1)

        if x.is_sparse:
            index = torch.randperm(x._nnz(), device=self.device)[:k]
            keys = x._indices()[:, index]
            values = x._values()[index]
        else:
            index = torch.randperm(x.numel(), device=self.device)[:k]
            values = x.view(-1)[index]
            keys = self._convert_idx_1d_nd(index, x.shape)

        return torch.sparse_coo_tensor(keys, values, size=x.shape)

class QuantizedCompressor(Compressor):
    def __init__(self, s, device=torch.device('cuda:0')):
        super().__init__(device=device)
        self.s = s

    def __call__(self, x : torch.Tensor):
        shape = x.shape
        is_sparse = x.is_sparse

        if is_sparse:
            indices = x._indices()
            x = x._values()
        else: 
            x = x.view(-1)

        norm = torch.max(torch.abs(x))
        scale_x = x.abs() / norm * self.s
        l = torch.clamp(scale_x, 0, self.s - 1).int()
        p = scale_x - l.float()

        r = torch.rand(p.size(), device=x.device)
        l += (r < p).int()
        signs = (x > 0)

        if is_sparse:
            return torch.sparse_coo_tensor(indices, norm * (2 * signs.float() - 1) * l.float() / self.s, shape)
        else:
            # return norm, signs.view(shape), l.view(shape)
            return (norm * (2 * signs.float() - 1) * l.float() / self.s).view(shape)
