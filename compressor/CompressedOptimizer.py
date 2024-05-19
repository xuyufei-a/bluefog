import torch

class _CompressedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, Compressor=None):
        super(self.__class__, self).__init__(params)
        self.Compressor = Compressor

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and self.Compressor:
                    p.grad = self.Compressor(p.grad)
                    if p.grad.is_sparse:
                        p.grad = p.grad.to_dense()

        super(self.__class__, self).step()
        return loss

def CompressedOptimizer(optimizer, Compressor):
    cls = type(optimizer.__class__.__name__, 
               (optimizer.__class__,), 
               dict(_CompressedOptimizer.__dict__))
    return cls(optimizer.param_groups, Compressor=Compressor)