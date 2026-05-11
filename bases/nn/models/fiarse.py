import torch

class Bern(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold):
        # mask on the parameters greater than threshold
        scalar = 1
        ctx.save_for_backward(scores, scalar*threshold)
        return (scores>=threshold)

    @staticmethod
    def backward(ctx, grad_output):
        scores, threshold = ctx.saved_tensors
        grad = 2 * threshold / torch.pow(scores+threshold, 2)
        grad = torch.nan_to_num(grad, nan=0., posinf=0., neginf=0.)
        mask = (scores>=threshold)
        return grad_output*grad*mask, None
    
