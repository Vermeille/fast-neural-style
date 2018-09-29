import torch
import torch.nn.functional as F

class SRLoss:
    def __init__(self, model, style, ratio=1e3):
        self.ratio = ratio
        self.model = model
        self.style_grams = SRLoss.get_matrices(model, style, detach=True)

    @staticmethod
    def get_matrices(m, style, detach):
        acts = m(style, detach=detach)
        acts = {k: gram(v) for k, v in acts.items() if k in ['1', '6']}
        return acts

    def __call__(self, x):
        batch_size_changed = (
                x.dim() != self.style_grams['1'].dim()
                or x.shape[0] != self.style_grams['1'].shape[0]
        )

        if batch_size_changed:
            self.style_grams = {
                k: v.expand(x.shape[0], -1, -1)
                for k, v in self.style_grams.items()
            }

        train_style = SRLoss.get_matrices(self.model, x, detach=False)
        loss = 0
        for l in train_style:
            layer_loss = F.mse_loss(train_style[l], self.style_grams[l])
            loss += 0.2 * self.ratio * layer_loss
        return loss

