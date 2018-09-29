import torch


def gram(m):
    b, c, h, w = m.shape
    m = m.view(b, c, h * w)
    m1 = m
    m2 = m.permute(0, 2, 1)
    g = torch.bmm(m1, m2) / (c * h * w)
    return g


class StyleLoss:
    def __init__(self, model, style, ratio=1e3):
        self.ratio = ratio
        self.model = model
        self.style_grams, _ = StyleLoss.get_matrices(model, style, detach=True)

    @staticmethod
    def get_matrices(m, style, detach):
        acts = m(style, detach=detach)
        content = acts['22']
        acts = {
                k: gram(v)
                for k, v in acts.items()
                if k in ['1', '6', '11', '20', '29']
        }
        return acts, content

    def __call__(self, x, content):
        batch_size_changed = (
                content.dim() != self.style_grams['1'].dim()
                or content.shape[0] != self.style_grams['1'].shape[0]
        )

        if batch_size_changed:
            self.style_grams = {
                k: v.expand(content.shape[0], -1, -1)
                for k, v in self.style_grams.items()
            }

        ref_content = self.model(content, detach=True)['22']
        train_style, train_content = StyleLoss.get_matrices(
                self.model, x, detach=False)
        loss = 0
        for l in train_style:
            layer_loss = F.mse_loss(train_style[l], self.style_grams[l])
            loss += 0.2 * self.ratio * layer_loss

        crop = train_content[:, :, :ref_content.shape[2], :ref_content.shape[3]]
        loss += F.mse_loss(crop, ref_content)
        return loss

