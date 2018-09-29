import torch
import torchvision.models as M
import functools

class WithSavedActivations:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.detach = True

        # We want to save activations of convs and relus. Also, MaxPool creates
        # actifacts so we replace them with AvgPool that makes things a bit
        # cleaner.

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(functools.partial(self._save, name))


    def _save(self, name, module, input, output):
        if self.detach:
            self.activations[name] = output.detach().clone()
        else:
            self.activations[name] = output.clone()

    def __call__(self, input, detach):
        self.detach = detach
        self.activations = {}
        self.model(input)
        acts = self.activations
        self.activations = {}
        return acts

    def to(self, device):
        self.model = self.model.to(device)


def PerceptualNet():
    m = M.vgg19(pretrained=True).eval()
    m = m.features[:30]
    m = WithSavedActivations(m)
    return m

