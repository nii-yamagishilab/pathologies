# Taken from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
# Additional license and copyright information for this source code are available at:
# https://github.com/gpleiss/temperature_scaling/blob/master/LICENSE

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return self.temperature_scale(outputs.logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        return logits / self.temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, n_bins):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss(n_bins).cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            num_batches = len(valid_loader)
            for batch in tqdm(valid_loader, total=num_batches):
                batch = [b.cuda() for b in batch]
                inputs = self.model.build_inputs(batch)
                logits = self(**inputs)
                logits_list.append(logits)
                labels_list.append(inputs["labels"])
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        nll = nll_criterion(logits, labels).item()
        ece, acc = ece_criterion(logits, labels)
        print(f"Initial temperature: {self.temperature.item():.3f}, n_bins: {n_bins}")
        print(
            f"Before temperature - NLL: {nll:.3f}, Acc: {acc:.2f}, ECE: {ece*100:.2f}"
        )

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        nll = nll_criterion(self.temperature_scale(logits), labels).item()
        ece, acc = ece_criterion(self.temperature_scale(logits), labels)
        print(f"Optimal temperature: {self.temperature.item():.3f}")
        print(f"After temperature - NLL: {nll:.3f}, Acc: {acc:.2f}, ECE: {ece*100:.2f}")
        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        acc = (
            accuracy_score(
                predictions.detach().cpu().numpy(), labels.detach().cpu().numpy()
            )
            * 100.0
        )

        return ece.item(), acc
