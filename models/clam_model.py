import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
CLAM für binäres MIL mit CNN-Backbone (angepasst an Eriks Attention-MIL-Setup).

Geprüft gegen mahmoodlab/CLAM (models/model_clam.py, CLAM_SB).
Bewusste Abweichungen vom Original sind im Code markiert:

  [ANPASSUNG 1] CNN-Backbone statt vorextrahierter Features.
      Original-CLAM bekommt fertige ResNet-50-Features (1024-dim). Hier ist der
      Feature-Extraktor Teil des Modells, damit End-to-End trainiert werden kann.

  [ANPASSUNG 2] Ein einzelner Instanz-Klassifikator statt einer pro Klasse.
      Original hält n_classes Instanz-Klassifikatoren (für Subtyping). Bei binärer
      Aufgabe (Ähre/keine Ähre) genügt ein Branch mit 2 Output-Logits.

  [ANPASSUNG 3] Out-of-the-class-Clustering läuft auch ohne Subtyping.
      Original überspringt inst_eval_out bei einfacher Klassifikation. Hier ist es
      aktiv, weil Eriks negative Bags garantiert nur negative Patches enthalten --
      "alle Top-Attention-Instanzen sind negativ" ist damit korrekte Supervision.
"""


class AttnNetGated(nn.Module):
    """Gated Attention (Ilse et al. 2018), wie im CLAM-Original."""
    def __init__(self, L=512, D=256, dropout=0., n_classes=1):
        super().__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a.mul(b))  # [N, n_classes]
        return A, x


class CLAM(nn.Module):
    def __init__(self, M=500, L=128, num_maps=50, kernel_size=5, pool_size=4,
                 in_channels=3, k_sample=8, pseudo_threshold = False, dropout=0.25,
                 instance_loss_fn=None, subtyping=False,
                 pseudo_quantile_pos=0.5, pseudo_quantile_neg=0.25):
        super().__init__()
        self.M = M
        self.L = L
        self.num_maps = num_maps
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.k_sample = k_sample
        self.pseudo_threshold = pseudo_threshold
        # Quantile fuer inst_eval_threshold: >= q_pos -> pseudo-positiv, <= q_neg -> pseudo-negativ
        self.pseudo_quantile_pos = pseudo_quantile_pos
        self.pseudo_quantile_neg = pseudo_quantile_neg
        self.subtyping = subtyping
        self.instance_loss_fn = instance_loss_fn if instance_loss_fn is not None else nn.CrossEntropyLoss()

        # -- [ANPASSUNG 1] CNN-Backbone (identisch zu model.Attention) --------
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, num_maps, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((pool_size, pool_size)),
        )
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(num_maps * pool_size * pool_size, M),
            nn.ReLU(),
        )

        # -- Attention-Zweig (gated, wie CLAM_SB) -----------------------------
        self.attention_net = AttnNetGated(L=M, D=L, dropout=dropout, n_classes=1)

        # -- Bag-Klassifikator: 2 Logits (negativ/positiv) --------------------
        self.classifiers = nn.Linear(M, 2)

        # -- [ANPASSUNG 2] EIN Instanz-Klassifikator --------------------------
        self.instance_classifier = nn.Linear(M, 2)

    # -- Instance-Clustering-Hilfsfunktionen (wie Original) -------------------

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, dtype=torch.long, device=device)

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, dtype=torch.long, device=device)

    def inst_eval(self, A, h):
        """In-the-class (positive Bag): Top-k -> positiv, Bottom-k -> negativ."""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k = min(self.k_sample, h.shape[0] // 2) if h.shape[0] >= 2 else 1

        top_p_ids = torch.topk(A, k, dim=1)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)

        all_targets = torch.cat([self.create_positive_targets(k, device),
                                 self.create_negative_targets(k, device)], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = self.instance_classifier(all_instances)
        preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        loss = self.instance_loss_fn(logits, all_targets)
        return loss, preds, all_targets

    def inst_eval_out(self, A, h):
        """Out-of-the-class (negative Bag): alle Top-k sind negativ."""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k = min(self.k_sample, h.shape[0]) if h.shape[0] >= 1 else 1

        top_p_ids = torch.topk(A, k, dim=1)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        targets = self.create_negative_targets(k, device)
        logits = self.instance_classifier(top_p)
        preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        loss = self.instance_loss_fn(logits, targets)
        return loss, preds, targets

    def inst_eval_threshold(self, A, h, label_is_pos):
        """Pseudo-Labels ueber Attention-Schwelle statt festem Top-k.
        Skaliert mit der tatsaechlichen Positiv-Zahl der Bag."""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        A_flat = A.view(-1)

        if label_is_pos:
            # Adaptiv: alles ueber dem pos-Quantil ist pseudo-positiv,
            # alles unter dem neg-Quantil pseudo-negativ. Mittelfeld bleibt ungelabelt.
            thr_pos = torch.quantile(A_flat, self.pseudo_quantile_pos)
            thr_neg = torch.quantile(A_flat, self.pseudo_quantile_neg)
            pos_ids = (A_flat >= thr_pos).nonzero(as_tuple=True)[0]
            neg_ids = (A_flat <= thr_neg).nonzero(as_tuple=True)[0]
            pos_inst = torch.index_select(h, 0, pos_ids)
            neg_inst = torch.index_select(h, 0, neg_ids)
            targets = torch.cat([self.create_positive_targets(len(pos_ids), device),
                                self.create_negative_targets(len(neg_ids), device)])
            instances = torch.cat([pos_inst, neg_inst], dim=0)
        else:
            # negative Bag: alle negativ (wie inst_eval_out)
            instances = h
            targets = self.create_negative_targets(h.shape[0], device)

        logits = self.instance_classifier(instances)
        return self.instance_loss_fn(logits, targets)
    # -- Forward --------------------------------------------------------------

    def forward(self, x, label=None, instance_eval=False):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.num_maps * self.pool_size * self.pool_size)
        H = self.feature_extractor_part2(H)  # [N, M]

        A, H = self.attention_net(H)         # A: [N, 1], H: [N, M]
        A = torch.transpose(A, 1, 0)         # [1, N]
        A_raw = A
        A = F.softmax(A, dim=1)              # ueber N Instanzen

        # -- Instance Clustering ----------------------------------------------
        instance_loss = torch.tensor(0.0, device=H.device)
        if instance_eval and label is not None:
            lbl = int(label.item())
            if self.pseudo_threshold:
                instance_loss = self.inst_eval_threshold(A, H, label_is_pos=(lbl == 1))
            else:
                if lbl == 1:
                    instance_loss, _, _ = self.inst_eval(A, H)        # positive Bag
                else:
                    # [ANPASSUNG 3]: bei dir immer aktiv (negative Bags sind rein negativ)
                    instance_loss, _, _ = self.inst_eval_out(A, H)    # negative Bag

        # -- Bag-Aggregation und -Klassifikation ------------------------------
        Mvec = torch.mm(A, H)                # [1, M]
        logits = self.classifiers(Mvec)      # [1, 2]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]  # [1, 1]

        return logits, Y_prob, Y_hat, A_raw, instance_loss

    # -- Trainings-/Eval-Schnittstelle ----------------------------------------

    def calculate_objective(self, X, Y, instance_eval=True, bag_weight=0.7):
        """total = bag_weight*Bag-CE + (1-bag_weight)*Instanz-CE (Paper-Default 0.7)."""
        Y = Y.long().view(-1)
        logits, Y_prob, _, A_raw, instance_loss = self.forward(X, label=Y, instance_eval=instance_eval)
        bag_loss = F.cross_entropy(logits, Y)
        total_loss = bag_weight * bag_loss + (1. - bag_weight) * instance_loss
        return total_loss, A_raw, bag_loss, instance_loss

    def calculate_classification_error(self, X, Y):
        Y = Y.long().view(-1)
        _, _, Y_hat, _, _ = self.forward(X)
        error = 1. - Y_hat.view(-1).eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def count_positive_instances(self, X, threshold=0.5):
        """Zaehlen ueber den Instanz-Klassifikator (direkter als Attention-Threshold)."""
        self.eval()
        with torch.no_grad():
            x = X.squeeze(0)
            K = x.shape[0]
            threshold = 1/K
            H = self.feature_extractor_part1(x)
            H = H.view(-1, self.num_maps * self.pool_size * self.pool_size)
            H = self.feature_extractor_part2(H)
            inst_probs = F.softmax(self.instance_classifier(H), dim=1)[:, 1]
            count = int((inst_probs > threshold).sum().item())
        return count, inst_probs

    # -- Counting-Threshold-Auswertung ---------------------------------------

    @staticmethod
    def otsu_threshold(scores, n_bins=64):
        """Otsu-Schwelle fuer ein 1D-Score-Array in [0,1]. Pro Bag anwendbar."""
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            return 0.5
        hist, edges = np.histogram(scores, bins=n_bins, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0.5
        p = hist / total
        omega = np.cumsum(p)                      # kumulative Klassenwahrscheinlichkeit
        centers = (edges[:-1] + edges[1:]) / 2.0
        mu = np.cumsum(p * centers)
        mu_t = mu[-1]
        denom = omega * (1.0 - omega)
        denom[denom == 0] = 1e-12
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom   # Zwischenklassen-Varianz
        return float(centers[np.argmax(sigma_b2)])

    @staticmethod
    def _gate(pred, pred_pos):
        """Bag-Gating: setzt den Count auf 0 fuer Bags, die als negativ
        klassifiziert wurden. pred_pos ist ein bool-Array (True = Bag positiv
        vorhergesagt) in derselben Reihenfolge wie scores_per_bag. None = kein
        Gating (Rueckwaertskompatibilitaet)."""
        pred = np.asarray(pred, dtype=np.float64)
        if pred_pos is None:
            return pred
        mask = np.asarray(pred_pos, dtype=bool)
        pred = pred.copy()
        pred[~mask] = 0.0
        return pred

    @classmethod
    def counting_scores_per_bag(cls, scores_per_bag, true_counts, thr, pred_pos=None):
        """Bias (signiert) und MAE ueber alle Bags bei festem globalem Threshold.
        Mit pred_pos werden negativ klassifizierte Bags auf Count 0 gegatet."""
        pred = np.array([(np.asarray(s) > thr).sum() for s in scores_per_bag], dtype=np.float64)
        pred = cls._gate(pred, pred_pos)
        true = np.asarray([int(c) for c in true_counts], dtype=np.float64)
        return float((pred - true).mean()), float(np.abs(pred - true).mean())

    @classmethod
    def counting_scores_otsu(cls, scores_per_bag, true_counts, pred_pos=None):
        """Wie counting_scores_per_bag, aber Otsu-Schwelle PRO Bag (bag-gegatet)."""
        pred = np.array([(np.asarray(s) > cls.otsu_threshold(s)).sum()
                         for s in scores_per_bag], dtype=np.float64)
        pred = cls._gate(pred, pred_pos)
        true = np.asarray([int(c) for c in true_counts], dtype=np.float64)
        return float((pred - true).mean()), float(np.abs(pred - true).mean())

    @classmethod
    def counting_scores_soft(cls, scores_per_bag, true_counts, pred_pos=None):
        """Soft-Count: Erwartungswert der Positiven = Summe der Instanz-
        Wahrscheinlichkeiten pro Bag. Kein Threshold noetig; optional bag-gegatet."""
        pred = np.array([float(np.asarray(s).sum()) for s in scores_per_bag], dtype=np.float64)
        pred = cls._gate(pred, pred_pos)
        true = np.asarray([int(c) for c in true_counts], dtype=np.float64)
        return float((pred - true).mean()), float(np.abs(pred - true).mean())

    @classmethod
    def calibrate_threshold(cls, scores_per_bag, true_counts, grid=None, pred_pos=None):
        """Waehlt den globalen Threshold mit MAE minimal (Tie-Break: |Bias|).
        MAE ist die eigentliche Zielgroesse; auf der zero-inflated Verteilung
        faellt der bias-neutrale Punkt nicht mit dem MAE-Minimum zusammen.
        MUSS auf dem VALIDIERUNGSSET aufgerufen werden. pred_pos gaten die
        Val-Bags konsistent zur Test-Auswertung."""
        if grid is None:
            grid = np.arange(0.02, 0.90, 0.02)
        best = None
        for thr in grid:
            bias, mae = cls.counting_scores_per_bag(scores_per_bag, true_counts, thr, pred_pos=pred_pos)
            key = (mae, abs(bias))
            if best is None or key < best[0]:
                best = (key, float(thr), bias, mae)
        return best[1], best[2], best[3]   # thr, bias, mae

    def extract_features(self, x):
        """Fuer visualize_features.py (t-SNE/UMAP)."""
        self.eval()
        with torch.no_grad():
            x_in = x.squeeze(0) if x.dim() == 5 else x
            H = self.feature_extractor_part1(x_in)
            H = H.view(-1, self.num_maps * self.pool_size * self.pool_size)
            H = self.feature_extractor_part2(H)
            A, H = self.attention_net(H)
            A = F.softmax(torch.transpose(A, 1, 0), dim=1)
        return H.cpu().numpy(), A.squeeze(0).cpu().numpy()