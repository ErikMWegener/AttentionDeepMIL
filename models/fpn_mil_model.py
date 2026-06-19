"""
fpn_mil_model.py
================

Feature-Pyramid Multiple-Instance-Learning (FPN-MIL) model.

Kombiniert drei Bausteine:

  1. **Backbone / Feature Extractor** aus dem AttentionDeepMIL `model.py`
     (Conv-BN-ReLU-Stack), umgebaut zu einer *hierarchischen* Bottom-up-
     Architektur, die mehrere Merkmalskarten {F^1, ..., F^S} (fein -> grob)
     liefert.

  2. **Feature Pyramid Network** aus Lin et al. (2017) -- Top-down-Pfad mit
     Lateral Connections, exakt in der Form aus Gl. (1) von Mourao et al.:

         R^S = Conv3x3(Conv1x1(F^S))
         R^s = Conv3x3(Conv1x1(F^s) + Up(R^{s+1})),   s = 1..S-1

  3. **Multi-scale Attention-based MIL** aus Mourao et al. (2026):
       - deep-supervised, skalenspezifische (gated) AbMIL-Aggregatoren A^s
         -> Bag-Embedding h^s und Vorhersage P^s pro Skala,
       - aufmerksamkeitsbasierter Multi-Scale-Aggregator M
         -> h_ms und finale Vorhersage P_ms.

Eingabe-Konvention (kompatibel zum ursprünglichen model.py):

    x : Tensor [1, K, C, Hp, Wp]   (ein Bag mit K Patches)
        -> intern zu [K, C, Hp, Wp] gesqueezt.

Jeder Patch wird identisch und unabhängig durch den Backbone+FPN geschickt.
Anschließend werden die 2D-Karten R^s zu Instanzen geflattet, sodass jeder
Feature-Map-Pixel über alle Patches eine MIL-Instanz darstellt
(n_s = K * h_s * w_s), genau wie bei Mourao et al.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Bottom-up Backbone (abgeleitet aus model.py feature_extractor_part1)
# ---------------------------------------------------------------------------
class ConvGNReLU(nn.Module):
    """Ein Conv-GroupNorm-ReLU-Block im Stil von model.py (kernel_size konfigurierbar).

    GroupNorm statt BatchNorm: Bei MIL ist jeder Forward-Pass *ein Bag* (K Patches
    als "Batch"). BatchNorm würde im Training Bag-spezifische Statistiken nutzen,
    bei eval aber die laufenden Mittelwerte – ein Train/Eval-Mismatch, der zu
    Zufallsgenauigkeit führt. GroupNorm normalisiert pro Sample und verhält sich
    in Training und Eval identisch (vgl. Commit bcaaef0 für die übrigen Modelle).
    """

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 5, num_groups: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups=min(num_groups, out_c), num_channels=out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BottomUpBackbone(nn.Module):
    """Hierarchischer CNN-Backbone im Stil von model.py.

    Erzeugt S Bottom-up-Merkmalskarten {F^1, ..., F^S} (fein -> grob).
    Jede Stufe halbiert die räumliche Auflösung (Stride x2 via MaxPool),
    analog zum Bottom-up-Pfad in Lin et al. (Scaling-Step 2).

    Args:
        in_channels:  Eingangskanäle (1 für Graustufen-Mammogramme/MNIST).
        base_channels: Kanäle der ersten Stufe; verdoppeln sich pro Stufe.
        num_scales:   Anzahl der Pyramiden-/Backbone-Stufen S.
        kernel_size:  Conv-Kernelgröße (5 wie in model.py).
        blocks_per_stage: Anzahl Conv-BN-ReLU-Blöcke je Stufe.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_scales: int = 3,
        kernel_size: int = 5,
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.stages = nn.ModuleList()

        c_in = in_channels
        c_out = base_channels
        self.out_channels: List[int] = []
        for _ in range(num_scales):
            layers = [ConvGNReLU(c_in, c_out, kernel_size)]
            for _ in range(blocks_per_stage - 1):
                layers.append(ConvGNReLU(c_out, c_out, kernel_size))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Downsample x2
            self.stages.append(nn.Sequential(*layers))
            self.out_channels.append(c_out)
            c_in = c_out
            c_out = c_out * 2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: [K, C, Hp, Wp] -> Liste [F^1 (fein), ..., F^S (grob)]."""
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


# ---------------------------------------------------------------------------
# 2. Feature Pyramid Network (Lin et al. / Mourao Gl. 1)
# ---------------------------------------------------------------------------
class FeaturePyramid(nn.Module):
    """Top-down-FPN mit Lateral Connections.

        R^S = Conv3x3(Conv1x1(F^S))
        R^s = Conv3x3(Conv1x1(F^s) + Up(R^{s+1}))

    Alle Ausgaben haben einheitlich d_x Kanäle.
    """

    def __init__(self, in_channels_list: List[int], d_x: int = 256):
        super().__init__()
        # 1x1 laterale Projektionen auf einheitliche Kanalzahl d_x
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, d_x, kernel_size=1) for c in in_channels_list]
        )
        # 3x3 Ausgabe-Convs zum Glätten von Aliasing nach dem Upsampling
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(d_x, d_x, kernel_size=3, padding=1) for _ in in_channels_list]
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """feats: [F^1 (fein), ..., F^S (grob)] -> [R^1 (fein), ..., R^S (grob)]."""
        # Laterale Projektionen
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats)]

        # Top-down-Pfad: von grob (Index S-1) nach fein (Index 0)
        for s in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[s + 1], size=laterals[s].shape[-2:], mode="nearest"
            )
            laterals[s] = laterals[s] + upsampled

        # 3x3-Glättung
        outs = [conv(l) for conv, l in zip(self.output_convs, laterals)]
        return outs


# ---------------------------------------------------------------------------
# 3a. Skalenspezifischer (gated) AbMIL-Aggregator (Mourao Gl. 2-3 / Ilse et al.)
# ---------------------------------------------------------------------------
class GatedAbMIL(nn.Module):
    """Gated Attention-based MIL Aggregator.

    Encoder-MLP: dx -> d ; gated Attention-Pooling -> Bag-Embedding h in R^d.
    Gibt zusätzlich die per-Instanz-Attention a (für Heatmaps) zurück.

        z_i = MLP(x_i)
        a_i = softmax_i( w^T ( tanh(V z_i) ⊙ sigm(U z_i) ) )
        h   = Σ_i a_i z_i
    """

    def __init__(self, dx: int, d: int = 256, L: int = 128, gated: bool = True):
        super().__init__()
        self.gated = gated
        self.encoder = nn.Sequential(nn.Linear(dx, d), nn.ReLU(inplace=True))
        self.att_V = nn.Sequential(nn.Linear(d, L), nn.Tanh())
        if gated:
            self.att_U = nn.Sequential(nn.Linear(d, L), nn.Sigmoid())
        self.att_w = nn.Linear(L, 1)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """X: [n, dx] -> (h: [d], a: [n])."""
        Z = self.encoder(X)                       # [n, d]
        if self.gated:
            A = self.att_w(self.att_V(Z) * self.att_U(Z))  # [n, 1]
        else:
            A = self.att_w(self.att_V(Z))                  # [n, 1]
        A = torch.softmax(A.transpose(0, 1), dim=1)        # [1, n], Softmax über Instanzen
        h = A @ Z                                          # [1, d]
        return h.squeeze(0), A.squeeze(0)                  # [d], [n]


# ---------------------------------------------------------------------------
# 3b. Multi-Scale-Aggregator (Mourao Gl. 7) + Gesamtmodell
# ---------------------------------------------------------------------------
class FPNMIL(nn.Module):
    """Vollständiges FPN-MIL-Modell nach Mourao et al.

    Args:
        in_channels:   Eingangskanäle der Patches.
        base_channels: Basis-Kanalzahl des Backbones.
        num_scales:    Anzahl Skalen S (Backbone-Stufen = FPN-Level).
        d_x:           gemeinsame FPN-Kanalzahl (Instanz-Feature-Dim).
        d:             Embedding-Dim der Aggregatoren.
        L:             Attention-Pooling-Dim.
        kernel_size:   Conv-Kernelgröße des Backbones.
        gated:         Gated AbMIL (True, wie in Mourao Gl. 3) oder simpel.
        blocks_per_stage: Conv-Blöcke je Backbone-Stufe.

    Forward-Rückgabe:
        P_ms            : finale Bag-Wahrscheinlichkeit          [1]
        Y_hat           : binäre Vorhersage (>= 0.5)             [1]
        scale_probs     : Liste der skalenspezifischen P^s       S x [1]
        scale_scores    : Multi-Scale-Attention a^s             [S]
        inst_attention  : Liste per-Instanz-Attention je Skala,
                          jeweils reshaped zu [K, h_s, w_s] (für Heatmaps)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_scales: int = 3,
        d_x: int = 256,
        d: int = 256,
        L: int = 128,
        kernel_size: int = 5,
        gated: bool = True,
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        self.num_scales = num_scales

        # (1) Bottom-up Backbone
        self.backbone = BottomUpBackbone(
            in_channels, base_channels, num_scales, kernel_size, blocks_per_stage
        )
        # (2) FPN-Neck
        self.fpn = FeaturePyramid(self.backbone.out_channels, d_x)

        # (3a) skalenspezifische Aggregatoren A^s + Klassifikationsköpfe C^s
        self.aggregators = nn.ModuleList(
            [GatedAbMIL(d_x, d, L, gated) for _ in range(num_scales)]
        )
        self.scale_heads = nn.ModuleList(
            [nn.Linear(d, 1) for _ in range(num_scales)]  # Logits -> BCEWithLogits
        )

        # (3b) Multi-Scale-Aggregator M (gated AbMIL über Skalen-Embeddings)
        self.ms_att_V = nn.Sequential(nn.Linear(d, L), nn.Tanh())
        self.ms_att_U = nn.Sequential(nn.Linear(d, L), nn.Sigmoid())
        self.ms_att_w = nn.Linear(L, 1)
        self.ms_head = nn.Linear(d, 1)  # Logit für P_ms

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor):
        if x.dim() == 5:          # [1, K, C, Hp, Wp]
            x = x.squeeze(0)      # -> [K, C, Hp, Wp]

        feats = self.backbone(x)          # [F^1 ... F^S]
        pyramid = self.fpn(feats)         # [R^1 ... R^S], R^s: [K, d_x, h_s, w_s]

        scale_embeddings = []
        scale_logits = []
        inst_attention = []
        for s, R in enumerate(pyramid):
            k, dx, h_s, w_s = R.shape
            # Flatten: jeder Feature-Map-Pixel über alle K Patches = 1 Instanz
            X = R.permute(0, 2, 3, 1).reshape(-1, dx)   # [K*h_s*w_s, dx]
            h_emb, a = self.aggregators[s](X)           # [d], [K*h_s*w_s]
            scale_embeddings.append(h_emb)
            scale_logits.append(self.scale_heads[s](h_emb))      # [1]
            inst_attention.append(a.view(k, h_s, w_s))           # für Heatmaps

        H = torch.stack(scale_embeddings, dim=0)        # [S, d]

        # Multi-Scale gated Attention (Gl. 7)
        A_ms = self.ms_att_w(self.ms_att_V(H) * self.ms_att_U(H))  # [S, 1]
        A_ms = torch.softmax(A_ms.transpose(0, 1), dim=1)          # [1, S]
        h_ms = (A_ms @ H).squeeze(0)                               # [d]

        ms_logit = self.ms_head(h_ms)                              # [1]
        P_ms = torch.sigmoid(ms_logit)
        Y_hat = torch.ge(P_ms, 0.5).float()

        scale_probs = [torch.sigmoid(l) for l in scale_logits]

        # Per-Patch-Attention [1, K] für Counting / Patch-Level-AUC:
        # Die Pixel-Attention jeder Skala wird über die Patch-Fläche summiert
        # (Attention-Masse pro Patch) und die Skalen werden mit den Multi-Scale-
        # Scores gewichtet -- entspricht der aggregierten Heatmap aus Mourao et al.
        # Ergebnis summiert zu 1 über die K Patches, analog zur Patch-Attention der
        # übrigen model.py-Modelle (Schwellwert 1/K funktioniert identisch).
        K = inst_attention[0].shape[0]
        scale_scores = A_ms.squeeze(0)                             # [S], Summe = 1
        A_patch = x.new_zeros(K)
        for s, a in enumerate(inst_attention):
            A_patch = A_patch + scale_scores[s] * a.reshape(K, -1).sum(dim=1)
        A_patch = A_patch.unsqueeze(0)                             # [1, K], Summe = 1

        # Reichhaltige Ausgaben für Loss/Visualisierung zwischenspeichern.
        self._cache = {
            "ms_logit": ms_logit,
            "scale_logits": scale_logits,
            "scale_probs": scale_probs,
            "scale_scores": scale_scores,
            "inst_attention": inst_attention,   # Pixel-Heatmaps je Skala [K, h_s, w_s]
            "patch_attention": A_patch,          # [1, K]
        }
        return P_ms, Y_hat, A_patch

    # -- Loss (klassen-gewichtete BCE, multi-scale + scale-specific) -------
    def calculate_objective(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        pos_weight: torch.Tensor = None,
        scale_loss_weight: float = 1.0,
    ):
        """Kombinierte Verlustfunktion analog Mourao et al.

        L = BCE(P_ms, Y) + scale_loss_weight * mean_s BCE(P^s, Y)

        Args:
            X: Eingabe-Bag.
            Y: Bag-Label (0/1).
            pos_weight: optionales Gewicht der positiven Klasse (Tensor [1]).
            scale_loss_weight: Gewicht der Deep-Supervision-Terme.

        Returns:
            loss, A_finest   (kompatibel zur `calculate_objective`-Konvention
            der übrigen model.py-Modelle; reichhaltige Ausgaben liegen in
            self._cache)
        """
        Y = Y.float().view(1)
        _, _, A_patch = self.forward(X)
        ms_logit = self._cache["ms_logit"].view(1)
        scale_logits = [l.view(1) for l in self._cache["scale_logits"]]

        bce = lambda logit: F.binary_cross_entropy_with_logits(
            logit, Y, pos_weight=pos_weight
        )

        ms_loss = bce(ms_logit)
        scale_loss = torch.stack([bce(l) for l in scale_logits]).mean()
        loss = ms_loss + scale_loss_weight * scale_loss
        return loss, A_patch

    def calculate_classification_error(self, X: torch.Tensor, Y: torch.Tensor):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def count_positive_instances(self, X: torch.Tensor, threshold: float = None):
        """Zählt Patches mit aggregierter Per-Patch-Attention über einem Schwellwert.

        A ist die multi-skalige Per-Patch-Attention [1, K] (summiert zu 1), sodass
        der Schwellwert 1/K -- wie bei den übrigen model.py-Modellen -- Patches mit
        überdurchschnittlicher Attention zählt (vergleichbar mit Ground-Truth `count`).

        Returns:
            count, A, threshold
        """
        _, _, A = self.forward(X)
        n = A.shape[1]  # = K (Anzahl Patches)
        if threshold is None:
            threshold = 1.0 / n  # über-uniformer Schwellwert
        count = int((A.squeeze(0) > threshold).sum().item())
        return count, A, threshold


# ---------------------------------------------------------------------------
# Smoke-Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Beispiel: ein Bag mit K=12 Graustufen-Patches der Größe 64x64
    K, C, Hp, Wp = 12, 1, 64, 64
    bag = torch.randn(1, K, C, Hp, Wp)
    label = torch.tensor([1.0])

    model = FPNMIL(
        in_channels=C,
        base_channels=32,
        num_scales=3,
        d_x=256,
        d=256,
        L=128,
        kernel_size=5,
        gated=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter: {n_params:,}")

    P_ms, Y_hat, A_finest = model(bag)
    scale_probs = model._cache["scale_probs"]
    scale_scores = model._cache["scale_scores"]
    inst_att = model._cache["inst_attention"]
    print(f"P_ms            : {P_ms.item():.4f}   Y_hat: {Y_hat.item()}")
    print(f"Per-Patch Attn A: {tuple(A_finest.shape)}  (Summe {A_finest.sum().item():.3f})")
    print(f"Skalen-Probs P^s: {[round(p.item(), 4) for p in scale_probs]}")
    print(f"Multi-Scale a^s : {[round(a.item(), 4) for a in scale_scores]}  "
          f"(Summe {scale_scores.sum().item():.3f})")
    print("Instanz-Attention-Shapes je Skala (K, h_s, w_s):")
    for s, a in enumerate(inst_att):
        print(f"  R^{s+1}: {tuple(a.shape)}")

    # Backward-Test
    pos_w = torch.tensor([2.0])
    loss, *_ = model.calculate_objective(bag, label, pos_weight=pos_w)
    loss.backward()
    print(f"\nLoss: {loss.item():.4f}  -> backward OK")
    grad_ok = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    print(f"Alle Parameter haben Gradienten: {grad_ok}")