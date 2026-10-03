"""Variational autoencoders for collaborative filtering: MultVAE (Liang et al. 2018) and RecVAE (Shenbin et al. 2020).

Both read a user's whole interaction vector, compress it to a small latent code, and decode it into a
probability for every item (a multinomial over the catalog). Because scoring only needs the history vector,
any user with a history can be scored ("fold-in"), with no per-user parameters.

- MultVAE: encoder [items -> 600 -> 200], decoder [200 -> 600 -> items], tanh, KL weight annealed to a cap.
- RecVAE: a deeper encoder with dense residual connections, layer norm and swish; a "composite prior" (a
  mixture of N(0, I), the previous epoch's posterior, and a wide normal); a KL weight that grows with the
  user's number of interactions (gamma); and alternating encoder / decoder updates.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recbench.data import HistoryBatch, TrainView
from recbench.methods._torch import autocast, early_stopping_loop, resolve_device
from recbench.protocol import NEG_INF, MethodSpec, Recommender, Task
from recbench.registry import register_method

VAE_TASKS = {Task.topn, Task.sequential, Task.similar_items}


class MultVAENet(nn.Module):
    def __init__(self, n_items: int, hidden: int, latent: int, dropout: float):
        super().__init__()
        self.enc = nn.Linear(n_items, hidden)
        self.enc_out = nn.Linear(hidden, 2 * latent)
        self.dec = nn.Linear(latent, hidden)
        self.dec_out = nn.Linear(hidden, n_items)
        self.drop = nn.Dropout(dropout)
        self.latent = latent

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.enc(self.drop(F.normalize(x, dim=-1))))
        mu, logvar = self.enc_out(h).split(self.latent, dim=-1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.dec_out(torch.tanh(self.dec(z))), mu, logvar


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _log_norm_pdf(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class RecVAEEncoder(nn.Module):
    """Five layers; each sees the sum of all previous layers' outputs (dense residuals), as in the authors' code."""

    def __init__(self, n_items: int, hidden: int, latent: int, eps: float = 0.1):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(n_items, hidden)] + [nn.Linear(hidden, hidden) for _ in range(4)])
        self.ln = nn.ModuleList([nn.LayerNorm(hidden, eps=eps) for _ in range(5)])
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor, dropout: float) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.dropout(F.normalize(x, dim=-1), p=dropout, training=self.training)
        outputs = [self.ln[0](_swish(self.fc[0](x)))]
        for layer in range(1, 5):
            outputs.append(self.ln[layer](_swish(self.fc[layer](outputs[-1]) + sum(outputs))))
        return self.mu(outputs[-1]), self.logvar(outputs[-1])


class RecVAENet(nn.Module):
    MIXTURE = (3 / 20, 3 / 4, 1 / 10)  # standard normal, previous posterior, wide ("uniform") normal

    def __init__(self, n_items: int, hidden: int, latent: int):
        super().__init__()
        self.encoder = RecVAEEncoder(n_items, hidden, latent)
        self.encoder_old = RecVAEEncoder(n_items, hidden, latent)
        self.encoder_old.requires_grad_(False)
        self.decoder = nn.Linear(latent, n_items)

    def update_prior(self) -> None:
        self.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))

    def log_prior(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        post_mu, post_logvar = self.encoder_old(x, 0.0)
        zeros = torch.zeros_like(z)
        parts = [_log_norm_pdf(z, zeros, zeros), _log_norm_pdf(z, post_mu, post_logvar), _log_norm_pdf(z, zeros, zeros + 10.0)]
        stacked = torch.stack([p + float(np.log(w)) for p, w in zip(parts, self.MIXTURE)], dim=-1)
        return torch.logsumexp(stacked, dim=-1)

    def loss(self, x: torch.Tensor, gamma: float, dropout: float) -> torch.Tensor:
        mu, logvar = self.encoder(x, dropout)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        mll = (F.log_softmax(self.decoder(z), dim=-1) * x).sum(dim=-1).mean()
        kl_weight = gamma * x.sum(dim=-1)  # the KL weight grows with the user's number of interactions
        kld = ((_log_norm_pdf(z, mu, logvar) - self.log_prior(x, z)).sum(dim=-1) * kl_weight).mean()
        return -(mll - kld)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x, 0.0)
        return self.decoder(mu)


class _VAEBase(Recommender):
    """Shared data handling: dense user vectors on the device, batches of users, scoring by fold-in."""

    def _prepare(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.device = resolve_device(cfg)
        self.seen = data.weighted_matrix(cfg.get("decay_half_life_days")).astype(np.float32).tocsr()
        self.train_users = data.warm_users()
        self.batch_size = int(cfg.get("vae_batch_size", 500))
        self.rng = np.random.default_rng(int(cfg.get("seed", 42)))

    def _rows(self, users: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.seen[users].toarray(), device=self.device)

    def _epoch_batches(self):
        order = self.rng.permutation(self.train_users)
        for start in range(0, len(order), self.batch_size):
            yield self._rows(order[start : start + self.batch_size])

    @torch.no_grad()
    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        self.net.eval()
        out = self._score(self._rows(users)).float().cpu().numpy()
        out[:, 0] = NEG_INF
        return out


@register_method
class MultVAE(_VAEBase):
    spec = MethodSpec(
        name="multvae",
        tasks=VAE_TASKS,
        uses_history=True,
        needs_torch=True,
        upstream="in-repo PyTorch, following Liang et al. 2018 (Mult-VAE^PR)",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self._prepare(data, cfg)
        self.net = MultVAENet(self.n_items + 1, int(cfg.get("vae_hidden", 600)), int(cfg.get("vae_latent", 200)),
                              float(cfg.get("vae_dropout", 0.5))).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("vae_l2", 0.0)))
        cap = float(cfg.get("vae_beta_cap", 0.2))
        steps_per_epoch = max(1, int(np.ceil(len(self.train_users) / self.batch_size)))
        anneal_steps = max(1, int(cfg.get("vae_anneal_epochs", 10)) * steps_per_epoch)
        state = {"step": 0}

        def run_epoch(_: int) -> float:
            total, count = 0.0, 0
            for x in self._epoch_batches():
                beta = cap * min(1.0, state["step"] / anneal_steps)  # KL annealing from 0 to the cap
                with autocast(self.device, cfg):
                    logits, mu, logvar = self.net(x)
                    nll = -(F.log_softmax(logits.float(), dim=-1) * x).sum(dim=-1).mean()
                    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()
                    loss = nll + beta * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                state["step"] += 1
                total, count = total + float(loss.detach()), count + 1
            return total / max(count, 1)

        self.fit_info = early_stopping_loop(self.net, run_epoch, cfg, owner=self)

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.net(x)
        return logits


@register_method
class RecVAE(_VAEBase):
    spec = MethodSpec(
        name="recvae",
        tasks=VAE_TASKS,
        uses_history=True,
        needs_torch=True,
        upstream="in-repo PyTorch, following the authors' code (ilya-shenbin/RecVAE, Apache-2.0)",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self._prepare(data, cfg)
        self.net = RecVAENet(self.n_items + 1, int(cfg.get("vae_hidden", 600)), int(cfg.get("vae_latent", 200))).to(self.device)
        lr = float(cfg.get("lr", 5e-4))
        opt_enc = torch.optim.Adam(self.net.encoder.parameters(), lr=lr)
        opt_dec = torch.optim.Adam(self.net.decoder.parameters(), lr=lr)
        gamma = float(cfg.get("recvae_gamma", 0.005))
        enc_passes, dec_passes = int(cfg.get("recvae_enc_epochs", 3)), int(cfg.get("recvae_dec_epochs", 1))
        dropout = float(cfg.get("vae_dropout", 0.5))

        def run_pass(optimizer, drop: float) -> float:
            total, count = 0.0, 0
            for x in self._epoch_batches():
                with autocast(self.device, cfg):
                    loss = self.net.loss(x, gamma, drop)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total, count = total + float(loss.detach()), count + 1
            return total / max(count, 1)

        def run_epoch(_: int) -> float:  # the authors' alternating schedule
            for _ in range(enc_passes):
                loss = run_pass(opt_enc, dropout)
            self.net.update_prior()
            for _ in range(dec_passes):
                loss = run_pass(opt_dec, 0.0)
            return loss

        self.fit_info = early_stopping_loop(self.net, run_epoch, cfg, owner=self)

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.scores(x)
