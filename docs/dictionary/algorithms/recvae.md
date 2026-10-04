# RecVAE

> MultVAE with better training: a deeper encoder, a smarter "prior" that keeps each epoch close to the last,
> a regularisation strength that grows with how much a user has done, and alternating encoder and decoder
> updates.

!!! abstract "In plain words"
    RecVAE is [MultVAE](multvae.md) with better training habits:

    - a deeper network to read the history;
    - a rule that keeps each user's code close to what it was in the previous epoch, so learning is steadier;
    - regularisation scaled to how active a user is;
    - training the reading half and the rebuilding half of the network in turns.

--8<-- "generated/methods/recvae.md"

!!! tip "When to use it"
    - When MultVAE does well and you want a stronger autoencoder at a similar cost.
    - On dense, long histories (movies, music), where autoencoders shine.

!!! warning "When not to"
    - When order matters, for cold items, or with very short histories.

## 1. Intuition

!!! info "Words used on this page"
    - **Code:** the short vector the encoder makes from a user's history.
    - **Prior:** the model's default guess of what codes look like, before seeing the user.
    - **KL term:** a penalty that grows the further a user's code moves from the prior.
    - **Swish:** a smooth activation function, $x \cdot \sigma(x)$, used instead of ReLU.

RecVAE keeps MultVAE's idea (encode the history, decode a softmax over items) and changes four things:

1. **A deeper encoder.** Five layers with layer normalisation and the swish activation. Each layer also sees
   the sum of all earlier layers' outputs (dense residual connections), which makes deep training stable.
2. **A composite prior.** The KL term pulls each user's code towards a mixture of three distributions:
   - a standard normal (weight 3/20);
   - what the encoder said about the same user in the previous epoch (weight 3/4);
   - a very wide normal (weight 1/10).

   The middle part stops the codes from jumping around between epochs.
3. **A KL weight that grows with activity.** For a user with $n$ interactions, the KL weight is $\gamma n$.
   The reconstruction term grows with $n$ too, so this keeps the two balanced for light and heavy users.
4. **Alternating updates.** Each epoch trains the encoder for three passes (with input dropout), copies the
   encoder into the prior, then trains the decoder for one pass (without dropout).

## 2. A tiny worked example

With γ = 0.005:

| User's interactions $n$ | KL weight $\gamma n$ |
|---|---|
| 10 | 0.05 |
| 100 | 0.5 |
| 400 | 2.0 |

A user with 10 interactions is regularised lightly: their few items carry most of the information. A
heavy user's reconstruction term is about 40 times larger, so their KL weight is scaled up to match, instead
of using one β for everybody as MultVAE does.

**The composite prior in one dimension.** A user's code is now 2.0; last epoch it was 1.8. With unit-variance
normals, the KL penalty against a component centred at $m$ is $(2.0 - m)^2 / 2$:

| Prior component (weight) | Centre | Penalty |
|---|---|---|
| standard normal (3/20) | 0 | (2.0 − 0)² / 2 = **2.0** |
| previous epoch (3/4) | 1.8 | (2.0 − 1.8)² / 2 = **0.02** |

The mixture is dominated by its heaviest, closest part, so this code is cheap: it may sit far from zero as long
as it moved little since the last epoch. A code that jumped to 2.0 from −1.0 would pay heavily under both parts.

## 3. How it works

```mermaid
flowchart LR
    X[history vector] --> E[5-layer dense-residual encoder]
    E --> Z[code]
    Z --> D[linear decoder -> softmax over items]
    E -. after the encoder passes .-> O[old encoder = prior]
    O --> K[KL to a mixture: N(0,I), old posterior, wide N]
```

Per epoch: 3 encoder passes (dropout 0.5) → update the prior → 1 decoder pass (no dropout). recbench then
scores the validation fold, keeps the best epoch, and stops early.

## 4. The math, symbol by symbol

$$
\mathcal{L}(x_u) = -\sum_i x_{ui}\log \pi_i(z_u) + \gamma\,|x_u|\;\Big(\log q(z_u \mid x_u) - \log p(z_u)\Big),
\quad p(z) = \tfrac{3}{20}\mathcal N(0, I) + \tfrac{3}{4} q_{\text{old}}(z \mid x_u) + \tfrac{1}{10}\mathcal N(0, e^{10} I)
$$

| Symbol | Meaning |
|---|---|
| $\pi(z_u)$ | the decoder's softmax over items |
| $q(z_u \mid x_u)$ | the encoder's Gaussian (sampled during training) |
| $q_{\text{old}}$ | the encoder as it was after the previous epoch's encoder passes |
| $\gamma$ | KL scale per interaction (`recvae_gamma`) |
| $\lvert x_u\rvert$ | the user's number of interactions (sum of the vector) |

## 5. Training and inference

- **Training:** about four passes over the users per epoch, each a deep encoder plus a linear decoder.
- **Inference:** encoder mean → decoder → scores (fold-in).
- **Hardware:** GPU recommended.

## 6. Hyperparameters

| Name in recbench config | What it does | Default | Searched over |
|---|---|---|---|
| `recvae_gamma` | KL weight per interaction | 0.005 | 0.001–0.05 |
| `vae_hidden`, `vae_latent` | layer sizes | 600, 200 | 200–1000, 64–256 |
| `vae_dropout` | input dropout in encoder passes | 0.5 | 0.3–0.7 |
| `recvae_enc_epochs`, `recvae_dec_epochs` | passes per epoch | 3, 1 | fixed |
| `lr` | Adam learning rate (both optimisers) | 1e-3 from the preset (the paper uses 5e-4) | 1e-4–5e-3 |
| `max_epochs`, `patience` | early stopping on the validation fold; each epoch is 3 encoder passes and 1 decoder pass | 30, 3 | fixed: 50, 5 (the paper's 50 epochs) |

## 7. In recbench

- Code: `src/recbench/methods/vae.py::RecVAE`. The alternating schedule runs inside
  `src/recbench/methods/_torch.py::early_stopping_loop`.
- `tests/test_new_methods.py` checks that it clearly beats Random on toy data.

!!! info "Fidelity: faithful"
    It follows the authors' code (Apache-2.0): the encoder, the composite prior and its mixture weights, the
    γ-scaled KL, and the alternating optimisers. Their optional "implicit SLIM" extension is not included.

## 8. Results in this benchmark

--8<-- "generated/methods/recvae-results.md"

## 9. Strengths and weaknesses

- **Strengths:** usually stronger than MultVAE for little extra cost; stable training; fold-in scoring.
- **Weaknesses:** more moving parts; parameters grow with the catalog; ignores order.

## 10. Common pitfalls

- **Comparing RecVAE and MultVAE with different tuning budgets.** The quick tier gives both the same budget.
- **Dropout in decoder passes:** the authors switch it off there, and so does recbench.
- **Comparing it with MultVAE at the same number of epochs.** Each RecVAE epoch is four passes over the users
  (three for the encoder, one for the decoder), so the same epoch count costs RecVAE about four times more.

## 11. Check your understanding

??? question "Why does the prior include the previous epoch's posterior?"
    It keeps each user's code near where it was, which stabilises training, a bit like a moving target that
    changes slowly.

??? question "Why scale the KL weight by the number of interactions?"
    The reconstruction term sums over a user's items, so it is larger for heavy users. Scaling the KL term the
    same way keeps the trade-off similar for everyone.

??? question "What does the 'previous epoch' part of the prior do?"
    It pulls each user's code towards what the encoder said about that user one epoch earlier, so codes can drift
    away from zero, but only gradually. That steadies training without forcing every code towards the same point.

## 12. Further reading

- Shenbin, Alekseev, Tutubalina, Malykh, Nikolenko (2020). *RecVAE: a New Variational Autoencoder for Top-N
  Recommendations with Implicit Feedback.* WSDM. [arXiv](https://arxiv.org/abs/1912.11160)
- [MultVAE](multvae.md), [EASE](ease.md).
