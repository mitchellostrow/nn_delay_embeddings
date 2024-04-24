import torch.nn as nn
import numpy as np
from einops import einsum, repeat
import math
import torch


class S4D_rnn(nn.Module):
    # this computes an s4d model
    def __init__(
        self,
        input_dim,
        n_states_per_input,
        dt_min=1e-3,
        dt_max=1e-1,
        seed=None,
        lr=None,
        noC=False,
    ):
        super().__init__()

        self.noC = noC
        if seed is not None:
            torch.manual_seed(seed)

        log_dt = torch.rand(input_dim) * (math.log(dt_max) - math.log(dt_min)) + np.log(
            dt_min
        )
        self.register("log_dt", log_dt, lr)

        # s4d-lin intialization
        log_A_real = torch.log(0.5 * torch.ones(input_dim, n_states_per_input // 2))
        A_imag = math.pi * repeat(
            torch.arange(n_states_per_input // 2), "n -> h n", h=input_dim
        )

        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        B = torch.ones((input_dim, n_states_per_input)) + 0j
        self.register("B", B, lr)

        C = torch.rand(input_dim, n_states_per_input // 2) + 1j * torch.rand(
            input_dim, n_states_per_input // 2
        )
        self.register("C", C, lr)

    def discretize(self, log_dt, log_A_real, A_imag, B):

        dt, A = torch.exp(log_dt), -torch.exp(log_A_real) + 1j * A_imag

        dtA = einsum(dt, A, "n, n h -> n h")
        dtB = einsum(dt, B, "n, n h -> n h")

        dA = (1 + dtA / 2) / (1 - dtA / 2)  # bilinear
        dB = dtB / (1 - dtA / 2)  # bilinear

        return dA, dB

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, us, rnn=True):
        # num_dimensions is the number of dimensions in the input
        # num_states is the number of dimensions of the SSM per each input
        # us has input (num_dimensions, time (L)) or (batch, num_dim, time)
        # dA has shape (num_dimensions, num_states) (diagonal)
        # dB has shape (num_dimensions, num_states)
        # C has shape (num_dimensions, num_states, 1)
        if len(us.shape) == 2:
            us = us.unsqueeze(0)

        b, nd, L = us.shape
        device = us.device
        # expand A_imag and log_A_real to include complex conjugates (do here so weights are tied)
        A_imag = torch.cat([self.A_imag, -self.A_imag], dim=1)
        # double log_A_real to match
        log_A_real = torch.cat([self.log_A_real, self.log_A_real], dim=1)

        dA, dB = self.discretize(self.log_dt, log_A_real, A_imag, self.B)

        nstates = dA.shape[-1]
        L = us.shape[-1]

        if rnn:
            C = torch.cat([self.C, self.C], dim=1)
            # xs = torch.zeros((b, nd, nstates, L),dtype=torch.complex64) #num_dimensions, num_states, time
            xs = torch.zeros((b, nd, nstates), dtype=torch.complex64).to(
                device
            )  # num_dimensions, num_states, time
            all_xs = []
            if self.noC:
                ys = torch.zeros((b, nd, nstates // 2, L), dtype=torch.complex64).to(
                    device
                )
            else:
                ys = torch.zeros((b, nd, L), dtype=torch.complex64).to(
                    device
                )  # num_dimensions, time)
            inp = einsum(dB, us, "d n, b d t -> b d n t")
            for i in range(L):
                xs = einsum(dA, xs, "d n, b d n -> b d n") + inp[:, :, :, i]
                all_xs.append(torch.clone(xs))

                if self.noC:
                    ys[:, :, :, i] = xs[
                        :, :, : nstates // 2
                    ]  # only take the first half, second half is repeated with conj pairs
                else:
                    ys[:, :, i] = einsum(C, xs, "d n, b d n -> b d")

            all_xs = (
                torch.stack(all_xs, dim=-1).transpose(-1, -2).transpose(-2, -3)
            )  # (B, L, H, N)

            return all_xs, ys.real

        else:
            # compute the kernel, then convolve
            dt = torch.exp(self.log_dt)  # (H)
            A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

            # Vandermonde multiplication
            dtA = A * dt.unsqueeze(-1)  # (H N)
            K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)

            if self.noC:
                k_no_c = 2 * torch.exp(K).real
                ys = []

                u_f = torch.fft.rfft(us, n=2 * L)  # (B H L)
                for kk in range(k_no_c.shape[1]):
                    k_f = torch.fft.rfft(k_no_c[:, kk], n=2 * L)  # (H L)
                    y_k = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)
                    ys.append(y_k)

                y = torch.stack(ys, dim=1)

            else:
                C = self.C * (torch.exp(dtA) - 1.0) / A
                # no multiplying by 2 here because we plugged in the conjugate

                k = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real

                # Convolution
                k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
                u_f = torch.fft.rfft(us, n=2 * L)  # (B H L)
                y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

            return None, y


class S4DMinimal(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        d_state,
        dropout=0.0,
        prenorm=False,
        expansion=4,
        noC=False,
        seed=10,
    ):
        super().__init__()
        self.noC = noC
        self.prenorm = prenorm
        self.encoder = nn.Linear(input_dim, d_model)
        self.s4 = S4D_rnn(
            input_dim=d_model, n_states_per_input=d_state, noC=noC, seed=seed
        )
        if self.noC:  # if we return all A's (the delay embedded formulation)
            self.dm = d_state * d_model // 2  # dividing by 2 bc of conjugate pairs
        else:
            self.dm = d_model

        self.norm = nn.LayerNorm(self.dm) if not prenorm else nn.LayerNorm(d_model)

        # self.decoder = nn.Linear(dm,input_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.dm, expansion * self.dm),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * self.dm, input_dim),
        )

        self.rnn = False

    def eval(self):
        super().eval()  # do everything in the superclass, we love inheritance!
        self.rnn = True  # want to savve hidden states so do this
        # TODO: can we get the hidden states out of the cnn mode?
        # answer: only in noC

    def train(self, mode=True):  # super requires the mode argument
        super().train(mode)
        self.rnn = False

    def forward(self, x):
        u = self.encoder(x)  # (B, L, input_dim) -> (B, L, d_model)
        b, l, dinp = u.shape

        u = u.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        if self.prenorm:
            # Prenorm
            u = self.norm(u.transpose(-1, -2)).transpose(-1, -2)

        # Apply S4 block: we ignore the state input and output
        xs, ys = self.s4(u, rnn=self.rnn)
        if xs is not None:
            xs = xs.reshape(xs.shape[0], xs.shape[1], -1)
        self.ssm_states = xs
        self.ssm_outs = ys

        ys = ys.transpose(-1, -2)  # (B, d_model, L) -> (B, L, d_model)
        u = u.transpose(-1, -2)
        # Residual connection
        if self.noC:
            zs = ys.reshape(b, l, self.dm)
            # u won't be the right dimension, and besides, we already have this in y
            # also,flatten each ssm output
        else:
            zs = ys + u  # B, L, d_model

        if not self.prenorm:
            # Postnorm
            zs = self.norm(zs)

        # zs = zs.transpose(-1, -2) # B, d_model, L

        # Decode the outputs
        zs = self.decoder(zs)  # (B, L, d_model) -> (B, L, input_dim)

        return zs, xs
