import numpy as np
import torch


# -------- Bernoulli diffusion noise scheduler --------
class BernoulliNoiseScheduler(torch.nn.Module):
    def __init__(self, steps=40, beta_type='linear_cumprod', pi=0.5, print_info=False):
        super().__init__()
        self.pi = pi
        if beta_type == 'linear_cumprod':
            alpha = 1 - 1 / (steps - np.arange(1, steps+1) + 1)
            alpha_cumprod = np.cumprod(alpha, 0)

        elif beta_type == 'linear':
            scale = 1000 / steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            alpha = 1 - np.linspace(beta_start, beta_end, steps)
            alpha_cumprod = np.cumprod(alpha, 0)

        elif beta_type == 'cos':
            alpha_cumprod = np.linspace(0.0, 1.0, steps+1)
            alpha_cumprod = alpha_cumprod * np.pi
            alpha_cumprod = 0.5 + 0.5 * np.cos(alpha_cumprod)
            alpha = (alpha_cumprod[1:] / alpha_cumprod[:-1])
        
        elif beta_type == 'sigmoid':
            
            def sigmoid(x):
                z = 1/(1 + np.exp(-x))
                return z

            def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=0.0):
                # A gamma function based on sigmoid function.
                v_start = sigmoid(start / tau)
                v_end = sigmoid(end / tau)
                output = sigmoid((t * (end - start) + start) / tau)
                output = (v_end - output) / (v_end - v_start)
                return np.clip(output, clip_min, 1.)

            alpha_cumprod = np.linspace(0.0, 1.0, steps+1)
            alpha_cumprod = sigmoid_schedule(alpha_cumprod, 0, 3, 0.8)
            alpha = (alpha_cumprod[1:] / alpha_cumprod[:-1])

        else:
            raise NotImplementedError

        alpha = np.hstack([1, alpha])
        alpha_cumprod = np.hstack([1, alpha_cumprod])
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('alpha_cumprod', torch.tensor(alpha_cumprod))

        print(f'Noise scheduler with {beta_type}:')

        if print_info:
            print(f'Diffusion 1.0 -> {pi}:')
            data = (1.0 * self.alpha_cumprod + pi * (1 - self.alpha_cumprod)).data.numpy()
            print(' '.join([f'{d:0.4f}' for d in data]))

            print(f'Diffusion 0.0 -> {pi}:')
            data = (0.0 * self.alpha_cumprod + pi * (1 - self.alpha_cumprod)).data.numpy()
            print(' '.join([f'{d:0.4f}' for d in data]))

            print(f'Alpha:')
            print(' '.join([f'{d:0.4f}' for d in self.alpha.data.numpy()]))

    def sample_from_prob(self, prob):
        out = torch.bernoulli(prob).triu(diagonal=1)
        out = out.transpose(-1, -2) + out
        return out

    def one_step(self, x, t):
        dim = x.ndim - 1
        a = self.alpha[t].view(-1, *([1]*dim))
        x = x * a + self.pi * (1 - a)
        return x

    def forward(self, x, t):
        dim = x.ndim - 1
        a_bar = self.alpha_cumprod[t].view(-1, *([1]*dim))
        x = x * a_bar + self.pi * (1 - a_bar)
        return x


# -------- discrete diffusion noise schedulers --------
# all returning alphas
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return 1 - torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64).numpy()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas

def binary_alpha_schedule(timesteps, type='cosine', s=0.008):
    if type == 'cosine':
        return cosine_beta_schedule(timesteps, s)
    elif type == 'linear':
        return linear_beta_schedule(timesteps)
    else:
        raise NotImplementedError(f'Unsupported schedule type {type}')