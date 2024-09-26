import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, WavLMForXVector, AutoModel
import torchaudio




class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=fft_size, win_length=win_length, hop_length=shift_size, window_fn=window)

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)    
        return sc_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window=torch.hann_window):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss
    
    
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

""" https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """
def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

class GeneratorLoss(torch.nn.Module):

    def __init__(self, md):
        """Initilize spectral convergence loss module."""
        super(GeneratorLoss, self).__init__()
        self.md = md
        
    def forward(self, y, y_hat):
        d_fake = self.md(y_hat)
        d_real = self.md(y)
        
        loss_g = 0
        loss_rel = 0

        for x_fake, x_real in zip(d_fake, d_real):
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)
#             loss_rel += generator_TPRLS_loss([x_real[-1]], [x_fake[-1]])

        loss_feature = 0
        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

        
        loss_gen_all = loss_g + loss_feature + loss_rel
        
        return loss_gen_all.mean()
    
class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, md):
        """Initilize spectral convergence loss module."""
        super(DiscriminatorLoss, self).__init__()
        self.md = md
        
    def forward(self, y, y_hat):
        d_fake = self.md(y_hat)
        d_real = self.md(y)
        loss_d = 0
        loss_rel = 0
        
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        
#             loss_rel += discriminator_TPRLS_loss([x_real[-1]], [x_fake[-1]])


        d_loss = loss_d + loss_rel
        
        return d_loss.mean()
    
    
    

    
class WavLMLoss(torch.nn.Module):

    def __init__(self, mwd):
        """Initilize spectral convergence loss module."""
        super(WavLMLoss, self).__init__()
        self.wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus-sv').to('cuda')
        self.mwd = mwd
        self.resample = torchaudio.transforms.Resample(24000, 16000)
        
    def forward(self, wav, y_rec, text, generator_turn=False, discriminator_turn=False):
        assert generator_turn or discriminator_turn
        
        if generator_turn:
            return self.generator(wav, y_rec, text)
        if discriminator_turn:
            return self.discriminator(wav, y_rec, text)
        
    def generator(self, wav, y_rec, text, adv=True):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))
    
        if adv:
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

            with torch.no_grad():
                y_d_rs, r_hidden = self.mwd(y_embeddings, text, 
                         torch.ones(y_embeddings.size(0)).to(text.device).long() * y_embeddings.size(-1),
                        y_embeddings.size(-1))

            y_d_gs, f_hidden = self.mwd(y_rec_embeddings, text, 
                         torch.ones(y_rec_embeddings.size(0)).to(text.device).long() * y_rec_embeddings.size(-1),
                        y_rec_embeddings.size(-1))

            y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

            loss_fm = 0
            for r, f in zip(r_hidden, f_hidden):
                loss_fm = F.l1_loss(r, f)

            loss_gen_f = torch.mean((1-y_df_hat_g)**2)
    #         loss_rel = generator_TPRLS_loss([y_df_hat_r], [y_df_hat_g])
            loss_rel = 0

            loss_gen_all = loss_gen_f + loss_rel + loss_fm
        else:
            loss_gen_all = torch.zeros(1).to(wav.device)
            
        return loss_gen_all, floss.mean()
    
    def discriminator(self, wav, y_rec, text):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs, _ = self.mwd(y_embeddings, text, 
                     torch.ones(y_embeddings.size(0)).to(text.device).long() * y_embeddings.size(-1),
                    y_embeddings.size(-1))
        
        y_d_gs, _ = self.mwd(y_rec_embeddings, text, 
                     torch.ones(y_rec_embeddings.size(0)).to(text.device).long() * y_rec_embeddings.size(-1),
                    y_rec_embeddings.size(-1))
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)
        
        loss_disc_f = r_loss + g_loss
        
#         loss_rel = discriminator_TPRLS_loss([y_df_hat_r], [y_df_hat_g])
        loss_rel = 0
    
        d_loss = loss_disc_f + loss_rel
        
        return d_loss.mean()
    
    
    # for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

# for F0 consistency loss
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean

def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


# This needs to be changed to the Pyannote, but just to have the code working

class SVLoss(nn.Module): 
    def __init__(self):
        """Initialize spectral convergence loss module."""
        super(SVLoss, self).__init__()
        self.resample = torchaudio.transforms.Resample(24000, 16000)
        self.sv_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
       
    def forward(self, y, y_hat):
    
        y = y.squeeze(1) if y.dim() == 4 else y
        y_hat = y_hat.squeeze(1) if y_hat.dim() == 4 else y_hat
        

        if y.size(1) > 1:
            y = y.mean(dim=1, keepdim=True)
        if y_hat.size(1) > 1:
            y_hat = y_hat.mean(dim=1, keepdim=True)

        y = self.resample(y.squeeze(1))
        y_hat = self.resample(y_hat.squeeze(1))
        

        y_np = y.detach().cpu().numpy()
        y_hat_np = y_hat.detach().cpu().numpy()
        

        h_real = self.feature_extractor(y_np, sampling_rate=16000, padding=True, return_tensors="pt")
        h_fake = self.feature_extractor(y_hat_np, sampling_rate=16000, padding=True, return_tensors="pt")

        h_real = {k: v.to(y.device) for k, v in h_real.items()}
        h_fake = {k: v.to(y_hat.device) for k, v in h_fake.items()}
        
        with torch.no_grad():
            emb_real = self.sv_model(**h_real).embeddings
        emb_fake = self.sv_model(**h_fake).embeddings
        

        emb_real = F.normalize(emb_real, dim=-1)
        emb_fake = F.normalize(emb_fake, dim=-1)
        
        # Compute losses
        loss_feat = F.l1_loss(h_fake['input_values'], h_real['input_values'])
        loss_sim = 1 - F.cosine_similarity(emb_fake, emb_real, dim=-1).mean()
       
        return loss_feat, loss_sim
