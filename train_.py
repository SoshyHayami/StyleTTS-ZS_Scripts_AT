import os
import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
import click
import warnings
warnings.simplefilter('ignore')

# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa

from models import *
from meldataset import build_dataloader
from utils import *
from losses import *
from optimizers import build_optimizer
import time

# from accelerate import Accelerator
# from accelerate.utils import LoggerType
# from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, WavLMForXVector
import torchaudio

import logging
# from accelerate.logging import get_logger
# logger = get_logger(__name__, log_level="DEBUG")

import logging
from collections import deque
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from Modules.discriminators import * 
from Modules.discriminators_ZS import *
from Modules.hifigan import *


from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

from models import *
from losses import *

from utils import get_data_path_list
# load F0 model




@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)

def main(config_path):
    
    config = yaml.safe_load(open(config_path))
    batch_size = config.get('batch_size', 10)
    
    device = config.get('device', 'cuda')
    
    epochs = config.get('epoch', 100)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    
    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    
    train_path = data_params['train_data']
    val_path = data_params['val_data']

    min_length = data_params['min_length']

    
    max_len = config.get('max_len', 150)
    

    
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    F0_model = pitch_extractor
    
    
    F0_model = F0_model.to(device)
    F0_model_copy = copy.deepcopy(F0_model)
    
    
    print("Pitch Extractor Loaded...")


    # load ASR model

    ASR_MODEL_PATH = config.get('ASR_path', False)
    ASR_MODEL_CONFIG = config.get('ASR_config', False)


    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location='cpu')['model']
        model.load_state_dict(params)
        return model


    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()
    asr_model = asr_model.to('cuda')

    asr_model_copy = copy.deepcopy(asr_model)

    print("Text Alginer Loaded...")



    train_list, val_list = get_data_path_list(train_path, val_path)
  


    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                    batch_size=batch_size,
                                    validation=True,
                                    num_workers=2,
                                    device=device,
                                    dataset_config={})



    model = build_model(asr_model=asr_model, F0_model=F0_model)


    print('model loaded')



    # simple fix for dataparallel that allows access to class attributes
    class MyDataParallel(torch.nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
            
    for key in model:
        if key != "md" and key != "discriminator":
            model[key] = MyDataParallel(model[key])


    gl = GeneratorLoss(model.md).to('cuda')
    dl = DiscriminatorLoss(model.md).to('cuda')
    wl = WavLMLoss(model.discriminator).to('cuda')
    sv = SVLoss().to('cuda')
    stft_loss = MultiResolutionSTFTLoss().to('cuda')

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)
    sv = MyDataParallel(sv)




    scheduler_params = {
        "max_lr": 1e-4,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict= {key: scheduler_params.copy() for key in model}

    optimizer = build_optimizer({key: model[key].parameters() for key in model},lr=0.00002,
                                        scheduler_params_dict=scheduler_params_dict)


    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    loss_train_record = list([])
    loss_test_record = list([])
    iters = 0


    saving_steps = config.get('saving_steps', 250) # save every 250 steps, good for large datasets

    criterion = nn.L1Loss() # F0 loss (regression)
    torch.cuda.empty_cache()

    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('load_ckpt', False)
    
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the model at %s ...' % first_stage_path)
            model, optimizer, start_epoch, iters = load_checkpoint(model, 
                None, 
                first_stage_path,
                load_only_params=False)

            iters += iters
            epochs += start_epoch
        else:
            raise ValueError('You need to specify the path to the checkpoint.') 

    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True))

    print("iters->", iters)

    start_ds = True
    start_lm = True

    coll = 0


    logging.basicConfig(filename='training_log.txt', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    loss_history = {
        'running_loss': deque(maxlen=log_interval),
        'loss_sty': deque(maxlen=log_interval),
        'd_loss': deque(maxlen=log_interval),
        'loss_algn': deque(maxlen=log_interval),
        'loss_s2s': deque(maxlen=log_interval),
        'loss_F0_rec': deque(maxlen=log_interval),
        'loss_gen_all': deque(maxlen=log_interval),
        'loss_lm': deque(maxlen=log_interval),
        'loss_gen_lm': deque(maxlen=log_interval),
        'fsim_loss': deque(maxlen=log_interval),
        'sim_loss': deque(maxlen=log_interval)
    }

    for epoch in range(epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].train() for key in model]
        asr_model_copy = asr_model_copy.eval()
        F0_model_copy = F0_model_copy.eval()
        

        i = 0

        for i, batch in enumerate(train_dataloader):
            
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths = batch

            ### data preparation step
            libri = False
            
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** asr_model.n_down)).to('cuda')
                text_mask = length_to_mask(input_lengths).to(texts.device)

            ppgs, s2s_pred, s2s_attn = asr_model(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)
            
            with torch.no_grad():
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                attn_mask = (attn_mask < 1)

            s2s_attn.masked_fill_(attn_mask, 0.0)
            
            with torch.no_grad():
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** asr_model.n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)
            
            mel_len_st = int(adj_mels_lengths.min().item() / 2 - 1)
            mel_len = min([int(mel_input_length.min().item() / 2 - 1)])

            st = []
            gt = []
            
            for bib in range(len(adj_mels_lengths)):
                mel_length = int(adj_mels_lengths[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(adj_mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
            st = torch.stack(st).detach()
            gt = torch.stack(gt).detach()
            
            mmwd = False
            if random.random() < 1:
                st = gt
                mmwd = True
            
            s, t_en = model.style_encoder(st, t_en, input_lengths, t_en.size(-1))
            
            if random.random() < 0.2:
                with torch.no_grad():
                    s_null, t_en_null = model.style_encoder(torch.zeros_like(st).to('cuda'), t_en, input_lengths, t_en.size(-1))
                if bool(random.getrandbits(1)):
                    s = s_null
                else:
                    t_en = t_en_null

            if bool(random.getrandbits(1)):
                asr = (t_en @ s2s_attn)
            else:
                asr = (t_en @ s2s_attn_mono)

            # get clips
            mel_len = min([int(mel_input_length.min().item() / 2 - 1), max_len // 2])

            en = []
            gt = []
            wav = []
            
            rs = []
            
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                rs.append(random_start)
                
                en.append(asr[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
                if not libri:
                    y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                    wav.append(torch.from_numpy(y).to('cuda'))
                else:
                    y = waves.squeeze()[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                    wav.append((y).to('cuda'))

            en = torch.stack(en)
            gt = torch.stack(gt).detach()
            wav = torch.stack(wav).float().detach()

            with torch.no_grad():
                _, F0_gt, _ = F0_model_copy(gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                F0_real, _, _ = F0_model(gt.unsqueeze(1))

            y_rec = model.decoder(en, F0_real, real_norm, s)
            
            loss_F0_rec = 0
            loss_sty = 0

            if start_ds:
                optimizer.zero_grad()
                d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                if start_lm:
                    d_loss += wl(wav.detach().squeeze(), y_rec.detach().squeeze(), en.detach(), discriminator_turn=True).mean()
                d_loss.backward()
                optimizer.step('md')
                optimizer.step('discriminator')
            else:
                d_loss = 0
            
            # generator loss
            optimizer.zero_grad()
            loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= texts.size(0)

            loss_algn = F.l1_loss(s2s_attn, s2s_attn_mono)
            
            if start_ds:
                loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
            else:
                loss_gen_all = 0
                
            if start_ds:
                loss_gen_lm, loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze(), en.detach(), generator_turn=True)
                loss_gen_lm, loss_lm = loss_gen_lm.mean(), loss_lm.mean()
            else:
                loss_lm, loss_gen_lm = 0, 0
            
            sim_loss, fsim_loss = sv(wav.unsqueeze(1).detach(), y_rec.unsqueeze(1))
            
            sim_loss, fsim_loss = sim_loss.mean(), fsim_loss.mean()
            
            if start_ds:
                g_loss = loss_mel * 5 + loss_sty + loss_algn * 10 + loss_s2s + loss_F0_rec + loss_gen_all + loss_lm + loss_gen_lm + sim_loss * 5 + fsim_loss * 5
            else:
                g_loss = loss_mel
                
            running_loss += loss_mel.item()
            g_loss.backward()

            if torch.isnan(g_loss):
                from IPython.core.debugger import set_trace
                set_trace()
            optimizer.step('text_encoder')
            optimizer.step('style_encoder')
            optimizer.step('decoder')
            if start_ds:
                optimizer.step('text_aligner')
                optimizer.step('pitch_extractor')
            
            i += 1
            
            # Add losses to history
            loss_history['running_loss'].append(loss_mel.item())
            loss_history['loss_sty'].append(loss_sty)
            loss_history['d_loss'].append(d_loss)
            loss_history['loss_algn'].append(loss_algn.item())
            loss_history['loss_s2s'].append(loss_s2s.item())
            loss_history['loss_F0_rec'].append(loss_F0_rec)
            loss_history['loss_gen_all'].append(loss_gen_all)
            loss_history['loss_lm'].append(loss_lm)
            loss_history['loss_gen_lm'].append(loss_gen_lm)
            loss_history['fsim_loss'].append(fsim_loss.item())
            loss_history['sim_loss'].append(sim_loss.item())
            
            iters = iters + 1
            if (i+1) % log_interval == 0:
                # Calculate average losses
                avg_losses = {k: sum(v)/len(v) if v else 0 for k, v in loss_history.items()}
                
                log_message = (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_dataloader)//batch_size}], '
                            f'Avg Running Loss: {avg_losses["running_loss"]:.5f}, '
                            f'Avg Sty Loss: {avg_losses["loss_sty"]:.5f}, '
                            f'Avg Disc Loss: {avg_losses["d_loss"]:.5f}, '
                            f'Avg Algn Loss: {avg_losses["loss_algn"]:.5f}, '
                            f'Avg S2S Loss: {avg_losses["loss_s2s"]:.5f}, '
                            f'Avg F0 Loss: {avg_losses["loss_F0_rec"]:.5f}, '
                            f'Avg Gen Loss: {avg_losses["loss_gen_all"]:.5f}, '
                            f'Avg WavLM Loss: {avg_losses["loss_lm"]:.5f}, '
                            f'Avg GenLM Loss: {avg_losses["loss_gen_lm"]:.5f}, '
                            f'Avg SIM Loss: {avg_losses["sim_loss"]:.5f}, '
                            f'Avg FSim Loss: {avg_losses["fsim_loss"]:.5f}')
                
                logging.info(log_message)
                logging.info(f'Time elapsed: {time.time()-start_time:.2f} seconds')
                print(f"Logged info for step {i+1}. Check training_log.txt for details.")
                print(log_message)
                # Reset the loss history
                for k in loss_history:
                    loss_history[k].clear()
                running_loss = 0
            
            # new code block for saving every 250 steps, you can change it though
            if (i + 1) % saving_steps == 0:
                logging.info(f'Saving model checkpoint at step {i+1}...')
                state = {
                    'net':  {key: model[key].state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'train_loss': running_loss / (i + 1),
                    'epoch': epoch,
                    'step': i + 1,
                }
                if not os.path.isdir('checkpoint_PH_E2E_real_fixed_exactF0_gtstyle_snake_R_newsine_conf_dropout_F0_librilight_gt'):
                    os.mkdir('checkpoint_PH_E2E_real_fixed_exactF0_gtstyle_snake_R_newsine_conf_dropout_F0_librilight_gt')
                torch.save(state, f'./checkpoint_PH_E2E_real_fixed_exactF0_gtstyle_snake_R_newsine_conf_dropout_F0_librilight_gt/step_{i+1}_loss_{running_loss / (i + 1):.5f}.t7')
                print(f"Saved model checkpoint at step {i+1}. Check training_log.txt for details.")
            
        if (i + 1) % 10000 == 0:
            loss_test = 0

            _ = [model[key].eval() for key in model]

            with torch.no_grad():
                    iters_test = 0
                    for batch_idx, batch in enumerate(val_dataloader):
                        optimizer.zero_grad()

                        waves = batch[0]
                        batch = [b.to(device) for b in batch[1:]]
                        texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths = batch

                        with torch.no_grad():
                            mask = length_to_mask(mel_input_length // (2 ** asr_model.n_down)).to('cuda')
                            ppgs, s2s_pred, s2s_attn = asr_model(mels, mask, texts)

                            s2s_attn = s2s_attn.transpose(-1, -2)
                            s2s_attn = s2s_attn[..., 1:]
                            s2s_attn = s2s_attn.transpose(-1, -2)

                            text_mask = length_to_mask(input_lengths).to(texts.device)
                            attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                            attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                            attn_mask = (attn_mask < 1)
                            s2s_attn.masked_fill_(attn_mask, 0.0)
                            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** asr_model.n_down))
                            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)

                        mel_len_st = int(mel_input_length.min().item() / 2 - 1)
                        st = []
                        for bib in range(len(mel_input_length)):
                            mel_length = int(mel_input_length[bib].item() / 2)
                            random_start = np.random.randint(0, mel_length - mel_len_st)
                            st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                        st = torch.stack(st).detach()

                        s, t_en = model.style_encoder(st, t_en, input_lengths, t_en.size(-1))

                        asr = (t_en @ s2s_attn)

                        # get clips
                        mel_len = min([int(mel_input_length.min().item() / 2 - 1), 80])
                        en = []
                        gt = []
                        wav = []
                        for bib in range(len(mel_input_length)):
                            mel_length = int(mel_input_length[bib].item() / 2)

                            random_start = np.random.randint(0, mel_length - mel_len)
                            en.append(asr[bib, :, random_start:random_start+mel_len])
                            gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                            y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                            wav.append(torch.from_numpy(y).to('cuda'))

                        wav = torch.stack(wav).float().detach()

                        en = torch.stack(en)
                        gt = torch.stack(gt).detach()

            #             with torch.no_grad():
                        F0_real, _, F0 = F0_model(gt.unsqueeze(1))
                        F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                        # reconstruct
            #             ss = []

            #             for bib in range(len(mel_input_length)):
            #                 mel_length = int(mel_input_length[bib].item())
            #                 mel = mels[bib, :, :mel_input_length[bib]]
            #                 s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1), labels[bib].unsqueeze(0))
            #                 ss.append(s)
            #             s = torch.stack(ss).squeeze()            
            #             s = model.style_encoder(gt.unsqueeze(1), labels)

                        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                        y_rec = model.decoder(en, F0_real, real_norm, s)

                        loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                        loss_test += loss_mel
                        iters_test += 1

            logging.info(f'Epochs: {epoch + 1}')
            logging.info(f'Validation loss: {loss_test / iters_test:.3f}')
            print(f"Logged validation info for epoch {epoch + 1}. Check training_log.txt for details.")

            if epoch % saving_epoch == 0:
                if (loss_test / iters_test) < best_loss:
                    best_loss = loss_test / iters_test
                logging.info('Saving model checkpoint...')
                state = {
                    'net':  {key: model[key].state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'val_loss': loss_test / iters_test,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint_PH_E2E_real_fixed_exactF0_gtstyle_snake_R_newsine_conf_dropout_F0_librilight_gt'):
                    os.mkdir('checkpoint_PH_E2E_real_fixed_exactF0_gtstyle_snake_R_newsine_conf_dropout_F0_librilight_gt')
                torch.save(state, f'./checkpoint_PH_E2E_real_fixed_exactF0_gtstyle_snake_R_newsine_conf_dropout_F0_librilight_gt/val_loss_{loss_test / iters_test:.5f}.t7')
                print(f"Saved model checkpoint for epoch {epoch + 1}. Check training_log.txt for details.")

            _ = [model[key].train() for key in model]
            asr_model_copy = asr_model_copy.eval()
            F0_model_copy = F0_model_copy.eval()
            
    
if __name__=="__main__":
    main()
