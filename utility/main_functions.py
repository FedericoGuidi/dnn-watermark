import numpy as np
from watermark_regularizers import get_wmark_regularizers

def save_watermark_sign(prefix, model):
    for layer_id, watermark_regularizer in get_wmark_regularizers(model):
        fname_w = prefix + '_layer{}_w.npy'.format(layer_id)
        fname_b = prefix + '_layer{}_b.npy'.format(layer_id)
        np.save(fname_w, watermark_regularizer.get_matrix())
        np.save(fname_b, watermark_regularizer.get_signature())

lr_schedule = [60, 120, 160]  # Epoch steps

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02 # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008
