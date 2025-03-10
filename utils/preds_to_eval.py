import numpy as np
import os

data_dir = '/media/lucas/PhData/InstanceSegSSL/SSL_graphcut_instance_segmentation_validation/08/raw_pred'
output_dir = './pred_eval/sequences/08/predictions/'
for i, fname in enumerate(sorted(os.listdir(data_dir))):
    pred = np.load(os.path.join(data_dir,fname))
    output_fname = os.path.join(output_dir, fname.split('.')[0].zfill(6) + '.label')
    print(output_fname)
    
    sem = np.zeros_like(pred[:,-1]).astype(np.float32)
    ins = pred[:,-1].astype(int) + 1
    pred_eval = sem + (ins << 16)
    pred_eval.astype(np.uint32).tofile(output_fname)