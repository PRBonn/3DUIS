import open3d as o3d
import MinkowskiEngine as ME
from models.minkunet import MinkUNet, Identity
from dataloader.SemanticKITTIDataLoader import SemanticKITTIDataLoader
import torch
from utils.utils import *
from utils.collations import *
import matplotlib.pyplot as plt
from utils.corr_utils import *
import networkx as nx
import click
import yaml
from os.path import join, dirname, abspath
import subprocess
import tqdm


def instance_segmentation(params, labels, model):
    # eval only model
    for param in model.parameters():
        param.require_grads = False

    # get numpy coords and features
    c = labels['coord'][0]
    f = labels['feats'][0]

    labels['cluster'] += 1

    ins, num_pts = np.unique(labels['cluster'], return_counts=True)
    # sort to run over the smaller proposals first
    # ins = ins[num_pts.argsort()]

    slc_full = np.zeros((len(c),), dtype=int)
    pred_ins_full = np.zeros((len(c),), dtype=int)

    ground_labels = np.zeros_like(c)
    if params['graphcut']['ground_remove']['instance_assign'] or params['graphcut']['ground_remove']['seed_sampling']:
        ground_labels = get_ground_labels(labels)

    for cluster in ins:
        # instance with id 0 is the ground ignore this and instance with just few point
        cls_points = np.where(labels['cluster'] == cluster)[0]
        if cluster == 0 or len(cls_points) <= 5:
            continue
        
        # get cluster
        cluster_center = c[cls_points].mean(axis=0)

        # crop a ROI around the cluster
        window_points = crop_region(c, labels['cluster'], cluster, params['graphcut']['roi_size'])

        # skip when ROI is empty        
        if not np.sum(window_points):
            continue

        # get closest point to the center
        center_dists = np.sqrt(np.sum((c[window_points] - cluster_center)**2, axis=-1))
        cluster_center = np.argmin(center_dists)

        # build input only with the ROI points
        x_forward = numpy_to_sparse_tensor(c[window_points][np.newaxis, :, :], f[window_points][np.newaxis, :, :])

        # forward pass ROI 
        model.eval()
        x_forward.F.requires_grad = True
        out = model(x_forward.sparse())
        out = out.slice(x_forward)

        # reset grads to compute saliency
        x_forward.F.grad = None

        # compute saliency for the point in the center
        slc = get_cluster_saliency(x_forward, out, np.where(labels['cluster'][window_points] == cluster)[0])
        slc_ = slc.copy()

        # place the computed saliency into the full point cloud for comparison
        slc_full[window_points] = np.maximum(slc_full[window_points], slc)

        # build graph representation
        G = build_graph(out.F.detach().cpu().numpy(),
                        slc[:,np.newaxis],
                        c[window_points],
                        cluster_center,
                        np.sum(labels['cluster'] == cluster),
                        params,
                        ground_labels[window_points],
                        np.where(labels['cluster'][window_points] == cluster)[0],
                    )
        # perform graph cut
        ins_points = graph_cut(G)

        # create point-wise prediction matrix
        pred_ins = np.zeros((len(x_forward),)).astype(int)
        if len(ins_points) != 0:
            pred_ins[ins_points] = cluster

        if params['graphcut']['ground_remove']['instance_assign']:
            # ignore assigned ground labels
            ins_ground = ground_labels[window_points] == 9
            pred_ins[ins_ground] = 0

        pred_ins_full[window_points] = np.maximum(pred_ins_full[window_points], pred_ins)
        # uncomment to see the instance-wise prediction
        # vis_pcd_saliency_corr(c, slc, pred_ins, labels['cluster'].copy().astype(int), window_points)

    # uncomment to see the full pcd instance prediction
    # vis_pcd_saliency_corr(c, slc_full, pred_ins_full.astype(int), labels['cluster'].astype(int).copy())
    return pred_ins_full

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/instance_seg.yaml'))
def main(config):
    cfg = yaml.safe_load(open(config))
    cfg['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())

    print(cfg)

    set_deterministic()
    model = sparse_models[cfg['model']['backbone']](in_channels=4, out_channels=96).type(torch.FloatTensor)
    model.cuda()

    checkpoint = torch.load(cfg['model']['checkpoint'], map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint[cfg['model']['checkpoint_key']])

    model.dropout = Identity()

    data_val = data_loaders[cfg['data']['dataset']](root=cfg['data']['path'], split=cfg['data']['split'])
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=1, collate_fn=SparseCollation(0.05, np.inf), shuffle=False, num_workers=10)

    data_iterator = iter(val_loader)

    for batch_data in tqdm.tqdm(data_iterator):

        coord = batch_data['feats'][0,:,:3]
        pred = instance_segmentation(cfg, batch_data, model)

        if cfg['experiment']['dump_pred']:
            scan_info, scan_file, seq_num = parse_scan_file(batch_data['scan_file'])
            pred = np.concatenate((coord, pred[:, np.newaxis]), axis=-1)
            dump_preds(pred, seq_num, scan_file, cfg, 'raw_pred')

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
