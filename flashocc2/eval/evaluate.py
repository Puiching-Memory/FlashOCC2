# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from mmengine.logging import MMLogger

from mmdet3d.evaluation import SegMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class EvalMetric(SegMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        super(SegMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)

    def evaluation_semantic(self, 
                            gt_labels,
                            seg_preds,
                            label2cat,
                            ignore_index,
                            logger=None):
        assert len(seg_preds) == len(gt_labels)
        classes_num = len(label2cat)
        ret_dict = dict()
        results = []
        for i in tqdm(range(len(gt_labels))):
            gt_i = gt_labels[i].astype(np.int64)
            pred_i = seg_preds[i].astype(np.int64)
            mask = (gt_i != ignore_index)
            gt = gt_i[mask]
            pred = pred_i[mask]

            non_zero_gt = (gt != 0)
            non_zero_pred = (pred != 0)
            tp_j0 = np.sum(non_zero_gt & non_zero_pred)
            p_j0 = np.sum(non_zero_gt)
            g_j0 = np.sum(non_zero_pred)

            tp_all = np.bincount(gt[gt == pred], minlength=classes_num)
            p_all = np.bincount(gt, minlength=classes_num)
            g_all = np.bincount(pred, minlength=classes_num)

            score = np.zeros((classes_num, 3))
            score[0] = [tp_j0, p_j0, g_j0]
            score[1:,0] = tp_all[1:]
            score[1:,1] = p_all[1:]
            score[1:,2] = g_all[1:]

            results.append(score)
            
        results = np.stack(results, axis=0).mean(0)
        mean_ious = []
        for i in range(classes_num):
            tp = results[i, 0]
            p = results[i, 1]
            g = results[i, 2]
            union = p + g - tp
            mean_ious.append(tp / union)
        mean_ious = np.nan_to_num(mean_ious)
        for i in range(len(label2cat)):
            ret_dict[label2cat[i]] = mean_ious[i]
        ret_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])
        
        return ret_dict
        
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index']

        gt_semantic_masks = []
        pred_semantic_masks = []

        for eval_ann, sinlge_pred_results in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            pred_semantic_masks.append(
                sinlge_pred_results['pts_semantic_mask'])

        ret_dict = self.evaluation_semantic(
            gt_semantic_masks,
            pred_semantic_masks,
            label2cat,
            ignore_index,
            logger=logger)

        return ret_dict