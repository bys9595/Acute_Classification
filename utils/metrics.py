import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics.functional.classification.auroc import binary_auroc, multilabel_auroc
from torchmetrics.functional.classification.confusion_matrix import binary_confusion_matrix, multiclass_confusion_matrix
from torchmetrics.functional.classification.roc import binary_roc, multilabel_roc
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from typing_extensions import Literal
from torchmetrics.utilities.data import dim_zero_cat
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import precision_recall_curve, auc, f1_score, accuracy_score, roc_auc_score, hamming_loss


class BinaryEvalMetrics(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        multidim_average: Literal["global", "samplewise"] = "samplewise",
        **kwargs: Any,
    ) -> None:
        super(BinaryEvalMetrics, self).__init__(**kwargs)
        self.multidim_average = multidim_average
        self._create_state(size=1, multidim_average=multidim_average)
        

    def _create_state(
        self,
        size: int,
        multidim_average: Literal["global", "samplewise"] = "samplewise",
    ) -> None:
        """Initialize the states for the different statistics."""
        default: Union[Callable[[], list], Callable[[], Tensor]]
        if multidim_average == "samplewise":
            default = list
            dist_reduce_fx = "cat"
        else:
            default = lambda: torch.zeros(size, dtype=torch.long)
            dist_reduce_fx = "sum"

        self.add_state("prob", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("target", default(), dist_reduce_fx=dist_reduce_fx)
    
    def update(self, logit: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        prob = torch.sigmoid(logit.detach())
        target = target.long().detach()
        
        if self.multidim_average == "samplewise":
            self.prob.append(prob)
            self.target.append(target)
    
    def _final_state(self) -> Tuple[Tensor, Tensor]:
        """Aggregate states that are lists and return final states."""
        prob = dim_zero_cat(self.prob)
        target = dim_zero_cat(self.target)
        
        return prob, target
    
    def compute(self) -> float:
        """Compute metrics based on inputs passed in to ``update`` previously."""
        prob, target = self._final_state()
        target = target.cpu()
        
        if len(target.shape) == 1:
            prob = prob.cpu().squeeze(1)
            AUC = binary_auroc(prob, target)
        else:
            AUC = multilabel_auroc(prob, target, num_labels=target.shape[1], average='none')
        
        
        return AUC
        
    def on_epoch_end_compute(self, best_thres: float=None) -> Dict:
        """Compute metrics based on inputs passed in to ``update`` previously."""
        
        metric_dict = {}
        
        prob, target = self._final_state()
        target = target.cpu()
        
        if len(target.shape) == 1:
            prob = prob.cpu().squeeze(1)
            fpr, tpr, thres = binary_roc(prob, target)
            idx = torch.argmax(tpr - fpr).item()
        else:
            fpr, tpr, thres = multilabel_roc(prob, target, num_labels=target.shape[1], average='none')
            idx = torch.argmax(tpr - fpr).item()
        
        if best_thres == None:
            best_thres = thres[idx].item()
        
        # Compute AUC
        AUC = binary_auroc(prob, target)
        CI95_lower, CI95_upper = self.bootstrap_auc(prob, target)
        
        # Prediction
        pred = prob.clone()
        pred[pred<best_thres] = 0
        pred[pred>=best_thres] = 1
        
        # Compute other metrics
        a = 1
        conf_mat = binary_confusion_matrix(pred, target, best_thres)
        tn, fp, fn, tp = conf_mat[0, 0].item(), conf_mat[0, 1].item(), conf_mat[1, 0].item(), conf_mat[1, 1].item()
        
        sensitivity = tp / max((tp + fn), 1e-6)
        specificity = tn / max((fp + tn), 1e-6)
        precision = tp / max((tp + fp), 1e-6)
        accuracy = (tp + tn) / max((tp + tn + fp + fn), 1e-6)
        f1 = 2 * precision * sensitivity / max((precision + sensitivity), 1e-6)
        
        metric_dict['Sensitivity'] = float(sensitivity)
        metric_dict['Specificity'] = float(specificity)
        metric_dict['Precision'] = float(precision)
        metric_dict['Accuracy'] = float(accuracy)
        metric_dict['F1_Score'] = float(f1)
        metric_dict['AUC'] = float(AUC)
        metric_dict['CI95_lower'] = float(CI95_lower)
        metric_dict['CI95_upper'] = float(CI95_upper)
        metric_dict['Best_thres'] = float(best_thres)
        
        ECE = self.metric_ece()
        metric_dict['ECE'] = float(ECE)
        
        return metric_dict
        
    def bootstrap_auc(self, prob, target, n_bootstraps=5000, alpha=0.05) -> Tuple[float, float]:
        assert len(target) == len(prob)
        n = len(target)
        auc_scores = []

        # Bootstrap resampling
        for _ in range(n_bootstraps):
            indices = torch.randint(0, len(target), (n,))
            auc = binary_auroc(prob[indices], target[indices]).item()
            auc_scores.append(auc)

        # Compute Confidence Interval
        sorted_scores = torch.tensor(sorted(auc_scores))
        lower = torch.quantile(sorted_scores, (alpha / 2.)).item()
        upper = torch.quantile(sorted_scores, (1 - alpha / 2.)).item()

        return lower, upper
    
    def plot_graphs(self, result_root_path):
        self.plot_roc_curve(result_root_path)
        self.plot_calibration_curve(result_root_path)
        
    def plot_roc_curve(self, result_root_path):
        prob, target = self._final_state()
        
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
        
        fpr, tpr, _ = binary_roc(prob, target)
        final_auc = auc(fpr, tpr)
        
        plt.plot(fpr.numpy(), tpr.numpy(), label='ROC (area = %0.2f)' % (final_auc))
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        save_path = os.path.join(result_root_path, 'images')
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(os.path.join(save_path, 'ROC_curve.jpg'))
        plt.close()
        plt.clf()


    def plot_calibration_curve(self, result_root_path):
        prob, target = self._final_state()
    
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
        
        display = CalibrationDisplay.from_predictions(target.numpy(), prob.numpy(), n_bins=20)
        cal_fig = display.figure_
        cal_fig.savefig(os.path.join(result_root_path, 'images', 'calibration_curve.jpg'))
        plt.close()
        plt.clf()
        
        plt.figure()
        bins = np.linspace(0, 1, 20)
        plt.hist(prob.numpy(), bins)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        
        save_path = os.path.join(result_root_path, 'images')
        os.makedirs(save_path, exist_ok=True)        
        
        plt.savefig(os.path.join(save_path, 'prob_histogram.jpg'))
        plt.close()
        plt.clf()


    def metric_ece(self, bin_size=0.1):
        prob, target = self._final_state()
        
        prob = prob.cpu().squeeze(1).numpy()
        target = target.cpu().numpy()
            
        prob = np.asarray(prob)
        target = np.asarray(target)
        
        total = len(prob)
        
        zero_class_conf = 1 - prob
        prob = np.stack((zero_class_conf, prob), axis=1)
        
        predictions = np.argmax(prob, axis=1)
        max_confs = np.amax(prob, axis=1)
        
        upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
        accs = []
        avg_confs = []
        bin_counts = []
        ces = []

        for upper_bound in upper_bounds:
            lower_bound = upper_bound - bin_size
            acc, avg_conf, bin_count = self.compute_bin(lower_bound, upper_bound, max_confs, predictions, target)
            accs.append(acc)
            avg_confs.append(avg_conf)
            bin_counts.append(bin_count)
            ces.append(abs(acc - avg_conf) * bin_count)

        ece = sum(ces) / total

        return ece
     
    def compute_bin(self, conf_thresh_lower, conf_thresh_upper, conf, pred, true):
        filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
        if len(filtered_tuples) < 1:
            return 0,0,0
        else:
            correct = len([x for x in filtered_tuples if x[0] == x[1]])
            avg_conf = sum([x[2] for x in filtered_tuples]) / len(filtered_tuples)
            accuracy = float(correct)/len(filtered_tuples)
            bin_count = len(filtered_tuples)
            return accuracy, avg_conf, bin_count
    




class MultiClassEvalMetrics(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        multidim_average: Literal["global", "samplewise"] = "samplewise",
        **kwargs: Any,
    ) -> None:
        super(MultiClassEvalMetrics, self).__init__(**kwargs)
        self.multidim_average = multidim_average
        self._create_state(size=1, multidim_average=multidim_average)
        

    def _create_state(
        self,
        size: int,
        multidim_average: Literal["global", "samplewise"] = "samplewise",
    ) -> None:
        """Initialize the states for the different statistics."""
        default: Union[Callable[[], list], Callable[[], Tensor]]
        if multidim_average == "samplewise":
            default = list
            dist_reduce_fx = "cat"
        else:
            default = lambda: torch.zeros(size, dtype=torch.long)
            dist_reduce_fx = "sum"

        self.add_state("prob", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("target", default(), dist_reduce_fx=dist_reduce_fx)
    
    def update(self, logit: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        prob = torch.softmax(logit.detach().float(), 1) # float16 -> float32 to prevent softmax failure 
        target = target.long().detach()
        
        if self.multidim_average == "samplewise":
            self.prob.append(prob)
            self.target.append(target)
    
    def _final_state(self) -> Tuple[Tensor, Tensor]:
        """Aggregate states that are lists and return final states."""
        prob = dim_zero_cat(self.prob)
        target = dim_zero_cat(self.target)
        
        return prob, target
    
    def compute(self) -> float:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        prob, target = self._final_state()
        
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
        
        # Prediction
        pred = torch.argmax(prob, 1)
        
        Accuracy = torch.sum(pred == target) / pred.shape[0]
                
        return Accuracy
        
    def on_epoch_end_compute(self, best_thres: float=None) -> Dict:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        
        metric_dict = {'Sensitivity': [], 'Specificity': [], 'Precision': [], 'Accuracy': [], 'F1_Score': []}
        
        prob, target = self._final_state()
        
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
                  
        # Prediction
        pred = torch.argmax(prob, 1)
        
        acc=torch.sum(pred == target) / pred.shape[0]
        metric_dict['total_Accuracy']= float(acc)

        # Compute other metrics
        conf_mat = multiclass_confusion_matrix(pred, target, prob.shape[1]) # j axis : false negative, i axis : false positives
        
        metric_dict['Sensitivity']= []
        metric_dict['Specificity']=[]
        metric_dict['Precision']=[]
        metric_dict['Accuracy']=[]
        metric_dict['F1_Score']=[]
        
        for i in range(len(conf_mat)):        
            tp = conf_mat[i, i]
            fn = conf_mat[i].sum() - tp
            fp = conf_mat[:, i].sum() - tp
            tn = conf_mat.sum() - (tp + fn + fp)
            
            sensitivity = tp / max((tp + fn), 1e-6)
            specificity = tn / max((fp + tn), 1e-6)
            precision = tp / max((tp + fp), 1e-6)
            accuracy = (tp + tn) / max((tp + tn + fp + fn), 1e-6)
            f1 = 2 * precision * sensitivity / max((precision + sensitivity), 1e-6)
            
            metric_dict['Sensitivity'].append(float(sensitivity))
            metric_dict['Specificity'].append(float(specificity))
            metric_dict['Precision'].append(float(precision))
            metric_dict['Accuracy'].append(float(accuracy))
            metric_dict['F1_Score'].append(float(f1))
        
        return metric_dict   
    


class MultilabelEvalMetrics(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        multidim_average: Literal["global", "samplewise"] = "samplewise",
        **kwargs: Any,
    ) -> None:
        super(MultilabelEvalMetrics, self).__init__(**kwargs)
        self.multidim_average = multidim_average
        self._create_state(size=1, multidim_average=multidim_average)
        

    def _create_state(
        self,
        size: int,
        multidim_average: Literal["global", "samplewise"] = "samplewise",
    ) -> None:
        """Initialize the states for the different statistics."""
        default: Union[Callable[[], list], Callable[[], Tensor]]
        if multidim_average == "samplewise":
            default = list
            dist_reduce_fx = "cat"
        else:
            default = lambda: torch.zeros(size, dtype=torch.long)
            dist_reduce_fx = "sum"

        self.add_state("prob", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("target", default(), dist_reduce_fx=dist_reduce_fx)
    
    def update(self, logit: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        prob = torch.sigmoid(logit.detach())
        target = target.long().detach()
        
        if self.multidim_average == "samplewise":
            self.prob.append(prob)
            self.target.append(target)
    
    def _final_state(self) -> Tuple[Tensor, Tensor]:
        """Aggregate states that are lists and return final states."""
        prob = dim_zero_cat(self.prob)
        target = dim_zero_cat(self.target)
        
        return prob, target
    
    def compute(self) -> float:
        """Compute metrics based on inputs passed in to ``update`` previously."""
        prob, target = self._final_state()
        target = target.cpu().as_tensor()
        prob = prob.cpu()
        
        AP = MultilabelAveragePrecision(num_labels=target.shape[1], average='macro')
        mAP = AP(prob, target).item()
        
        return mAP
        
    def on_epoch_end_compute(self, best_thres: float=None) -> Dict:
        """Compute metrics based on inputs passed in to ``update`` previously."""
        prob, target = self._final_state()
        target = target.cpu().as_tensor()
        prob = prob.cpu()
        
        # 클래스 개수
        num_classes = target.shape[1]

        # 각 클래스별 최적 Threshold 저장
        pr_aucs = []
        aurocs = []
        optimal_thresholds = []

        # Precision, Recall, Thresholds, F1 계산 및 최적 Threshold 선택
        for i in range(num_classes):
            precision, recall, thresholds = precision_recall_curve(target[:, i], prob[:, i])
            AUC = binary_auroc(prob[:, i], target[:, i])
            aurocs.append(AUC.item())
            
            # PR AUC 계산
            pr_auc = auc(recall, precision)
            pr_aucs.append(pr_auc)

            # F1 Score 계산
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)

            # 최적 F1 Score에서 Threshold 선택
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 1.0  # 안전 처리

            optimal_thresholds.append(float(optimal_threshold))

        if best_thres == None:
            best_thres = optimal_thresholds
        
        AP_func = MultilabelAveragePrecision(num_labels=target.shape[1], average='none')
        AP = AP_func(prob, target).tolist()
        mAP = np.mean(AP)
        
        # Compute AUC
        PR_AUC = np.mean(pr_aucs)
        AUROC = np.mean(aurocs)
                
        # Compute other metrics
        pred = prob.clone()
        pred = (prob >= torch.tensor(best_thres)).to(torch.int32)
        
        # accuracy 계산
        accuracy = (target == pred).sum(0) / len(target)
        accuracy = accuracy.tolist()
        mean_accuracy = np.mean(accuracy)

        f1_macro = f1_score(target, pred, average='macro')
        PR_AUC = np.mean(pr_aucs)
        AUROC = np.mean(aurocs)
        
        metric_dict = {}
        metric_dict['PR_AUC'] = PR_AUC
        metric_dict['AUROC'] = AUROC
        metric_dict['F1'] = f1_macro
        metric_dict['Accuracy'] = mean_accuracy
        metric_dict['Accuracy_per_class'] = accuracy
        metric_dict['AP'] = AP
        metric_dict['mAP'] = mAP
        metric_dict['Best_thres'] = best_thres
        
        return metric_dict

    
    def plot_graphs(self, result_root_path):
        self.plot_pr_curve(result_root_path)
        
    def plot_pr_curve(self, result_root_path):
        prob, target = self._final_state()
        
        prob = prob.cpu()
        target = target.cpu()
        
        precision, recall, thresholds = precision_recall_curve(target, prob)
        final_auc = auc(recall, precision)
        
        plt.plot(recall.numpy(), precision.numpy(), label='PR (area = %0.2f)' % (final_auc))
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")

        save_path = os.path.join(result_root_path, 'images')
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(os.path.join(save_path, 'PR_curve.jpg'))
        plt.close()
        plt.clf()