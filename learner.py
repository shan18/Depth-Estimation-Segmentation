import torch

from tensornet.engine import Learner


class ModelLearner(Learner):
    
    def activate_logits(self, logits):
        return tuple(torch.sigmoid(x) for x in logits)
    
    def fetch_data(self, data):
        # Move data and targets to GPU
        inputs = {
            'bg': data[0]['bg'].to(self.device),
            'bg_fg': data[0]['bg_fg'].to(self.device),
        }
        targets = (
            data[1]['bg_fg_depth_map'].to(self.device),
            data[1]['bg_fg_mask'].to(self.device),
        )
        return inputs, targets
    
    def save_checkpoint(self, epoch=None):
        if not self.checkpoint is None:
            metric = None
            params = {}
            if self.checkpoint.monitor == 'train_loss':
                metric = self.train_losses[-1]
            elif self.checkpoint.monitor == 'val_loss':
                metric = self.val_losses[-1]
            elif self.metrics:
                if self.checkpoint.monitor.startswith('train_'):
                    if self.record_train:
                        metric = self.train_metrics[0][
                            self.checkpoint.monitor.split('train_')[-1]
                        ][-1]
                else:
                    metric = self.val_metrics[0][
                        self.checkpoint.monitor.split('val_')[-1]
                    ][-1]
            else:
                print('Invalid metric function, can\'t save checkpoint.')
                return
            
            self.checkpoint(self.model, metric, epoch)
