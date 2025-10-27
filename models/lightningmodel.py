import torch
from torchmetrics import Accuracy, MeanMetric, MetricCollection
from torchmetrics.classification import ConfusionMatrix
import lightning as L

class LightningModel(L.LightningModule):
    def __init__(self, model, cfg, class_weights):
        super().__init__()
        self.cfg = cfg
        self.model = model
        if class_weights is not None:
            self.register_buffer("class_weights", torch.from_numpy(class_weights).type(torch.FloatTensor))
        else:
            self.class_weights = None

        self.test_step_outputs = {'y_hat': [], 'y': []}

        self.metrics = MetricCollection(
            dict(
                train_acc_t = Accuracy(task="binary", num_classes=self.cfg.param.num_classes, average="macro"),
                train_acc_l = Accuracy(task="binary", num_classes=self.cfg.param.num_classes, average="macro"),
                train_loss = MeanMetric(),
                val_acc_t = Accuracy(task="binary", num_classes=self.cfg.param.num_classes, average="macro"),
                val_acc_l = Accuracy(task="binary", num_classes=self.cfg.param.num_classes, average="macro"),
                val_loss = MeanMetric()
            )
        )

        self.confusion_matrix = MetricCollection(
            dict(
                texture = ConfusionMatrix(task="binary", num_classes=self.cfg.param.num_classes),
                ld = ConfusionMatrix(task="binary", num_classes=self.cfg.param.num_classes)
            )
        )
        self.pred_acc = MetricCollection(
            dict(
                texture = Accuracy(task="binary", num_classes=self.cfg.param.num_classes, average="macro"),
                ld = Accuracy(task="binary", num_classes=self.cfg.param.num_classes, average="macro")
            )
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # forward pass
        outputs = self(batch[0])

        # calculate loss
        loss = self.criterion(outputs, batch[1])
        self.metrics["train_loss"].update(loss.detach())

        # calulate accuracy
        self.metrics["train_acc_t"].update(outputs[0], batch[1][0])
        self.metrics["train_acc_l"].update(outputs[1], batch[1][1])
        return loss

    def validation_step(self, batch, batch_idx):
        # forward pass
        outputs = self(batch[0])

        # calculate loss
        loss = self.criterion(outputs, batch[1])
        self.metrics["val_loss"].update(loss.detach())

        # calulate accuracy
        self.metrics["val_acc_t"].update(outputs[0], batch[1][0])
        self.metrics["val_acc_l"].update(outputs[1], batch[1][1])

    def on_validation_epoch_end(self):
        log_tmp = dict()
        log_metrics = self.metrics.compute()
        log_metrics = {k: v.item() for k, v in log_metrics.items()}
        log_tmp.update(log_metrics)
        self.metrics.reset()
        self.log_dict(log_tmp, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        # forward pass
        outputs = self(batch[0])

        # Calculate confusion matrix
        self.confusion_matrix['texture'].update(outputs[0], batch[1][0])
        self.confusion_matrix['ld'].update(outputs[1], batch[1][1])

        # Calculate accuracy
        self.pred_acc['texture'].update(outputs[0], batch[1][0])
        self.pred_acc['ld'].update(outputs[1], batch[1][1])

        self.test_step_outputs['y_hat'].append(outputs[0])
        self.test_step_outputs['y'].append(batch[1].argmax(dim=1)[0])

        return 0

    def on_predict_end(self) -> None:
        self.confusion_matrix.compute()
        self.pred_acc.compute()
        self.test_step_outputs['y_hat'] = torch.stack(self.test_step_outputs['y_hat'])
        self.test_step_outputs['y'] = torch.stack(self.test_step_outputs['y'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.trainer.model.parameters(), lr=self.cfg.param.lr
        )
        return optimizer
    
    def criterion(self, pred, target):
        # Define loss function
        if self.cfg.model.criterion == "BCEWithLogitsLoss":
            criterion = torch.nn.BCEWithLogitsLoss(weight=self.class_weights)
        elif self.cfg.model.criterion == "CrossEntropyLoss":
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        elif self.cfg.model.criterion == "MSELoss":
            criterion = torch.nn.MSELoss(weight=self.class_weights)
        return criterion(pred, target)