import os
import time
import shutil

import numpy as np
import torch
import h5py
from torchsummary import summary
from sklearn.utils.class_weight import compute_class_weight
from omegaconf import OmegaConf

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from utils import LFPDataset
from models import ResNet_1D_Custom_Multihead, LightningModel


class TrainerMultihead():

    def __init__(self, cfg):
        self.cfg = cfg
        self.class_weights = None

        self.create_result_dir()
        self.load_data()

        # split data into k folds
        buf = np.arange(self.aligned_lfp.shape[0])
        np.random.seed(42)
        np.random.shuffle(buf)
        self.val_idx = np.array_split(buf, self.cfg.param.k_folds)

    def load_data(self):
        self.aligned_lfp = np.load(os.path.join(
            self.cfg.path.data_dir, 'aligned_lfp.npy'))
        self.texture = np.load(os.path.join(
            self.cfg.path.data_dir, 'label_texture.npy'))
        self.LD = np.load(os.path.join(self.cfg.path.data_dir, 'label_ld.npy'))

    def create_result_dir(self):
        # make dir for saving model weights
        if not os.path.exists(self.cfg.path.result_dir):
            os.mkdir(self.cfg.path.result_dir)

        if self.cfg.param.cross_val:
            for i in range(self.cfg.param.k_folds):
                if not os.path.exists(os.path.join(self.cfg.path.result_dir, f"/k_fold_{i+1}")):
                    os.mkdir(os.path.join(
                        self.cfg.path.result_dir, f"/k_fold_{i+1}"))
        else:
            if not os.path.exists(os.path.join(self.cfg.path.result_dir, f"/k_fold_1")):
                os.mkdir(os.path.join(self.cfg.path.result_dir, f"/k_fold_1"))

    def remove_result_dir(self):
        # remove dir for saving model weights
        if os.path.exists(self.cfg.path.result_dir):
            shutil.rmtree(self.cfg.path.result_dir)

    def report_model_summary(self):
        if self.cfg.preprocess.limit_chs != None:
            model = self.get_model(chs=1)
        else:
            model = self.get_model()
        # get pytorch model parameters
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad)

        self.cfg.result.total_params = pytorch_total_params
        self.cfg.result.total_params_trainable = pytorch_total_params_trainable

    def get_kfold_dataset(self, kfold_idx):
        # get train and validation idx
        val_idx = self.val_idx[kfold_idx]

        dataset_train = LFPDataset(
            self.cfg, self.aligned_lfp, self.texture, self.LD, val_idx, validation=False)
        dataset_val = LFPDataset(
            self.cfg, self.aligned_lfp, self.texture, self.LD, val_idx, validation=True)

        return dataset_train, dataset_val

    def get_model(self, chs=32, checkpoint_path=None):
        # get model
        print("Using ResNet_1D_Custom_Multihead")
        model = ResNet_1D_Custom_Multihead(
            [1, 1, 1, 1], num_classes=self.cfg.param.num_classes)

        model.apply(self.reset_weights)
        if checkpoint_path is not None:
            model = LightningModel.load_from_checkpoint(
                checkpoint_path,
                model=model,
                cfg=self.cfg,
                class_weights=self.class_weights)
        else:
            model = LightningModel(model, self.cfg, self.class_weights)

        return model

    def reset_weights(self, model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def train_kfold(self, kfold_idx):
        # set seed
        seed_everything(42, workers=True)

        # get dataset
        dataset_train, dataset_test = self.get_kfold_dataset(kfold_idx)

        # compute class_weight
        if self.cfg.param.class_weight:
            labels = np.array(np.argmax(dataset_train.label, axis=1))
            self.class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels),
                y=labels
            )
        else:
            self.class_weights = None

        # get dataloader
        if self.cfg.param.strategy == "ddp_spawn":
            num_workers = 0
        else:
            num_workers = 47

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.cfg.param.batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.cfg.param.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # create logger
        if not os.path.exists(self.cfg.path.result_dir + f"/k_fold_{kfold_idx+1}/logs/"):
            os.mkdir(self.cfg.path.result_dir + f"/k_fold_{kfold_idx+1}/logs/")
        logger = TensorBoardLogger(
            save_dir=self.cfg.path.result_dir + f"/k_fold_{kfold_idx+1}/logs/")

        if self.cfg.preprocess.limit_chs != None:
            model = self.get_model(chs=1)
        else:
            model = self.get_model()

        print(summary(model, tuple(dataset_train[0][0].shape), device='cpu'))

        # callbacks
        callbacks = []

        # early stopping callback
        if self.cfg.param.early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.param.early_stopping_patience,
                mode="min"
            )
            callbacks.append(early_stopping)

        # model checkpoint callback
        if self.cfg.param.model_checkpoint:
            model_checkpoint = ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                mode="min"
            )
            callbacks.append(model_checkpoint)

        if len(callbacks) == 0:
            callbacks = None

        # checkpoint callback
        trainer = L.Trainer(
            max_epochs=self.cfg.param.epochs,
            accelerator="gpu",
            devices=4,
            strategy=self.cfg.param.strategy,
            logger=logger,
            check_val_every_n_epoch=1,
            deterministic=True,
            callbacks=callbacks,
            num_sanity_val_steps=0
        )

        # Train
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # Save model
        trainer.save_checkpoint(
            self.cfg.path.result_dir + f"/k_fold_{kfold_idx+1}/model.ckpt")

        # predict
        if trainer.global_rank == 0:
            result = self.predict(
                model_checkpoint.best_model_path, dataset_test)
            with h5py.File(self.cfg.path.result_dir + f"/k_fold_{kfold_idx+1}/result.h5", "w") as f:
                f.create_dataset("cm_t", data=result["cm_t"])
                f.create_dataset("cm_l", data=result["cm_l"])
                f.create_dataset("acc_t", data=result["acc_t"])
                f.create_dataset("acc_l", data=result["acc_l"])
                f.create_dataset("y_hat", data=result["y_hat"])
                f.create_dataset("y", data=result["y"])

            print("Finished")

    def train(self):
        # preview data
        if self.cfg.debug.preview:
            self.dataloader.preview_data()

        # calculate training duration
        start = time.time()

        # Train model
        if self.cfg.param.cross_val:
            for kfold_idx in range(self.cfg.param.k_folds):
                self.train_kfold(kfold_idx)
        else:
            self.train_kfold(0)
        end = time.time()

        # report training duration
        self.cfg.result.duration = end - start

        # report model summary
        self.report_model_summary()

        # save config
        OmegaConf.save(self.cfg, self.cfg.path.result_dir + "/config.yaml")

    def predict(self, checkpoint_path, dataset_test):
        # print data
        print('Classes texture:', torch.bincount(
            dataset_test.label.type(torch.IntTensor)[:, 0]))
        print('Classes LD:', torch.bincount(
            dataset_test.label.type(torch.IntTensor)[:, 1]))
        # get dataloader
        num_workers = 11

        pred_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.cfg.param.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # Load trained model weights
        if self.cfg.preprocess.limit_chs != None:
            model = self.get_model(chs=1, checkpoint_path=checkpoint_path)
        else:
            model = self.get_model(checkpoint_path=checkpoint_path)

        trainer = L.Trainer(
            accelerator="gpu",
            devices=1
        )

        predictions = trainer.predict(
            model=model,
            dataloaders=pred_loader
        )

        # get result
        print("Converting tensor to numpy")

        def tsrlist_to_np(tensor_list):
            return np.array([x.cpu().numpy() for x in tensor_list])

        cm_t = model.confusion_matrix["texture"].compute()
        cm_l = model.confusion_matrix["ld"].compute()
        y_hat = tsrlist_to_np(model.test_step_outputs['y_hat'])
        y = tsrlist_to_np(model.test_step_outputs['y'])
        acc_t = model.pred_acc["texture"].compute().cpu().numpy()
        acc_l = model.pred_acc["ld"].compute().cpu().numpy()

        result = {
            "cm_t": cm_t,
            "cm_l": cm_l,
            "acc_t": acc_t,
            "acc_l": acc_l,
            "y_hat": y_hat,
            "y": y
        }

        return result

    def get_intermidiate_output(self, checkpoint_path, kfold_idx, dataset_test, layer_name):
        # get dataloader
        num_workers = 11

        val_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.cfg.param.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # Load trained model weights
        model = self.get_model(checkpoint_path=checkpoint_path)

        # get intermidiate output
        model.eval()
        model.to("cuda")
        model.freeze()

        # print
        # print(summary(model.model, tuple(dataset_test[0][0].shape) ,device = 'cuda'))

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        model.model.avgpool.register_forward_hook(get_activation('avgpool'))
        # model.model.layer3[0].downsample[1].register_forward_hook(get_activation('layer3.0.downsample'))

        intermidiate_output = []
        for i, (x, y) in enumerate(val_loader):
            x = x.to("cuda")
            with torch.no_grad():
                output = model.model(x)
                intermidiate_output.append(activation["avgpool"].cpu().numpy())

        intermidiate_output = np.concatenate(intermidiate_output, axis=0)

        return intermidiate_output
