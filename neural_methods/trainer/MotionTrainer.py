"""Trainer for Motion network."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.model.MotionNet import MotionNet as MN1
from neural_methods.model.MotionNet2 import MotionNet as MN2
from neural_methods.model.MotionNet3 import MotionNet as MN3
from neural_methods.model.MotionNet4 import MotionNet as MN4
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.model.PhysFormerPP import ViT_ST_ST_Compact3_TDC_PP_gra_sharp
from neural_methods.model.APNET import APNET
from tqdm import tqdm


class MotionTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.dropout_rate = config.MODEL.DROP_RATE
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.sec_pre = config.MODEL.SECONDARY_PREPROCESS
        self.base_model_type = config.MODEL.MOVEMENT_BASE
        self.base_model_loc = config.MODEL.MOVEMENT_BASE_LOC

        if self.base_model_type == "Physnet":
            self.base_model = PhysNet_padding_Encoder_Decoder_MAX(frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)
        elif self.base_model_type == "Tscan":
            self.base_model = TSCAN(frame_depth=config.MODEL.TSCAN.FRAME_DEPTH, img_size=72).to(self.device)
            self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        elif self.base_model_type == "EfficientPhys":
            self.base_model = EfficientPhys(config.MODEL.EFFICIENTPHYS.FRAME_DEPTH, img_size=72).to(self.device)
            self.base_model = torch.nn.DataParallel(self.base_model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.frame_depth = config.MODEL.EFFICIENTPHYS.FRAME_DEPTH
        elif self.base_model_type == "DeepPhys":
            self.base_model = DeepPhys(img_size=72, sec_pre=self.sec_pre).to(self.device)
            self.base_model = torch.nn.DataParallel(self.base_model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        elif self.base_model_type == "PhysFormer":
            self.base_model = ViT_ST_ST_Compact3_TDC_gra_sharp(
                            image_size=(self.chunk_len,128,128), 
                            patches=(config.MODEL.PHYSFORMER.PATCH_SIZE,) * 3, dim=config.MODEL.PHYSFORMER.DIM, ff_dim=config.MODEL.PHYSFORMER.FF_DIM, num_heads=config.MODEL.PHYSFORMER.NUM_HEADS, num_layers=config.MODEL.PHYSFORMER.NUM_LAYERS, 
                            dropout_rate=self.dropout_rate, theta=config.MODEL.PHYSFORMER.THETA).to(self.device)
        elif self.base_model_type == "PhysFormerPP":
            self.base_model = ViT_ST_ST_Compact3_TDC_PP_gra_sharp(
                            image_size=(self.chunk_len,128,128), 
                            patches=(config.MODEL.PHYSFORMER.PATCH_SIZE,) * 3, dim=config.MODEL.PHYSFORMER.DIM, ff_dim=config.MODEL.PHYSFORMER.FF_DIM, num_heads=config.MODEL.PHYSFORMER.NUM_HEADS, num_layers=config.MODEL.PHYSFORMER.NUM_LAYERS, 
                            dropout_rate=self.dropout_rate, theta=config.MODEL.PHYSFORMER.THETA, frame=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH).to(self.device)
        elif self.base_model_type == "APNET":
            self.base_model = APNET(hwDim=128, tDim=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH, dropout=config.MODEL.DROP_RATE).to(self.device)
        
        state_dict = torch.load(self.base_model_loc)
        #remove_prefix = 'module.'
        #state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
        self.base_model.load_state_dict(state_dict)
        self.base_model.eval()
        
        if config.TOOLBOX_MODE == "train_and_test":
            if config.MODEL.MOVEMENT_VER == 1:
                self.model = MN1(self.dropout_rate).to(self.device)
            elif config.MODEL.MOVEMENT_VER == 2:
                self.model = MN2(self.dropout_rate).to(self.device)
            elif config.MODEL.MOVEMENT_VER == 3:
                self.model = MN3(self.dropout_rate).to(self.device)
            elif config.MODEL.MOVEMENT_VER == 4:
                self.model = MN4(self.dropout_rate).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            if config.MODEL.MOVEMENT_VER == 1:
                self.model = MN1(self.dropout_rate).to(self.device)
            elif config.MODEL.MOVEMENT_VER == 2:
                self.model = MN2(self.dropout_rate).to(self.device)
            elif config.MODEL.MOVEMENT_VER == 3:
                self.model = MN3(self.dropout_rate).to(self.device)
            elif config.MODEL.MOVEMENT_VER == 4:
                self.model = MN4(self.dropout_rate).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("DeepPhys trainer initialized in incorrect toolbox mode!")
    
    def base_model_run(self, data):
        with torch.no_grad():
            if self.base_model_type == "Physnet":
                pred_ppg_test, _, _, _ = self.base_model(data)
                return pred_ppg_test
            elif self.base_model_type == "Tscan":
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                data = data[:(N * D) // self.frame_depth * self.frame_depth]
                pred_ppg_test = self.base_model(data)
                return pred_ppg_test
            elif self.base_model_type == "EfficientPhys":
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                data = data[:(N * D) // self.frame_depth * self.frame_depth]
                last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(1, 1, 1, 1)
                data = torch.cat((data, last_frame), 0)
                pred_ppg_test = self.base_model(data)
                if pred_ppg_test.shape[0] < N*D:
                    new_pred = torch.zeros((N*D, 1)).to(self.device)
                    new_pred[:pred_ppg_test.shape[0]] = pred_ppg_test
                    pred_ppg_test = new_pred
                return pred_ppg_test.view((N, D))
            elif self.base_model_type == "DeepPhys":
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                pred_ppg_test = self.base_model(data)
                return pred_ppg_test.view((N, D))
            elif self.base_model_type == "PhysFormer":
                gra_sharp = 2.0
                pred_ppg_test, _, _, _ = self.base_model(data, gra_sharp)
                return pred_ppg_test
            elif self.base_model_type == "PhysFormerPP":
                gra_sharp = 2.0
                pred_ppg_test, _ = self.base_model(data, gra_sharp)
                return pred_ppg_test
            elif self.base_model_type == "APNET":
                pred_ppg_test = self.base_model(data)
                return pred_ppg_test


    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                base_rppg = self.base_model_run(data)
                motion = self.get_motion_data(batch[4])
                labels = labels.view(-1, 1)
                self.optimizer.zero_grad()
                motion_inputs = self.create_inputs(base_rppg.cpu(), motion).to(self.device)
                pred_ppg = self.model(motion_inputs)
                pred_ppg = pred_ppg.view(-1, 1)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix({"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]})

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                base_rppg = self.base_model_run(data_valid)
                motion = self.get_motion_data(valid_batch[4])
                labels_valid = labels_valid.view(-1, 1)
                motion_inputs = self.create_inputs(base_rppg.cpu(), motion).to(self.device)
                pred_ppg_valid = self.model(motion_inputs)
                pred_ppg_valid = pred_ppg_valid.view(-1, 1)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        config = self.config
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            state_dict = torch.load(self.config.INFERENCE.MODEL_PATH)
            self.model.load_state_dict(state_dict)
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                base_rppg = self.base_model_run(data_test)
                motion = self.get_motion_data(test_batch[4])
                labels_test = labels_test.view(-1, 1)
                motion_inputs = self.create_inputs(base_rppg.cpu(), motion).to(self.device)
                pred_ppg_test = self.model(motion_inputs)
                pred_ppg_test = pred_ppg_test.view(-1, 1)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
        
        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        """Inits parameters from args and the writer for TensorboardX."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
    
    @staticmethod
    def get_motion_data(files):
        data = []
        for file in files:
            fname = f"{file.split('.')[0]}_pl.npy"
            data.append(np.load(fname))
        return torch.Tensor(data)

    @staticmethod
    def create_inputs(rppg, motion):
        movement = MotionTrainer.diff_normalize_data(motion[:, :, :3])
        pose = MotionTrainer.standardized_data(motion[:, :, 3:6])
        luminance = MotionTrainer.standardized_data(motion[:, :, 6].view((motion.shape[0], motion.shape[1], 1)))
        rppg = torch.tensor(np.array(rppg.view((rppg.shape[0], rppg.shape[1], 1))))
        return torch.concat((rppg, movement, pose, luminance), dim=2)
    
    @staticmethod
    def diff_normalize_data(data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, t, c = data.shape
        tensors = []
        for i in range(n):
            data_i = np.array(data[i])
            diffnormalized_len = t - 1
            diffnormalized_data = np.zeros((diffnormalized_len, c), dtype=np.float32)
            diffnormalized_data_padding = np.zeros((1, c), dtype=np.float32)
            for j in range(diffnormalized_len):
                diffnormalized_data[j, :] = (data_i[j + 1, :] - data_i[j, :]) / (
                        data_i[j + 1, :] + data_i[j, :] + 1e-7)
            diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
            diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
            diffnormalized_data[np.isnan(diffnormalized_data)] = 0
            tensors.append(diffnormalized_data)
        return torch.Tensor(tensors)

    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        tensors = []
        for i in range(data.shape[0]):
            data_i = np.array(data[i])
            data_i = data_i - np.mean(data_i)
            data_i = data_i / np.std(data_i)
            data_i[np.isnan(data_i)] = 0
            tensors.append(data_i)
        return torch.Tensor(tensors)
 