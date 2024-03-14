import glob
import os
import re

import cv2
import csv
import copy
import h5py
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader


class VicarPPG2Loader(BaseLoader):
    """The data loader for the VicarPPG-2 dataset."""
    BULK_FRAME_WORK = 300

    def __init__(self, name, data_path, config_data, sec_pre, model):
        """Initializes a VicarPPG-2 dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- Videos/
                     |      |-- 01-base.mp4
                     |      |-- 01-hrv.mp4
                     |      |...
                     |   |-- GroundTruth/
                     |      |-- PPG/
                     |          |-- Cleaned/
                     |              |-- 01-base PPG.csv
                     |              |-- 01-hrv PPG.csv
                     |              |-- ...
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, sec_pre, model)

    def get_raw_data(self, data_path):
        """Returns data directories under the path."""
        data_dirs = glob.glob(os.path.join(data_path, "Videos", "*.mp4"))
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            locs = list()
            locs.append(data_dir)
            name = os.path.split(data_dir)[-1][:-4]
            locs.append(os.path.join(data_path, "GroundTruth", "PPG", "Cleaned", name + " PPG.csv"))
            dirs.append({"index": name, "path": locs})
        return dirs
    
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocesses the raw data."""
        
        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(data_dirs[i]["path"][1])
        
        num_frames = self.read_video_frames(data_dirs[i]["path"][0])
        bvps = BaseLoader.resample_ppg(bvps, num_frames)
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        # Read Video Frames
        VidObj = cv2.VideoCapture(data_dirs[i]["path"][0])
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        raw_frames = list()
        frames = list()
        count = 1
        partial_pre_config = copy.deepcopy(config_preprocess)
        partial_pre_config.defrost()
        partial_pre_config.LABEL_TYPE = "Raw"
        partial_pre_config.DO_CHUNK = False
        while (success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0
            raw_frames.append(frame)
            # Videos are too massive to store whole thing in memory, so preprocess every X frames to decrease size requirements
            # Even with this, a single raw video cropped to 144x144 will take up 1.3GB of memory. Additional preprocessing that
            # turns the uint8s into float64s will multiply that memory cost by 8
            if count % self.BULK_FRAME_WORK == 0:
                partial_frame_clips, _ = self.preprocess(np.asarray(raw_frames), [], partial_pre_config)
                frames.append(partial_frame_clips)
                raw_frames = list()
            success, frame = VidObj.read()
            count += 1
        # Preprocess final batch of frames
        if len(raw_frames) > 0:
            partial_frame_clips, _ = self.preprocess(np.asarray(raw_frames), [], partial_pre_config)
            frames.append(partial_frame_clips)
            raw_frames = list()
        
        # Concatenate along time axis and then combine channels
        frames = np.concatenate(frames, axis=1)
        data = np.concatenate(frames, axis=-1)
            
        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            frames_clips, bvps_clips = self.chunk(
                data, bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, data_dirs[i]["index"])
        file_list_dict[i] = input_name_list
    
    def pose_lum_preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocesses the raw data."""
        
        # Read Video Frames
        VidObj = cv2.VideoCapture(data_dirs[i]["path"][0])
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        raw_frames = list()
        data = list()
        count = 1
        while (success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0
            raw_frames.append(frame)
            # Videos are too massive to store whole thing in memory, so preprocess every X frames to decrease size requirements
            # Even with this, a single raw video cropped to 144x144 will take up 1.3GB of memory. Additional preprocessing that
            # turns the uint8s into float64s will multiply that memory cost by 8
            if count % self.BULK_FRAME_WORK == 0:
                partial_data = self.pose_lum.process(raw_frames)
                data.append(partial_data)
                raw_frames = list()
            success, frame = VidObj.read()
            count += 1
        # Preprocess final batch of frames
        if len(raw_frames) > 0:
            partial_data = self.pose_lum.process(raw_frames)
            data.append(partial_data)
            raw_frames = list()
        
        # Concatenate along time axis
        data = np.concatenate(data, axis=1)
            
        if config_preprocess.DO_CHUNK:
            data_clips = self.pose_lum_chunk(data, config_preprocess.CHUNK_LENGTH)
        else:
            data_clips = np.array([data])
        input_name_list = self.pose_lum_save_multi_process(data_clips, data_dirs[i]["index"])
        file_list_dict[i] = input_name_list
    
    @staticmethod
    def read_video_frames(video_file):
        """Reads a video file, returns number of frames
        """
        VidObj = cv2.VideoCapture(video_file)
        return int(VidObj.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            return np.asarray([float(row["Signal"]) for row in reader])
