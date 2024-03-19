"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect and crop the human faces in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

For more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2
import numpy as np

from dataset.head_pose_estimator.face_detection import FaceDetector
from dataset.head_pose_estimator.mark_detection import MarkDetector
from dataset.head_pose_estimator.pose_estimation import PoseEstimator
from dataset.head_pose_estimator.utils import refine

class HeadPoseEstimator:
    def __init__(self):
        self.face_detector = FaceDetector("/home/adamh/rPPG/rPPG-Toolbox/dataset/head_pose_estimator/assets/face_detector.onnx")
        self.mark_detector = MarkDetector("/home/adamh/rPPG/rPPG-Toolbox/dataset/head_pose_estimator/assets/face_landmarks.onnx")
        self.pose_estimator = None

    def process(self, frames):
        if self.pose_estimator == None:
            self.pose_estimator = PoseEstimator(frames.shape[1], frames.shape[2])
            self.width = frames.shape[1]
            self.height = frames.shape[2]
        
        poses = []
        lums = []
        # Initialize saving of previous pose in case face detection fails
        cur_pose = np.array([0, 0, 0, 0, 0, 0])
        for frame in frames:
            lum_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
            avg_lum = np.mean(lum_frame[:, :, 0])
            lums.append(avg_lum)

            # Step 1: Get faces from current frame.
            faces, _ = self.face_detector.detect(frame, 0.7)

            # Any valid face found?
            if len(faces) > 0:

                # Step 2: Detect landmarks. Crop and feed the face area into the
                # mark detector. Note only the first face will be used for
                # demonstration.
                face = refine(faces, self.width, self.height, 0.15)[0]
                x1, y1, x2, y2 = face[:4].astype(int)
                patch = frame[y1:y2, x1:x2]

                if x1 == x2 or y1 == y2:
                    poses.append(cur_pose)
                    continue

                # Run the mark detection.
                marks = self.mark_detector.detect([patch])[0].reshape([68, 2])

                # Convert the locations from local face area to the global image.
                marks *= (x2 - x1)
                marks[:, 0] += x1
                marks[:, 1] += y1

                # Step 3: Try pose estimation with 68 points.
                pose = self.pose_estimator.solve(marks)
                cur_pose = np.ndarray.flatten(np.array(pose))
            
            poses.append(cur_pose)
        
        poses = np.array(poses)
        lums = np.array(lums).reshape(len(lums), 1)
        final = np.concatenate((poses, lums), axis=1)
        return final
