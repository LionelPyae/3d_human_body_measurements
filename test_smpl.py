import cv2
import numpy as np
import pickle
from smpl import SMPL

# Load SMPL model
# Load SMPL model data from the pickle file
model_path = 'model/models/smpl/SMPL_NEUTRAL.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f, encoding='latin1')

# Create SMPL model
smpl = SMPL(model_path)
# Load image and detect keypoints using OpenCV
image = cv2.imread('Data/test.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = cv2.SIFT_create()
keypoints = detector.detect(image_gray, None)
keypoints = np.array([kp.pt for kp in keypoints])
print(keypoints)

# Fit SMPL to keypoints using iterative closest point (ICP) algorithm
R, T = smpl.iterative_closest_point(keypoints)

# Get 3D coordinates of SMPL vertices
vertices = smpl.get_vertices(R, T)

# Print some body measurements
height = smpl.get_height(vertices)
chest_circumference = smpl.get_circumference(vertices, ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'])
waist_circumference = smpl.get_circumference(vertices, ['left_hip', 'right_hip'])
hip_circumference = smpl.get_circumference(vertices, ['left_hip', 'right_hip', 'left_knee', 'right_knee'])
print('Height:', height)
print('Chest circumference:', chest_circumference)
print('Waist circumference:', waist_circumference)
print('Hip circumference:', hip_circumference)
