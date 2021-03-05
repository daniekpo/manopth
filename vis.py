# %%
from manopth.manolayer import ManoLayer
from manopth.demo import display_hand
import torch
import numpy as np

batch_size = 1

# number of principal componenets for pose space
ncomps = 45

mano_layer = ManoLayer(
    mano_root="mano/models",
    use_pca=False,
    ncomps=ncomps,
    flat_hand_mean=False
)

pose = torch.tensor(
    [
        10.0, 10.0, 10.0,
        1.25, 0.5, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.5,
        0.25, 0.5, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.7,
        0.25, 0.5, 1.5,
        1.0, 0.0, 1.0,
        0.0, 0.0, 1.5,
        0.25, 0.5, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.7,
        -0.5, 0.5, 1.5,
        0.0, 0.5, 0.0,
        0.0, 1.0, -0.5,
    ]
)
pose = pose.unsqueeze(dim=0)
shape = torch.rand(batch_size, 10)

min_val = 0
max_val = 15
values = np.arange(min_val, max_val, 1)
for value in values:
    pose[0, 3] = value
    hand_verts, hand_joints = mano_layer(pose, shape)
    display_hand(
        {"verts": hand_verts, "joints": hand_joints},
        mano_faces=mano_layer.th_faces
    )

# %%
rot_max_list = [
    10.0, 10.0, 10.0,
    0.25, 0.5, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 0.5,
    0.25, 0.5, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 0.7,
    0.25, 0.5, 1.5,
    1.0, 0.0, 1.0,
    0.0, 0.0, 1.5,
    0.25, 0.5, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 0.7,
    -0.5, 0.5, 1.5,
    0.0, 0.5, 0.0,
    0.0, 1.0, -0.5
]
