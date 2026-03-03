# PhysTwin-MPC

## Whole Process
'''
# Calibrate the camera
python calibrate_cameras.py

# Capture the current obs
python capture_observation.py

# Segment the image
python segment_image.py

# Project to object pcd
python segment_to_pcd.py

# Project full scene to world coordinate and visualize with Open3D
python scene_to_world.py
'''

## Replay saved best action on xArm7
'''
python replay_best_action.py \
	--action-file outputs/plan/best_action_sequence.npy \
	--xarm-ip 192.168.1.196 \
	--base2world base2world.pkl
'''