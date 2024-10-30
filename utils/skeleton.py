# COCO limb sequence (0-based indexing)
coco_limb_sequence = [
    (0, 1),  # Nose to Left Eye
    (0, 2),  # Nose to Right Eye
    (1, 3),  # Left Eye to Left Ear
    (2, 4),  # Right Eye to Right Ear
    (5, 7),  # Left Shoulder to Left Elbow
    (7, 9),  # Left Elbow to Left Wrist
    (6, 8),  # Right Shoulder to Right Elbow
    (8, 10), # Right Elbow to Right Wrist
    (5, 6),  # Left Shoulder to Right Shoulder
    (5, 11), # Left Shoulder to Left Hip
    (6, 12), # Right Shoulder to Right Hip
    (11, 12),# Left Hip to Right Hip
    (11, 13),# Left Hip to Left Knee
    (13, 15),# Left Knee to Left Ankle
    (12, 14),# Right Hip to Right Knee
    (14, 16) # Right Knee to Right Ankle
]

#MMPose limb sequence (output of MMPose pipeline)
limb_sequence = [
    (0,14),
    (0,15),
    (14,16),
    (15,17),
    (0,1),
    (1,2),
    (2,3),
    (3,4),
    (1,5),
    (5,6),
    (6,7),
    (1,8),
    (1,11),
    (8,9),
    (9,10),
    (11,12),
    (12,13),
    ]

mapping = {0:0,1:15,2:16,3:17,4:18,5:2,6:5,7:3,8:6,9:4,10:7,11:9,12:12,13:10,14:13,15:11,16:14}

# COCO part list
part_list = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle"
}



colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0]]