import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import json
import numpy as np
import os, cv2, glob, json, gc
import pandas as pd
import itertools
from itertools import chain
from moviepy.editor import VideoFileClip
import skvideo.io
from tqdm import tqdm
import circstat as CS
import scipy as sc
import math

np.float = np.float64
np.int = np.int_

def get_best_instance(instances):
    """
    Given a list of instances, return the index of the instance where keypoints are highest confidence.

    Parameters:
    instances (list): List of instances, where each instance contains a 'bbox' key with bounding box coordinates.

    Returns:
    int: The index of the instance whose center is closest to the frame center.
    """

    best_score = 0

    for e, instance in enumerate(instances):
        
        confidence = instance['keypoint_scores']

        if len(confidence) == 17:
            score = sum(confidence)

            if score > best_score:
                best_score = score
                n_instance = e

    
    return n_instance


def get_center_instance(instances, center_x, center_y):
    """
    Given a list of instances, return the index of the instance whose center is closest to the center of the video frame.

    Parameters:
    instances (list): List of instances, where each instance contains a 'bbox' key with bounding box coordinates.
    center_x (float): The x-coordinate of the frame center.
    center_y (float): The y-coordinate of the frame center.

    Returns:
    int: The index of the instance whose center is closest to the frame center.
    """

    distances = []
    for e, instance in enumerate(instances):
        
        bbox_x, bbox_y, bbox_width, bbox_height = instance['bbox'][0]
        bbox_center_x = bbox_x + bbox_width / 2
        bbox_center_y = bbox_y + bbox_height / 2

        # Calculate the Euclidean distance between the centers
        distance = math.sqrt((bbox_center_x - center_x)**2 + (bbox_center_y - center_y)**2)
        distances.append(distance)
    
    n_instance = distances.index(min(distances))
    
    return n_instance
        

def analyze_file(file_path, threshold=0.8):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    results = []
    for frame in data:
        frame_id = frame['frame_id']
        if frame['instances']:  # Ensure there is at least one detection
            instance_idx = get_best_instance(frame['instances'])
            first_instance = frame['instances'][instance_idx]
            keypoint_scores = first_instance['keypoint_scores']
            
            if len(keypoint_scores) != 17:
                continue
            
            all_above_thr = all(score > threshold for score in keypoint_scores)
            results.append({
                'frame_id': frame_id,
                'first_detection_keypoints': first_instance['keypoints'],
                'first_detection_confidence': keypoint_scores,
                'all_keypoints_above_thr': all_above_thr
            })
        else:
            continue
    
    return results


def find_continuous_good_blocks(analysis_results):
    
    """
    Find continuous blocks of frames where the first detection has all keypoints above a certain threshold.

    Parameters: 

    analysis_results (list): A list of dictionaries, where each dictionary contains the following keys:
        - frame_id: The frame number
        - first_detection_keypoints: The keypoints of the first detection
        - first_detection_confidence: The confidence scores of the first detection
        - all_keypoints_above_thr: A boolean indicating whether all keypoints are above a certain threshold
    """

    good_blocks = []
    current_block = []
    
    for result in analysis_results:
        if result['all_keypoints_above_thr']:
            current_block.append(result)
        else:
            if len(current_block) >= 30:
                good_blocks.append(current_block)
            current_block = []
    
    # Check if the last block in the sequence is a good block
    if len(current_block) >= 30:
        good_blocks.append(current_block)
    
    return good_blocks


def smooth_keypoints(block, window_size=3):
    """
    Apply rolling average smoothing to keypoints in a block.
    
    :param block: A list of frames, each frame is a dictionary with 'first_detection_keypoints'.
    :param window_size: Size of the rolling window for averaging.
    :return: A new block with smoothed keypoints.
    """
    # Convert block to numpy array for easier manipulation
    keypoints_array = np.array([frame['first_detection_keypoints'] for frame in block])
    num_frames, num_keypoints, _ = keypoints_array.shape
    
    # Initialize smoothed keypoints array
    smoothed_keypoints = np.copy(keypoints_array)
    
    # Apply rolling average
    for i in range(num_frames):
        start = max(0, i - window_size // 2)
        end = min(num_frames, i + window_size // 2 + 1)
        smoothed_keypoints[i] = np.mean(keypoints_array[start:end], axis=0)
    
    # Update block with smoothed keypoints
    smoothed_block = []
    for i, frame in enumerate(block):
        smoothed_frame = frame.copy()
        smoothed_frame['first_detection_keypoints'] = smoothed_keypoints[i].tolist()
        smoothed_block.append(smoothed_frame)
    
    return smoothed_block


def calculate_keypoint_displacements(block):
    """
    Calculate the total displacement for each keypoint in a block.
    
    :param block: List of frames, each frame is a dictionary with 'first_detection_keypoints'.
    :return: A list with the total displacement for each keypoint.
    """
    # Assuming each frame's keypoints are in the same order.
    displacements = [0] * len(block[0]['first_detection_keypoints'])  # Initialize displacements
    
    for i in range(1, len(block)):
        prev_keypoints = np.array(block[i-1]['first_detection_keypoints'])
        curr_keypoints = np.array(block[i]['first_detection_keypoints'])
        distances = np.linalg.norm(curr_keypoints - prev_keypoints, axis=1)
        displacements += distances  # Update total displacement for each keypoint

    displacements = np.array(displacements) / len(block)  # Normalize by number of frames
    
    return displacements

def filter_blocks_by_displacement(blocks, threshold):
    """
    Filter blocks to keep those where at least one keypoint's total displacement exceeds the threshold.
    
    :param blocks: List of blocks, each block is a list of frames.
    :param threshold: Displacement threshold for filtering.
    :return: Filtered list of blocks.
    """
    filtered_blocks = []
    
    for block in blocks:
        displacements = calculate_keypoint_displacements(block)
        if np.mean(displacements) > threshold:
            filtered_blocks.append(block)
    
    return filtered_blocks

def get_orig_video_info(file):
    file_path = file  # change to your own video path

    try:
        vid = cv2.VideoCapture(file_path)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = vid.get(cv2.CAP_PROP_FPS)

        center_x = (width) / 2
        center_y = (height) / 2
    except cv2.error as e:
        print(f"Caugt cv2 error, setting dummy params")
        width = 0
        height = 0
        center_x = 0
        center_y = 0
        fps = 0
        
    return width, height, center_x, center_y, fps


def find_file_by_basename(directory, base_name):
    """
    Find a file in the specified directory that has the given base name with any extension.

    Args:
    - directory (str): The directory to search in.
    - base_name (str): The base name of the file to find.

    Returns:
    - str: The path of the first matching file found, or None if no match is found.
    """

    for filename in os.listdir(directory):
        if os.path.splitext(filename)[0].lower() == base_name.lower():
            return os.path.join(directory, filename)
    return None

def reorder_keypoints(keypoints, confidence_scores):
    """
    Reorder the keypoints to the OpenPose format.
    The OpenPose format is as follows:
    0-17: [nose, neck, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist,
           right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle, right_eye, left_eye, right_ear, left_ear]
    The input 'keypoints' is a list of (x, y, c) tuples, where c is the confidence score.
    """

    # Reorder the keypoints to the OpenPose format
    keypoints = [keypoints[i] for i in [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]]
    confidence_scores = [confidence_scores[i] for i in [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]]

    return keypoints, confidence_scores

def rescale_keypoints(keypoints, scale):
    """
    Rescale the keypoints by the given scale.
    The input 'keypoints' is a list of (x, y) tuples
    """

    # Rescale the keypoints
    keypoints = [(x * scale, y * scale) for (x, y) in keypoints]

    return keypoints


def convert_coco_to_openpose(coco_keypoints, confidence_scores):
    """
    Convert COCO keypoints to OpenPose keypoints with the neck keypoint as the midpoint between the two shoulders.
    COCO keypoints format (17 keypoints): [nose, left_eye, right_eye, left_ear, right_ear,
                                           left_shoulder, right_shoulder, left_elbow, right_elbow,
                                           left_wrist, right_wrist, left_hip, right_hip,
                                           left_knee, right_knee, left_ankle, right_ankle]
    OpenPose keypoints format (18 keypoints): COCO keypoints + [neck]
    The neck is not a part of COCO keypoints and is computed as the midpoint between the left and right shoulders.
    """

    # Assuming coco_keypoints is a list of (x, y) tuples
    nose, left_eye, right_eye, left_ear, right_ear, \
    left_shoulder, right_shoulder, left_elbow, right_elbow, \
    left_wrist, right_wrist, left_hip, right_hip, \
    left_knee, right_knee, left_ankle, right_ankle = coco_keypoints

    # Calculate the neck as the midpoint between left_shoulder and right_shoulder
    neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
    neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
    neck = (neck_x, neck_y)


    # Assuming coco_keypoints is a list of (x, y) tuples
    c_nose, c_left_eye, c_right_eye, c_left_ear, c_right_ear, \
    c_left_shoulder, c_right_shoulder, c_left_elbow, c_right_elbow, \
    c_left_wrist, c_right_wrist, c_left_hip, c_right_hip, \
    c_left_knee, c_right_knee, c_left_ankle, c_right_ankle = confidence_scores

    # Calculate the neck as the midpoint between left_shoulder and right_shoulder
    c_neck = (c_left_shoulder + c_right_shoulder) / 2

    # Construct the OpenPose keypoints including the neck
    openpose_keypoints = [
        nose, left_eye, right_eye, left_ear, right_ear,
        left_shoulder, right_shoulder, left_elbow, right_elbow,
        left_wrist, right_wrist, left_hip, right_hip,
        left_knee, right_knee, left_ankle, right_ankle,
        neck  # Adding the neck as the last keypoint
    ]
    
    openpose_confidences = [
        c_nose, c_left_eye, c_right_eye, c_left_ear, c_right_ear,
        c_left_shoulder, c_right_shoulder, c_left_elbow, c_right_elbow,
        c_left_wrist, c_right_wrist, c_left_hip, c_right_hip,
        c_left_knee, c_right_knee, c_left_ankle, c_right_ankle,
        c_neck  # Adding the neck as the last keypoint
    ]

    openpose_keypoints, confidences = reorder_keypoints(openpose_keypoints, openpose_confidences)
    openpose_keypoints = rescale_keypoints(openpose_keypoints, 1)

    return openpose_keypoints, confidences

def get_fps(videoname):
    clip = VideoFileClip(videoname)
    return clip.fps

def read_video(video):

    videogen = skvideo.io.vreader(video)
    new_videogen = itertools.islice(videogen, 0, 1, 1)
    for image in new_videogen:
        a = 1
    return image

def get_video_information_yt(file_path):
    videofiles = np.array(glob.glob(os.path.join(file_path,'video*')))
    videofiles = videofiles[np.array([len(os.path.basename(i)) if i[-3:]!='pkl' else 0 for i in videofiles])==len('video_000000.mp4')]
    # get fps and screen dim
    df_fps = pd.DataFrame()
    fpsl = []
    rowlist = []; collist = []
    for ivideo in videofiles:
        print(ivideo)
        if os.path.basename(ivideo)[-3:]=='avi':
            fps = 30
        else:
            fps = get_fps(ivideo)
        fpsl.append(fps)
        img = read_video(ivideo)
        nrows = len(img)
        ncols = len(img[0])
        rowlist.append(nrows); collist.append(ncols);
    df_fps['fps'] = pd.Series(fpsl)
    df_fps['pixel_x'] = pd.Series(collist)
    df_fps['pixel_y'] = pd.Series(rowlist)
    df_fps['video'] = [os.path.basename(i)[:-4] for i in videofiles]
    return df_fps

def get_video_information_clinical(file_path):
    videofiles = np.array(glob.glob(os.path.join(file_path,'822487*')))
    videofiles = [i for i in videofiles if i[-3:]!='pkl']
    videofiles = [i for i in videofiles if i[-len('openposeLabeled.mp4'):]!='openposeLabeled.mp4']
    # get fps and screen dim
    df_fps = pd.DataFrame()
    fpsl = []
    rowlist = []; collist = []
    for ivideo in videofiles:
        if os.path.basename(ivideo)[:6]=='822487':
            fps = 30
        else:
            fps = get_fps(ivideo)
        fpsl.append(fps)
        img = read_video(ivideo)
        nrows = len(img)
        ncols = len(img[0])
        rowlist.append(nrows); collist.append(ncols);
    df_fps['fps'] = pd.Series(fpsl)
    df_fps['pixel_x'] = pd.Series(collist)
    df_fps['pixel_y'] = pd.Series(rowlist)
    df_fps['video'] = [os.path.basename(i)[:-4] for i in videofiles]
    return df_fps

def load_raw_pkl_files(path):
    pklfiles = np.array(glob.glob(os.path.join(path,'*.pkl')))
    df_pkl = pd.DataFrame()
    for file in pklfiles:
        one_file = pd.read_pickle(file).reset_index().drop('index',axis = 1)
        df_pkl = pd.concat([df_pkl, one_file])
    df_pkl = df_pkl.reset_index().drop('index', axis = 1)
    return df_pkl

def get_skel(df):
    if len(list(itertools.chain(*df.limbs_subset)))>0:
        peaks = df.peaks.iloc[0]
        parts_in_skel = df.limbs_subset.iloc[0]
        person_to_peak_mapping = [list(i[:-2]) for i in parts_in_skel] 
        skel_idx = [[i]*(len(iskel)-2) for i, iskel in enumerate(parts_in_skel)]
        idx_df = pd.DataFrame.from_dict({'peak_idx':list(itertools.chain(*person_to_peak_mapping)),\
         'person_idx':list(itertools.chain(*skel_idx))})
        peaks_list = list(chain.from_iterable(peaks))
        x = [ipeak[0] for ipeak in peaks_list]
        y = [ipeak[1] for ipeak in peaks_list]
        c = [ipeak[2] for ipeak in peaks_list]
        peak_idx = [ipeak[3] for ipeak in peaks_list]
        kp_idx = list(chain.from_iterable([len(ipeak)*[i] for i,ipeak in enumerate(peaks)]))
        peak_df = pd.DataFrame.from_dict({'x':x,'y':y,'c':c,'peak_idx':peak_idx,'part_idx':kp_idx})
        kp_df = pd.merge(idx_df, peak_df, on='peak_idx', how='left').drop('peak_idx',axis=1)
        kp_df = kp_df.loc[~kp_df.c.isnull(),:]
    else:
        kp_df = pd.DataFrame()
    return kp_df

def edit_df(df, df_fps):
    # keep person index with max number of keypoints per frame
    counts = df.groupby(['video','frame', 'person_idx'])['c'].count().reset_index()
    max_rows = counts.groupby(['video','frame'])['c'].idxmax().tolist()
    max_rows_df = counts.loc[max_rows,['video','frame', 'person_idx']]
    max_rows_df['dum'] = 1
    df = pd.merge(df.reset_index(), max_rows_df, on=['video','frame', 'person_idx'], how='inner')

    # add keypoint labels
    bps = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee",\
     "LAnkle","REye","LEye","REar","LEar"]
    df['bp'] = [bps[int(i)] for i in df.part_idx]
    df = df[['video','frame', 'x', 'y', 'bp', 'part_idx']] 

    # include row for each keypoint and frame
    max_frame = df.groupby('video').frame.max().reset_index()
    max_frame['frame_vec'] = max_frame.frame.apply(lambda x: np.arange(0,x+1))
    max_frame['bp'] = pd.Series([bps]*len(max_frame))
    y =[]
    _ = max_frame.apply(lambda x: [y.append([x['video'], x['bp'], i]) for i in x.frame_vec], axis=1)
    all_frames = pd.DataFrame(y, columns = ['video','bp','frame'])
    z =[]
    _ = all_frames.apply(lambda x: [z.append([x['video'], x['frame'], i]) for i in x.bp], axis=1)
    all_frames = pd.DataFrame(z, columns = ['video','frame', 'bp'])
    df = pd.merge(df, all_frames, on = ['video','frame','bp'], how='outer')
    df = pd.merge(df,df_fps, on = 'video', how='outer')
    df['time'] = df['frame']/df['fps']
    
    part_idx_df = df[['bp', 'part_idx']].drop_duplicates().dropna().sort_values('part_idx')
    df = pd.merge(df.drop('part_idx', axis=1), part_idx_df, on= 'bp', how='inner')
    
    return df

def interpolate_df(df):
    df = df.sort_values('frame')
    df['x']=df.x.interpolate()
    df['y']=df.y.interpolate()
    return df

def smooth(d, var, winmed, winmean):
    winmed1 = winmed
    winmed = int(winmed*d.fps.unique()[0])
    winmean = int(winmean*d.fps.unique()[0])
    d = d.reset_index(drop=True)
    if winmed>0:
        x = d.sort_values('frame')[var].rolling(center=True,window=winmed).median()
        d[var] = x.rolling(center=True,window=winmean).mean()
    else:
        d[var] = d[var]
    d['smooth'] = winmed1
    return d

def comp_joint_angle(df, joint_str):
    df = df.loc[(df.bp=='L'+joint_str)|(df.bp=='R'+joint_str)]
    df = pd.pivot_table(df, columns = ['bp'], values=['x', 'y'], index=['frame'])
    # zangle =np.arctan2(Rj.y.iloc[0]-Lj.y.iloc[0],Rj.x.iloc[0]-Lj.x.iloc[0])
    df[joint_str+'_angle']= np.arctan2((df['y', 'R'+joint_str]-df['y', 'L'+joint_str]),(df['x', 'R'+joint_str]-df['x', 'L'+joint_str]))
    df = df.drop(['x', 'y'], axis=1)
    return df

def comp_center_joints(df, joint_str, jstr):
    df = df.loc[(df.bp=='L'+joint_str)|(df.bp=='R'+joint_str)]
    df = pd.pivot_table(df, columns = ['bp'], values=['x', 'y'], index=['frame'])
    # zangle =np.arctan2(Rj.y.iloc[0]-Lj.y.iloc[0],Rj.x.iloc[0]-Lj.x.iloc[0])
    df[jstr+'y']= (df['y', 'R'+joint_str]+df['y', 'L'+joint_str])/2
    df[jstr+'x']= (df['x', 'R'+joint_str]+df['x', 'L'+joint_str])/2
    df = df.drop(['x', 'y'], axis=1)
    return df

def dont_normalise_skeletons(df):
    
    ''' Rotate keypoints around reference points (center of shoulders, center of hips)\
    Normalise points by reference distance (trunk length)'''

    bps = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee",\
     "LAnkle","REye","LEye","REar","LEar"]
    ubps = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist", "REye","LEye","REar","LEar"]
    lbps = ["RKnee","RAnkle","LHip","LKnee","LAnkle","RHip"]
    u_idx = np.where(np.isin(bps, ubps)==1)[0]
    l_idx = np.where(np.isin(bps, lbps)==1)[0]
    df['upper'] = np.isin(df.bp, ubps)*1
    # compute shoulder and hip angles for rotating, upper and lower body 
    # reference parts, now center of shoulders and hips
    
    s_angle = df.groupby(['video']).apply(lambda x: comp_joint_angle(x,'Shoulder')).reset_index()
    h_angle = df.groupby(['video']).apply(lambda x: comp_joint_angle(x,'Hip')).reset_index()
    uref = df.groupby(['video']).apply(lambda x: comp_center_joints(x, 'Shoulder', 'uref')).reset_index()
    lref = df.groupby(['video']).apply(lambda x: comp_center_joints(x, 'Hip', 'lref')).reset_index()
    s_angle['Hip_angle'] = h_angle['Hip_angle']
    s_angle = pd.merge(s_angle, uref, on=['video', 'frame'], how='inner')
    s_angle = pd.merge(s_angle, lref, on=['video', 'frame'], how='inner')
    s_angle.columns = s_angle.columns.get_level_values(0)

    df = pd.merge(df,s_angle, on=['video', 'frame'], how = 'outer')
    # set up columns, reference parts and reference angles

    df['refx'] = df['urefx']*df['upper'] + df['lrefx']*(1-df['upper'])
    df['refy'] = df['urefy']*df['upper'] + df['lrefy']*(1-df['upper'])
    df['ref_dist'] = np.sqrt((df['urefy']-df['lrefy'])**2+(df['urefx']-df['lrefx'])**2)
    df['ref_angle'] = df['Shoulder_angle']*df['upper'] + df['Hip_angle']*(1-df['upper'])

    df.loc[df.ref_angle<0,'ref_angle'] = 2*np.pi + df.loc[df.ref_angle<0,'ref_angle'] 
    df.loc[df.ref_angle<np.pi,'ref_angle'] = np.pi - df.loc[df.ref_angle<np.pi,'ref_angle'] 
    df.loc[(df.ref_angle>np.pi)&(df.ref_angle<2*np.pi),'ref_angle'] = 3*np.pi - df.loc[(df.ref_angle>np.pi)&(df.ref_angle<2*np.pi),'ref_angle']
    df['x_rotate'] = df['refx'] + np.cos(df['ref_angle'])*(df['x']-df['refx']) - np.sin(df['ref_angle'])*(df['y'] - df['refy'])
    df['y_rotate'] = df['refy'] + np.sin(df['ref_angle'])*(df['x']-df['refx']) + np.cos(df['ref_angle'])*(df['y'] - df['refy'])
    df['x_rotate'] = (df['x_rotate']-df['refx'])/df['ref_dist']
    df['y_rotate'] = (df['y_rotate']-df['refy'])/df['ref_dist']
    
    #### MODIFIED TO GET NON-NORMALISED SKELETONS
    #df['x'] = df['x_rotate']
    #df['y'] = df['y_rotate']
    # add to lower body to make trunk length 1
    #df.loc[df.upper==0,'y'] = df.loc[df.upper==0,'y']+1
    df['delta_t'] = 1/df['fps']
    
    return df


def normalise_skeletons(df):
    
    ''' Rotate keypoints around reference points (center of shoulders, center of hips)\
    Normalise points by reference distance (trunk length)'''

    bps = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee",\
     "LAnkle","REye","LEye","REar","LEar"]
    ubps = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist", "REye","LEye","REar","LEar"]
    lbps = ["RKnee","RAnkle","LHip","LKnee","LAnkle","RHip"]
    u_idx = np.where(np.isin(bps, ubps)==1)[0]
    l_idx = np.where(np.isin(bps, lbps)==1)[0]
    df['upper'] = np.isin(df.bp, ubps)*1
    # compute shoulder and hip angles for rotating, upper and lower body 
    # reference parts, now center of shoulders and hips
    
    s_angle = df.groupby(['video']).apply(lambda x: comp_joint_angle(x,'Shoulder')).reset_index()
    h_angle = df.groupby(['video']).apply(lambda x: comp_joint_angle(x,'Hip')).reset_index()
    uref = df.groupby(['video']).apply(lambda x: comp_center_joints(x, 'Shoulder', 'uref')).reset_index()
    lref = df.groupby(['video']).apply(lambda x: comp_center_joints(x, 'Hip', 'lref')).reset_index()
    s_angle['Hip_angle'] = h_angle['Hip_angle']
    s_angle = pd.merge(s_angle, uref, on=['video', 'frame'], how='inner')
    s_angle = pd.merge(s_angle, lref, on=['video', 'frame'], how='inner')
    s_angle.columns = s_angle.columns.get_level_values(0)

    df = pd.merge(df,s_angle, on=['video', 'frame'], how = 'outer')
    # set up columns, reference parts and reference angles

    df['refx'] = df['urefx']*df['upper'] + df['lrefx']*(1-df['upper'])
    df['refy'] = df['urefy']*df['upper'] + df['lrefy']*(1-df['upper'])
    df['ref_dist'] = np.sqrt((df['urefy']-df['lrefy'])**2+(df['urefx']-df['lrefx'])**2)
    df['ref_angle'] = df['Shoulder_angle']*df['upper'] + df['Hip_angle']*(1-df['upper'])

    df.loc[df.ref_angle<0,'ref_angle'] = 2*np.pi + df.loc[df.ref_angle<0,'ref_angle'] 
    df.loc[df.ref_angle<np.pi,'ref_angle'] = np.pi - df.loc[df.ref_angle<np.pi,'ref_angle'] 
    df.loc[(df.ref_angle>np.pi)&(df.ref_angle<2*np.pi),'ref_angle'] = 3*np.pi - df.loc[(df.ref_angle>np.pi)&(df.ref_angle<2*np.pi),'ref_angle']
    df['x_rotate'] = df['refx'] + np.cos(df['ref_angle'])*(df['x']-df['refx']) - np.sin(df['ref_angle'])*(df['y'] - df['refy'])
    df['y_rotate'] = df['refy'] + np.sin(df['ref_angle'])*(df['x']-df['refx']) + np.cos(df['ref_angle'])*(df['y'] - df['refy'])
    df['x_rotate'] = (df['x_rotate']-df['refx'])/df['ref_dist']
    df['y_rotate'] = (df['y_rotate']-df['refy'])/df['ref_dist']
    
    #### MODIFIED TO GET NON-NORMALISED SKELETONS
    df['x'] = df['x_rotate']
    df['y'] = df['y_rotate']
    # add to lower body to make trunk length 1
    df.loc[df.upper==0,'y'] = df.loc[df.upper==0,'y']+1
    df['delta_t'] = 1/df['fps']
    
    return df


