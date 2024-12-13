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

def get_joint_angles(df):
    
    ''' Compute joint angles from x,y coordinates '''

    df = df[~df.x.isnull()]
    df['delta_t'] = 1/df['fps']

    bps = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee",\
     "LAnkle","REye","LEye","REar","LEar"]
    joints = [['part0', 'part1', 'part2'],\
    ['RShoulder','Neck', 'RElbow'],\
    ['RElbow', 'RShoulder','RWrist'],\
    ['RHip','LHip', 'RKnee'],\
    ['RKnee', 'RHip','RAnkle'],\
    ['LShoulder','Neck', 'LElbow'],\
    ['LElbow', 'LShoulder','LWrist'],\
    ['LHip','RHip', 'LKnee'],\
    ['LKnee', 'LHip','LAnkle']]
    headers = joints.pop(0)
    df_joints = pd.DataFrame(joints, columns=headers).reset_index()
    df_joints['bpindex'] = df_joints['index']+1

    df_1 = df
    for i in [0,1,2]:
        df_joints['bp'] = df_joints['part'+str(i)]
        df_1 = pd.merge(df_1,df_joints[['bp', 'bpindex']], on='bp', how='left')
        df_1['idx'+str(i)] = df_1['bpindex'] 
        df_1 = df_1.drop('bpindex', axis=1)
        df_1['x'+str(i)] = df_1['x']*df_1['idx'+str(i)]/df_1['idx'+str(i)]
        df_1['y'+str(i)] = df_1['y']*df_1['idx'+str(i)]/df_1['idx'+str(i)]
    df0 = df_1[['video', 'frame', 'idx0', 'x0', 'y0', 'bp']]; df0 = df0.rename(index=str, columns={"bp": "bp0", "idx0": "idx"}); df0 = df0[~df0.idx.isnull()]
    df1 = df_1[['video', 'frame', 'idx1', 'x1', 'y1', 'bp']]; df1 = df1.rename(index=str, columns={"bp": "bp1", "idx1": "idx"}); df1 = df1[~df1.idx.isnull()]
    df2 = df_1[['video', 'frame', 'idx2', 'x2', 'y2', 'bp']]; df2 = df2.rename(index=str, columns={"bp": "bp2", "idx2": "idx"}); df2 = df2[~df2.idx.isnull()]
    df_2 = pd.merge(df0,df1, on=['video', 'frame', 'idx'], how='inner')
    df_2 = pd.merge(df_2,df2, on=['video', 'frame', 'idx'], how='inner')

    # compute angle here
    df_2['dot'] = (df_2['x1'] - df_2['x0'])*(df_2['x2'] - df_2['x0']) + (df_2['y1'] - df_2['y0'])*(df_2['y2'] - df_2['y0'])
    df_2['det'] = (df_2['x1'] - df_2['x0'])*(df_2['y2'] - df_2['y0']) - (df_2['y1'] - df_2['y0'])*(df_2['x2'] - df_2['x0'])
    df_2['angle_degs'] = np.arctan2(df_2['det'],df_2['dot'])*180/np.pi
    # hip and shoulder should be same regardless of side
    # elbow/knee give flexion/extension information only
    df_2['side'] = df_2.bp0.str[:1]
    df_2['part'] = df_2.bp0.str[1:]
    df_2['angle'] = df_2.angle_degs # same on left/right
    df_2.loc[(df_2['bp0']=='LShoulder')|(df_2['bp0']=='LHip'),'angle'] = \
    df_2.loc[(df_2['bp0']=='LShoulder')|(df_2['bp0']=='LHip'),'angle']*(-1)
    df_2.loc[(df_2['part']=='Elbow')|(df_2['part']=='Knee'),'angle'] = \
    np.abs(df_2.loc[(df_2['part']=='Elbow')|(df_2['part']=='Knee'),'angle'])
    # shoulders/hips: change to -180-+180 to 0-360 if neg: 360+angle
    df_2.loc[((df_2['part']=='Shoulder')|(df_2['part']=='Hip'))& (df_2.angle<0),'angle'] = \
    df_2.loc[((df_2['part']=='Shoulder')|(df_2['part']=='Hip'))& (df_2.angle<0),'angle']+360
    # can include shoulder rotation
    df_2['bp'] = df_2['bp0']

    df_info = df.groupby(['video', 'frame', 'fps','time', 'delta_t']).mean(numeric_only=True).reset_index()[['video', 'frame', 'fps','time', 'delta_t']]
    df_angle = pd.merge(df_2[['video', 'frame', 'bp', 'side', 'part', 'angle']],\
    df_info, on=['video', 'frame'], how='inner').drop_duplicates()
    return df_angle

def angular_disp(x,y): 
    possible_angles = np.asarray([y-x, y-x+360, y-x-360])
    idxMinAbsAngle = np.abs([y-x, y-x+360, y-x-360]).argmin(axis=0)
    smallest_angle = np.asarray([possible_angles[idxMinAbsAngle[i],i] for i in range(len(possible_angles[0]))])
    return smallest_angle

def get_angle_displacement(df, inp, outp): # different from other deltas - need shortest path
    df = df.sort_values('frame')
    angle = np.array(df[inp])
    a = angular_disp(angle[0:len(angle)-1], angle[1:len(angle)])
    df[outp] = np.concatenate((np.asarray([0]),a))
    return df

def smooth_dyn(df, inp, outp, win):
    fps = df['fps'].unique()[0]
    win = int(win*fps)
    x = df[inp].interpolate()
    df[outp] = x.rolling(window=win,center=False).mean()
    return df

def get_delta(df, inp, outp):
    x = df[inp]
    df[outp]  = np.concatenate((np.asarray([0]),np.diff(x)))*(np.asarray(x*0)+1)
    return df

def get_dynamics_xy(xdf, delta_window):
    # get velocity, acceleration
    xdf = xdf[['video','frame','x','y','bp','fps','pixel_x', 'pixel_y','time','delta_t', 'part_idx']]
    xdf = xdf.groupby(['bp','video']).apply(lambda x: get_delta(x,'x','d_x')).reset_index(drop=True)
    xdf = xdf.groupby(['bp','video']).apply(lambda x: get_delta(x,'y','d_y')).reset_index(drop=True)
    xdf['displacement'] = np.sqrt(xdf['d_x']**2 + xdf['d_y']**2)
    xdf['velocity_x_raw'] = xdf['d_x']/xdf['delta_t']
    xdf['velocity_y_raw'] = xdf['d_y']/xdf['delta_t']
    xdf = xdf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'velocity_x_raw','velocity_x', delta_window)).reset_index(drop=True)
    xdf = xdf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'velocity_y_raw','velocity_y', delta_window)).reset_index(drop=True)
    xdf = xdf.groupby(['bp','video']).apply(lambda x: get_delta(x,'velocity_x','delta_velocity_x')).reset_index(drop=True)
    xdf = xdf.groupby(['bp','video']).apply(lambda x: get_delta(x,'velocity_y','delta_velocity_y')).reset_index(drop=True)
    xdf['acceleration_x_raw'] = xdf['delta_velocity_x']/xdf['delta_t']
    xdf['acceleration_y_raw'] = xdf['delta_velocity_y']/xdf['delta_t']
    xdf = xdf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'acceleration_x_raw','acceleration_x', delta_window)).reset_index(drop=True)
    xdf = xdf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'acceleration_y_raw','acceleration_y', delta_window)).reset_index(drop=True)
    xdf['acceleration_x2'] = xdf['acceleration_x']**2
    xdf['acceleration_y2'] = xdf['acceleration_y']**2
    xdf['speed_raw'] = xdf['displacement']/xdf['delta_t']
    xdf = xdf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'speed_raw','speed', delta_window)).reset_index(drop=True)
    xdf['part'] = xdf.bp.str[1:]
    xdf['side'] = xdf.bp.str[:1]

    return xdf

def get_dynamics_angle(adf, delta_window):
    adf = adf.groupby(['bp','video']).apply(lambda x: get_angle_displacement(x,'angle','displacement')).reset_index(drop=True)
    adf['velocity_raw'] = adf['displacement']/adf['delta_t']
    adf = adf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'velocity_raw','velocity', delta_window)).reset_index(drop=True)
    adf = adf.groupby(['bp','video']).apply(lambda x: get_delta(x,'velocity','delta_velocity')).reset_index(drop=True)
    adf['acceleration_raw'] = adf['delta_velocity']/adf['delta_t']
    adf = adf.groupby(['bp','video']).apply(lambda x: smooth_dyn(x,'acceleration_raw','acceleration', delta_window)).reset_index(drop=True)
    adf['acceleration2'] = adf['acceleration']**2
    adf['part'] = adf.bp.str[1:]
    adf['side'] = adf.bp.str[:1]
    return adf


def angle_features(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    # - absolute angle
    a_mean = np.degrees(CS.nanmean(np.array(np.radians(df['angle']))))
    # - variability of angle
    a_stdev = np.sqrt(np.degrees(CS.nanvar(np.array(np.radians(df['angle'])))))
    # - measure of complexity (entropy)
    a_ent = ent(df['angle'].round())
    # - median absolute velocity
    median_vel = (np.abs(df['velocity'])).median()
    # - variability of velocity
    IQR_vel = (df['velocity']).quantile(.75) - (df['velocity']).quantile(.25)
    # - variability of acceleration
    IQR_acc = df['acceleration'].quantile(.75) - df['acceleration'].quantile(.25)

    return pd.DataFrame.from_dict({'video':np.unique(df.video),'bp':np.unique(df.bp),\
    'mean_angle':a_mean, 'stdev_angle':a_stdev, 'entropy_angle':a_ent,
    'median_vel_angle':median_vel,'IQR_vel_angle':IQR_vel,\
    'IQR_acc_angle': IQR_acc})

def rolling_angle_features(df, window_size=30, min_periods=1):
    """
    Compute rolling features for angles, velocities, and accelerations.

    Parameters:
    - df: DataFrame containing the data.
    - window_size: Rolling window size in frames (default: 30).
    - min_periods: Minimum number of observations in the window to compute a feature.

    Returns:
    - DataFrame with rolling features.
    """
    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Compute rolling features
    results = pd.DataFrame({
        'frame': df['frame'],
        'video': df['video'],
        'bp': df['bp'],
        # Absolute angle mean
        'mean_angle': np.degrees(df['angle'].rolling(window=window_size, min_periods=min_periods)
                                         .apply(lambda s: CS.nanmean(np.radians(s)), raw=False)),
        # Variability of angle (standard deviation)
        'stdev_angle': np.degrees(df['angle'].rolling(window=window_size, min_periods=min_periods)
                                          .apply(lambda s: np.sqrt(CS.nanvar(np.radians(s))), raw=False)),
        # Measure of complexity (entropy)
        'entropy_angle': df['angle'].rolling(window=window_size, min_periods=min_periods)
                                  .apply(lambda s: ent(s.round()), raw=False),
        # Median absolute velocity
        'median_vel': df['velocity'].rolling(window=window_size, min_periods=min_periods).apply(lambda s: np.median(np.abs(s)), raw=False),
        # Variability of velocity (IQR)
        'IQR_vel': df['velocity'].rolling(window=window_size, min_periods=min_periods).quantile(0.75) -
                           df['velocity'].rolling(window=window_size, min_periods=min_periods).quantile(0.25),
        # Variability of acceleration (IQR)
        'IQR_acc': df['acceleration'].rolling(window=window_size, min_periods=min_periods).quantile(0.75) -
                           df['acceleration'].rolling(window=window_size, min_periods=min_periods).quantile(0.25),
    })

    return results

def xy_features(df):
    # - absolute position/angle    
    median_x = df['x'].median()
    median_y = df['y'].median()
    IQR_x = df['x'].quantile(.75)-df['x'].quantile(.25)
    IQR_y = df['y'].quantile(.75)-df['y'].quantile(.25)
    # - median absolute velocity
    median_vel_x = np.abs(df['velocity_x']).median()
    median_vel_y = np.abs(df['velocity_y']).median()
    # - variability of velocity
    IQR_vel_x = df['velocity_x'].quantile(.75)-df['velocity_x'].quantile(.25)
    IQR_vel_y = df['velocity_y'].quantile(.75)-df['velocity_y'].quantile(.25)
    
    # - variability of acceleration
    IQR_acc_x = df['acceleration_x'].quantile(.75) - df['acceleration_x'].quantile(.25)
    IQR_acc_y = df['acceleration_y'].quantile(.75) - df['acceleration_y'].quantile(.25)
    
    # - measure of complexity (entropy)
    ent_x = ent(df['x'].round(2))
    ent_y = ent(df['y'].round(2))
    mean_ent = (ent_x+ent_y)/2
    # define part and side here
    return pd.DataFrame.from_dict({'video':np.unique(df.video),'bp':np.unique(df.bp),\
    'medianx': median_x, 'mediany': median_y, 'IQRx': IQR_x,'IQRy': IQR_y,\
    'medianvelx':median_vel_x, 'medianvely':median_vel_y,\
    'IQRvelx':IQR_vel_x,'IQRvely':IQR_vel_y,\
    'IQRaccx':IQR_acc_x,'IQRaccy':IQR_acc_y,'meanent':mean_ent})

def rolling_xy_features(df, window_size=30, min_periods=1):
    """
    Compute features over a rolling window.

    Parameters:
    - df: DataFrame containing required columns ('x', 'y', 'velocity_x', 'velocity_y', etc.).
    - sampling_rate: Sampling rate of the data (e.g., 30 Hz).
    - window_duration: Duration of the rolling window in seconds.

    Returns:
    - DataFrame containing computed features.
    """
    step_size = window_size // 2
    ent_x = df['x'].rolling(window=window_size, min_periods=min_periods).apply(lambda s: ent(s.round()), raw=False)
    ent_y = df['y'].rolling(window=window_size, min_periods=min_periods).apply(lambda s: ent(s.round()), raw=False)

    results = pd.DataFrame({
    'frame': df['frame'],
    'video': df['video'],
    'bp': df['bp'],
    'median_x': df['x'].rolling(window=window_size, min_periods=min_periods).median(),
    'median_y': df['y'].rolling(window=window_size, min_periods=min_periods).median(),
    'IQR_x': df['x'].rolling(window=window_size, min_periods=min_periods).quantile(.75) - \
             df['x'].rolling(window=window_size, min_periods=min_periods).quantile(.25),
    'IQR_y': df['y'].rolling(window=window_size, min_periods=min_periods).quantile(.75) - \
                df['y'].rolling(window=window_size, min_periods=min_periods).quantile(.25),
    'median_vel_x': np.abs(df['velocity_x']).rolling(window=window_size, min_periods=min_periods).median(),
    'median_vel_y': np.abs(df['velocity_y']).rolling(window=window_size, min_periods=min_periods).median(),
    'IQR_vel_x': df['velocity_x'].rolling(window=window_size, min_periods=min_periods).quantile(.75) - \
                 df['velocity_x'].rolling(window=window_size, min_periods=min_periods).quantile(.25),
    'IQR_vel_y': df['velocity_y'].rolling(window=window_size, min_periods=min_periods).quantile(.75) - \
                    df['velocity_y'].rolling(window=window_size, min_periods=min_periods).quantile(.25),
    'IQR_acc_x': df['acceleration_x'].rolling(window=window_size, min_periods=min_periods).quantile(.75) - \
                    df['acceleration_x'].rolling(window=window_size, min_periods=min_periods).quantile(.25),
    'IQR_acc_y': df['acceleration_y'].rolling(window=window_size, min_periods=min_periods).quantile(.75) - \
                    df['acceleration_y'].rolling(window=window_size, min_periods=min_periods).quantile(.25),
    'mean_ent': (ent_x + ent_y) / 2
    })

    return pd.DataFrame(results)

def ent(data):
    p_data= data.value_counts()/len(data) #  probabilities
    entropy=sc.stats.entropy(p_data) # corresponds to Shannon entropy
    return entropy

def corr_lr(df, var):
    idf = pd.DataFrame()
    idf['R'] = df[df.side=='R'].reset_index()[var]
    idf['L'] = df[df.side=='L'].reset_index()[var]
    return idf.corr().loc['L','R']


def rolling_corr_lr(df, window_size=30, min_periods=1, var='angle'):
    """
    Compute rolling left-right correlation for the specified variable, preserving frame numbers.

    Parameters:
    - df: DataFrame containing the data.
    - window_size: Rolling window size in frames (default: 30).
    - min_periods: Minimum number of observations in the window to compute correlation.
    - var: The column to compute the correlation on (default: 'angle').

    Returns:
    - DataFrame with rolling correlations and corresponding frame numbers.
    """
    # Ensure sides 'R' and 'L' are present
    # if 'R' not in df['side'].unique() or 'L' not in df['side'].unique():
    #     raise ValueError("Both 'R' and 'L' sides must be present in the data.")
    
    # Separate R and L data
    right = df[df.side == 'R'].reset_index(drop=True)
    left = df[df.side == 'L'].reset_index(drop=True)

    # Align lengths of R and L
    length = min(len(right), len(left))
    right = right.iloc[:length]
    left = left.iloc[:length]

    # Combine R and L with frame numbers
    combined = pd.DataFrame({
        'frame': right['frame'],  # Preserve frame numbers
        'R': right[var],
        'L': left[var]
    }).reset_index(drop=True)

    combined['rolling_corr'] = combined['R'].rolling(window=30, min_periods=1).corr(combined['L'])

    return combined