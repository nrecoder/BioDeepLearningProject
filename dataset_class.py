# Sleep stage mapping as before
#example usage
# Example usage
# subject_ids = ["S002", "S003", "S004"]
# data_directory = "/scratch/npr264/BioDeepL/dreamt/physionet.org/files/dreamt/2.0.0/data_64Hz"  
# demo_dataset = SleepDataset(subjects_list=subject_ids,
#                                  data_dir=data_directory, x_values='TEMPBVP')
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
def safe_float(value, default=np.nan):
    """
    Safely converts a value to a float.
    If the conversion fails, returns a default value.
    """
    try:
        return float(value)
    except (ValueError):
        return np.nan

SLEEP_STAGE_MAPPING = {
    "W": 0,    # Wake
    "N1": 1,   # non-REM stage 1
    "N2": 2,   # non-REM stage 2
    "N3": 3,   # non-REM stage 3
    "R": 4,    # REM
    "Missing": -1  # Missing label
}

def forward_fill(x):
    """
    Performs forward fill on a tensor.
    If x is 1D (shape [T]), it is temporarily unsqueezed to [T, 1].
    Assumes the first value is valid, or fills it with zero if needed.
    """
    single_channel = False
    if x.dim() == 1:
        x = x.unsqueeze(1)
        single_channel = True

    T, C = x.shape
    for c in range(C):
        if torch.isnan(x[0, c]):
            x[0, c] = 0.0
        for t in range(1, T):
            if torch.isnan(x[t, c]):
                x[t, c] = x[t - 1, c]
    if single_channel:
        x = x.squeeze(1)
    return x

numeric_columns = [
    'TIMESTAMP', 'BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP',
    'EDA', 'HR', 'IBI'
]
converters = {col: safe_float for col in numeric_columns}

class SleepDataset(Dataset):
    def __init__(self, subjects_list, data_dir, x_values, timestamp = False, max_length=2493810,debug=False):
        """ x_values = 'acc' or 'TEMPBVP'"""
        self.subjects = [{} for _ in range(len(subjects_list))]
        self.x_values = x_values
        if x_values == 'acc':
            downsample_freq=32
            cols = ['ACC_X', 'ACC_Y', 'ACC_Z']
        elif x_values == 'TEMPBVP':
            downsample_freq = 0.2
            cols = ['TEMP', 'BVP']
        else:
            print(x_values = 'acc' or 'TEMPBPV')
            return
        self.downsample = int(64 // downsample_freq)  # Downsample factor
        max_length = int(max_length // self.downsample)
        self.max_length = max_length
            
        all_cols = ['TIMESTAMP']+ cols
        #print(all_cols)
        
        for subjectNo, SID in enumerate(subjects_list):
            # Load the data for each subject
            file_path = os.path.join(data_dir, f"{SID}_whole_df.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(
                    file_path,
                    dtype={'Sleep_Stage': 'category'},
                    converters=converters,
                    low_memory=True
                )
                if debug:
                    print(f"loaded data for {SID}:")

                # Downsample the data if needed
                if self.downsample != 1:
                    df = df.iloc[::self.downsample].reset_index(drop=True)
                    if debug:
                        print(f"After downsampling by factor {self.downsample}, rows: {len(df)}")
                
                df = df[df['Sleep_Stage'] != 'P'] # remove data before PSG start
                for col in all_cols:
                    #print(df.columns)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df_X = df[all_cols].copy()
                # Normalize the features (z-score normalization per subject)
                columns_to_normalize = cols  # Exclude TIMESTAMP
                df_X[columns_to_normalize] = (df_X[columns_to_normalize] - df_X[columns_to_normalize].mean()) / df_X[columns_to_normalize].std()
                df['Sleep_Stage'] = df['Sleep_Stage'].astype(str).str.strip()
                df_Y = df['Sleep_Stage'].map(SLEEP_STAGE_MAPPING)
                
                # Pad/truncate the data to the downsampled max_length
                if len(df_X) > max_length:
                    if debug:
                        print(f"Truncating data for {SID} from {len(df_X)} to {max_length} samples.")
                    df_X = df_X.iloc[:max_length]
                    df_Y = df_Y.iloc[:max_length]
                else:
                    padding_length = max_length - len(df_X)
                    padding = pd.DataFrame(np.nan, index=np.arange(padding_length), columns=df_X.columns)
                    df_X = pd.concat([df_X, padding], ignore_index=True)
                    if timestamp == False:
                        df_X = df_X.drop('TIMESTAMP', axis = 1)
                    df_Y = pd.concat([df_Y, pd.Series([-1] * padding_length)], ignore_index=True)
                    
                self.subjects[subjectNo] = {
                    'data': df_X.values.astype(np.float32),  # shape: [T, C]
                    'labels': df_Y.to_numpy(),                 # shape: [T]
                    'SID': SID
                }
                if debug:
                    print(f"Data shape for {SID}: {df_X.shape}, Labels shape: {df_Y.shape}")
            else:
                warning(f"File {file_path} does not exist. Skipping subject {SID}.")
    def __len__(self):
        return len(self.subjects)

    def downsamplelabels(self, labels):
        #may need work
        samples_per_label = 32*5
        # Compute mode for every 5 seconds
        dataset_size = labels.shape[0]
        num_full_chunks = dataset_size // samples_per_label  # Number of complete 9600-sized chunks
        remainder_size = dataset_size % samples_per_label  # Remaining elements after full chunks
        
        modes = torch.tensor([
            torch.bincount(labels[i:i+samples_per_label]).argmax().item()
            for i in range(0, num_full_chunks * samples_per_label, samples_per_label)
        ])
        
        # Handle the remaining portion separately (if it exists)
        if remainder_size > 0:
            remainder_mode = torch.bincount(labels[num_full_chunks * samples_per_label:]).argmax()
            modes = torch.cat([modes, remainder_mode.unsqueeze(0)])
        return modes

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        data = torch.tensor(subject['data'], dtype=torch.float32)
        high_freq_labels = torch.tensor(subject['labels'], dtype=torch.long)
        labels = self.downsamplelabels(high_freq_labels)
        data = forward_fill(data) # fill NaNs with previous values
        labels = forward_fill(labels) # fill NaNs with previous values
        return data, labels

class SleepChunkDataset(Dataset):
    def __init__(self, subjects_list, data_dir, x_values, chunk_duration=600, chunk_stride=300, timestamp = False, debug=False):
        """
        Args:
            subjects_list (list): List of subject IDs, e.g. ["SID1", "SID2", ...].
            data_dir (str): Directory where files like "SID_whole_df.csv" are stored.
            chunk_duration (int): Chunk length in seconds (default 600 s for 10 minutes).
            chunk_stride (int): Time in seconds to step forward between chunks (default 300 s, for 50% overlap).
            downsample_freq (int): Desired sampling frequency after downsampling (original data are at 64 Hz).
            debug (bool): If True, print status messages.
        """
        self.x_values = x_values
        if x_values == 'acc':
            downsample_freq=32
            cols = ['ACC_X', 'ACC_Y', 'ACC_Z']
        elif x_values == 'TEMPBVP':
            downsample_freq = 0.2
            cols = ['TEMP', 'BVP']
        else:
            print(x_values = 'acc' or 'TEMPBPV')
            return
        self.downsample = int(64 // downsample_freq)  # Downsample factor
            
        all_cols = ['TIMESTAMP']+ cols
        self.chunks = []  # List to store each generated chunk (with its corresponding data, labels, and SID)
        # Effective sampling rate after downsampling becomes downsample_freq Hz.
        self.chunk_length = int(chunk_duration * downsample_freq)
        self.stride = int(chunk_stride * downsample_freq)

        for SID in subjects_list:
            file_path = os.path.join(data_dir, f"{SID}_whole_df.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, dtype={'Sleep_Stage': 'category'}, converters=converters, low_memory=True)
                if debug:
                    print(f"Loaded data for subject {SID}")
                
                # Downsample: every self.downsample-th row
                if self.downsample != 1:
                    df = df.iloc[::self.downsample].reset_index(drop=True)
                    if debug:
                        print(f"After downsampling (factor {self.downsample}), rows: {len(df)}")
                
                # Remove rows with "Preparation" phase if labeled 'P'
                df = df[df['Sleep_Stage'] != 'P']

                # Ensure numeric conversion for required columns
                for col in all_cols:
                    #print(df.columns)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df_X = df[all_cols].copy()
                # Normalize the features (z-score normalization per subject)
                columns_to_normalize = cols  # Exclude TIMESTAMP
                df_X[columns_to_normalize] = (df_X[columns_to_normalize] - df_X[columns_to_normalize].mean()) / df_X[columns_to_normalize].std()
                df['Sleep_Stage'] = df['Sleep_Stage'].astype(str).str.strip()
                df_Y = df['Sleep_Stage'].map(SLEEP_STAGE_MAPPING)
                if timestamp == False:
                    df_X = df_X.drop('TIMESTAMP', axis = 1)
                
                # Convert features and labels to numpy arrays
                data_arr = df_X.values.astype(np.float32)  # shape: [T, C]
                labels_arr = df_Y.to_numpy()                # shape: [T]
                T = data_arr.shape[0]

                # If the record is too short (less than one chunk), pad it with NaNs (-1 for labels)
                if T < self.chunk_length:
                    pad_size = self.chunk_length - T
                    padding_data = np.full((pad_size, data_arr.shape[1]), np.nan, dtype=np.float32)
                    data_arr = np.concatenate([data_arr, padding_data], axis=0)
                    padding_labels = np.full((pad_size,), -1)
                    labels_arr = np.concatenate([labels_arr, padding_labels], axis=0)
                    T = self.chunk_length  # update length

                # Slide a window over the data with the defined stride to create overlapping chunks
                for start in range(0, T - self.chunk_length + 1, self.stride):
                    end = start + self.chunk_length
                    chunk_data = data_arr[start:end, :]
                    chunk_labels = labels_arr[start:end]
                    self.chunks.append({
                        'data': chunk_data,
                        'labels': chunk_labels,
                        'SID': SID
                    })
                if debug:
                    num_chunks = (T - self.chunk_length) // self.stride + 1
                    print(f"Subject {SID}: {T} samples processed, generated {num_chunks} chunks")
            else:
                print(f"File {file_path} does not exist. Skipping subject {SID}")
                
    def downsamplelabels(self, labels):
        #may need work
        samples_per_label = 32*5
        # Compute mode for every 5 seconds elements
        dataset_size = labels.shape[0]
        num_full_chunks = dataset_size // samples_per_label  # Number of complete 9600-sized chunks
        remainder_size = dataset_size % samples_per_label  # Remaining elements after full chunks
        
        modes = torch.tensor([
            torch.bincount(labels[i:i+samples_per_label]).argmax().item()
            for i in range(0, num_full_chunks * samples_per_label, samples_per_label)
        ])
        
        # Handle the remaining portion separately (if it exists)
        if remainder_size > 0:
            remainder_mode = torch.bincount(labels[num_full_chunks * samples_per_label:]).argmax()
            modes = torch.cat([modes, remainder_mode.unsqueeze(0)])
        return modes
        
    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        data = torch.tensor(chunk['data'], dtype=torch.float32)
        high_freq_labels = torch.tensor(chunk['labels'], dtype=torch.long)
        labels = self.downsamplelabels(high_freq_labels)
        data = forward_fill(data)
        labels = forward_fill(labels)
        return data, labels