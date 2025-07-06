import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.transforms import PadOrCrop


class MRNetDataset(Dataset):
    def __init__(self, data_dir, df_labels, views=['sagittal', 'coronal', 'axial'], 
                 tasks=['abnormal', 'acl', 'meniscus'], transform=None, target_slices=32, return_series_id=False):
        """
        Flexible Multi-view, Multi-label MRNet Dataset with slice padding/cropping.

        Arguments:
            data_dir (str): base path to the train/ or valid/ folder.
            df_labels (pd.DataFrame): DataFrame with 'Series' and target labels as columns.
            views (list): List of views to load (e.g., ['sagittal'], or all 3).
            tasks (list): List of target labels to include (e.g., ['acl'] or ['abnormal', 'acl', 'meniscus']).
            transform (callable): Optional volume-level transform (applied per view).
            target_slices (int): Number of slices each scan should be padded/cropped to.
            return_series_id (bool): If True, also return series_id (useful for debugging/inference).
        """
        self.data_dir = data_dir
        self.views = views
        self.tasks = tasks
        self.transform = transform
        self.target_slices = target_slices
        self.return_series_id = return_series_id

        # Ensure 'Series' is the index
        self.df_labels = df_labels.set_index('Series')
        self.series_ids = self.df_labels.index.tolist()

        # Instantiate padding transform once
        self.pad_crop = PadOrCrop(self.target_slices)

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        series_id = self.series_ids[idx]
        volumes = []

        # Load each view and apply transform + padding/cropping
        for view in self.views:
            file_path = os.path.join(self.data_dir, view, f"{int(series_id):04d}.npy")
            volume = np.load(file_path).astype(np.float32)

            # Normalise if needed
            if volume.max() > 1.0:
                volume = volume / 255.0

            # Convert to tensor: [1, slices, H, W]
            volume_tensor = torch.from_numpy(volume).unsqueeze(0)

            # Apply augmentation (if any)
            if self.transform:
                volume_tensor = self.transform(volume_tensor)
            
            # Apply standardisation, pad or crop to fixed depth
            volume_tensor = self.pad_crop(volume_tensor)

            volumes.append(volume_tensor)

        # Stack views along the channel dimension [#views, slices, H, W]
        view_tensor = torch.cat(volumes, dim=0)

        # Extract label(s) based on selected tasks
        labels = self.df_labels.loc[series_id][self.tasks].values.astype(np.float32)
        label_tensor = torch.tensor(labels)

        # Optionally return series_id (for debugging/logging/inference)
        if self.return_series_id:
            return view_tensor, label_tensor, series_id
        else:
            return view_tensor, label_tensor