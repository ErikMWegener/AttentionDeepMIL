import h5py
import torch
import numpy as np

class DatasetWriter:
    def __init__(self, path):
        self.path = path

    # The write method takes the dataset name, image ID, coordinates of the patches, 
    # label of the bag, and optionally the patches themselves, count of instances in the bag, 
    # instance-level labels, and any additional metadata. 
    # It writes this information to an H5 file in a structured format.
    
    def write(self, dataset_name, image_id, coords, label, patches=None, 
              count=None, instance_label=None, split='train', **meta):
        with h5py.File(self.path, 'a') as f:
            key = f'{dataset_name}/{image_id}'
            if key in f:                
                del f[key]
            grp = f.create_group(key)
            if patches is not None:
                grp.create_dataset('patches', data=patches.numpy(),
                                   compression='gzip', compression_opts=4)
            grp.create_dataset('coords', data=coords.numpy())
            if instance_label is not None:
                grp.create_dataset('instance_label', data=instance_label)
            grp.attrs['label'] = label
            grp.attrs['count'] = count if count is not None else -1
            grp.attrs['split'] = split
            for k, v in meta.items():
                grp.attrs[k] = v

class DatasetReader(torch.utils.data.Dataset):
    def __init__(self, path, dataset_name, split='train'):
        self.path = path
        self.split = split
        with h5py.File(path, 'r') as f:
            self.keys = [
                f'{dataset_name}/{k}'
                for k in f[dataset_name]
                if f[f'{dataset_name}/{k}'].attrs.get('split') == split
            ]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.path, 'r') as f:
            grp = f[self.keys[idx]]
            patches = torch.from_numpy(grp['patches'][:])
            coords  = torch.from_numpy(grp['coords'][:])
            label   = int(grp.attrs['label'])
            count   = int(grp.attrs.get('count', -1))
            instance_label = grp.get('instance_label')
            if instance_label is None:
                # If 'instance_label' dataset doesn't exist, create a placeholder.
                # The size should match the number of patches.
                instance_label = torch.full((patches.shape[0],), -1, dtype=torch.int64)
            else:
                instance_label = torch.from_numpy(instance_label[:])
        return patches, coords, label, count, instance_label