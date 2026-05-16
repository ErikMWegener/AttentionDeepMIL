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
            if coords is not None:
                grp.create_dataset('coords', data=coords.numpy())
            if instance_label is not None:
                grp.create_dataset('instance_label', data=instance_label)
            grp.attrs['label'] = label
            grp.attrs['count'] = count if count is not None else -1
            grp.attrs['split'] = split
            for k, v in meta.items():
                grp.attrs[k] = v

    def delete_dataset(self, dataset_name):
        with h5py.File(self.path, 'a') as f:
            if dataset_name in f:
                del f[dataset_name]
                print(f"Dataset '{dataset_name}' wurde erfolgreich gelöscht.")
            else:
                print(f"Dataset '{dataset_name}' nicht gefunden.")


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
            if 'coords' in grp:
                coords = torch.from_numpy(grp['coords'][:])
            else:
                coords = torch.empty(patches.shape[0], 0)  # Placeholder if 'coords' dataset doesn't exist
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
    
    def analyze_count_distribution(self):
        """
        Analysiert die Verteilung von count-Werten im Dataset.
        
        Args:
            split: Optional - 'train', 'validation', 'test'. 
                Wenn None, werden alle Splits berücksichtigt.
        
        Returns:
            dict mit detaillierten Statistiken
        """
        counts = []
        
        with h5py.File(self.path, 'r') as f:
            # Bestimme Dataset-Namen aus den Keys
            if not self.keys:
                return {"error": "No data found in this split"}
            
            # Extrahiere Dataset-Namen aus dem ersten Key
            dataset_name = self.keys[0].split('/')[0]
            dataset_group = f[dataset_name]
            
            # Wenn kein Split gefiltert werden soll, lese alle
            if self.split is None:
                for key in dataset_group:
                    grp = dataset_group[key]
                    count = grp.attrs.get('count', -1)
                    if count > 0:
                        counts.append(count)
            else:
                # Nutze bereits gefilterte Keys vom split
                for key in self.keys:
                    grp = f[key]
                    count = grp.attrs.get('count', -1)
                    if count > 0:
                        counts.append(count)
        
        if not counts:
            return {"error": "No valid counts found"}
        
        counts = np.array(counts)
        
        return {
            "total_samples": len(counts),
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "std": float(np.std(counts)),
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
            "q25": float(np.percentile(counts, 25)),
            "q75": float(np.percentile(counts, 75)),
            "histogram": self._get_histogram(counts),
            "all_counts": counts.tolist()
        }

    def _get_histogram(self, counts, bins=100):
        """Erstellt ein Histogram der count-Verteilung"""
        hist, bin_edges = np.histogram(counts, bins=bins)
        return {
            "frequencies": hist.tolist(),
            "bin_edges": bin_edges.tolist()
        }

    def print_count_distribution(self):
        """Gibt eine schöne Zusammenfassung der Verteilung aus"""
        stats = self.analyze_count_distribution()
        
        if "error" in stats:
            print(stats["error"])
            return
        
        print(f"\n{'='*50}")
        print(f"Count Distribution ({self.split} split)")
        print(f"{'='*50}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Mean: {stats['mean']:.2f}")
        print(f"Median: {stats['median']:.2f}")
        print(f"Std Dev: {stats['std']:.2f}")
        print(f"Range: [{stats['min']}, {stats['max']}]")
        print(f"Quartiles: Q25={stats['q25']:.2f}, Q75={stats['q75']:.2f}")
        print(f"{'='*50}\n")