import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from dataset_manager import DatasetWriter # Passe den Import ggf. an

def generate_and_save_mnist_bags(
    output_path="../datasets/bags/mnist_bags.h5", 
    dataset_name="mnist_bags_base",
    target_number=9, 
    mean_bag_length=100, 
    # var_bag_length=20, 
    num_bag=1000, 
    seed=7, 
    split='train',
    method='base'
):
    var_bag_length = mean_bag_length / 5
    r = np.random.RandomState(seed)
    num_in_set = 60000 if split == 'train' else 10000
    split = split

    # 1. MNIST-Daten laden
    loader = data_utils.DataLoader(
        datasets.MNIST('../datasets', train=split == 'train', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=num_in_set,
        shuffle=False
    )

    for batch_data in loader:
        numbers = batch_data[0]
        labels = batch_data[1]

    # 2. DatasetWriter initialisieren
    writer = DatasetWriter(output_path)
    
    valid_bags_counter = 0
    label_of_last_bag = 0

    pos_indices = [i for i in range(len(labels)) if labels[i] == target_number]
    neg_indices = [i for i in range(len(labels)) if labels[i] != target_number]

    # 3. Bags generieren und direkt in die H5-Datei schreiben
    if method == 'base':
        while valid_bags_counter < num_bag:
            bag_length = int(r.normal(mean_bag_length, var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1
                
            indices = torch.LongTensor(r.randint(0, num_in_set, bag_length))
            labels_in_bag = labels[indices]

            # Wenn Positiv-Bag gewollt oder Zufall es ergibt
            if (target_number in labels_in_bag) and (label_of_last_bag == 0):
                # Positiver Bag
                label_of_last_bag = 1
            elif label_of_last_bag == 1:
                # Negativer Bag erzwingen: Keine Target Number erlaubt
                index_list = []
                bag_length_counter = 0
                while bag_length_counter < bag_length:
                    index = torch.LongTensor(r.randint(0, num_in_set, 1))
                    if labels[index].numpy()[0] != target_number:
                        index_list.append(index)
                        bag_length_counter += 1

                indices = torch.cat(index_list)
                labels_in_bag = labels[indices]
                label_of_last_bag = 0
            else:
                continue

            # Daten für das Abspeichern vorbereiten
            patches = numbers[indices]
            # Instance Labels (1 wenn Ziel-Ziffer, ansonsten 0)
            instance_label = (labels_in_bag == target_number).long().numpy() 
            bag_label = int(max(instance_label))
            
            # Koordinaten dummy generieren (falls notwendig für DatasetReader, z. B. alle 0)
            coords = torch.zeros(bag_length, 2)
            count = sum(instance_label)

            # Bag mit dem DatasetWriter speichern
            writer.write(
                output_path=output_path,
                dataset_name=dataset_name,
                image_id=f"{split}_{valid_bags_counter}",
                coords=coords,
                label=bag_label,
                patches=patches,
                count=count,
                instance_label=instance_label,
                split=split
            )
            
            valid_bags_counter += 1
    elif method == 'even':
        for i in range(num_bag//2):
            bag_length = 100
            num_positives = r.randint(1, bag_length)
            num_negatives = bag_length - num_positives

            # Positiver Bag
            pos_idx = r.choice(len(pos_indices), num_positives, replace=True)
            inst_idx = [pos_indices[i] for i in pos_idx]

            if num_negatives > 0:
                neg_idx = r.choice(len(neg_indices), num_negatives, replace=True)
                inst_idx += [neg_indices[i] for i in neg_idx]

            order = r.permutation(len(inst_idx))
            inst_idx = [inst_idx[i] for i in order]

            patches = numbers[inst_idx]
            patch_labels = labels[inst_idx]
            instance_label = (patch_labels == target_number).long().numpy()
            bag_label = int(max(instance_label))
            coords = torch.zeros(bag_length, 2)
            count = sum(instance_label)


            writer.write(
                output_path=output_path,
                dataset_name=dataset_name,
                image_id=f"{split}_{i}",
                coords=coords,
                label=bag_label,
                patches=patches,
                count=count,
                instance_label=instance_label,
                split=split
            )

        for i in range(num_bag//2, num_bag):
            bag_length = 100
            neg_idx = r.choice(len(neg_indices), bag_length, replace=True)
            inst_idx = [neg_indices[i] for i in neg_idx]

            order = r.permutation(len(inst_idx))
            inst_idx = [inst_idx[i] for i in order]

            patches = numbers[inst_idx]
            patch_labels = labels[inst_idx]
            instance_label = (patch_labels == target_number).long().numpy()
            bag_label = int(max(instance_label))
            coords = torch.zeros(bag_length, 2)
            count = sum(instance_label)


            writer.write(
                output_path=output_path,
                dataset_name=dataset_name,
                image_id=f"{split}_{i}",
                coords=coords,
                label=bag_label,
                patches=patches,
                count=count,
                instance_label=instance_label,
                split=split
            )
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == '__main__':
    # length = [10,50, 100]
    # bags = [50,100,150,200,300,400,500]
    # for l in length:
    #     for b in bags:
    #         dataset_name = f"bags{b}_len{l}"
    #         generate_and_save_mnist_bags(output_path='../datasets/mnist_bags_base.h5', dataset_name=dataset_name, split='train', num_bag=b, mean_bag_length=l)
    #         generate_and_save_mnist_bags(output_path='../datasets/mnist_bags_base.h5', dataset_name=dataset_name, split='test', num_bag=int(b/5), mean_bag_length=l)
    #         generate_and_save_mnist_bags(output_path='../datasets/mnist_bags_base.h5', dataset_name=dataset_name, split='validation', num_bag=int(b/5), mean_bag_length=l)
    generate_and_save_mnist_bags(output_path='../datasets/bags/mnist_bags_base.h5', dataset_name="mnist_bags_even", split='train', num_bag=1000, method ='even')
    generate_and_save_mnist_bags(output_path='../datasets//bags/mnist_bags_base.h5', dataset_name="mnist_bags_even", split='test', num_bag=200, method ='even')
    generate_and_save_mnist_bags(output_path='../datasets/bags/mnist_bags_base.h5', dataset_name="mnist_bags_even", split='validation', num_bag=200, method ='even')         