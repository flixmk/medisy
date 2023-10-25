from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from torchvision.datasets.folder import DatasetFolder
import pickle

def pickle_loader(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)

class PickleFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            samples_per_class: Optional[int] = None,
            total_samples: Optional[int] = None,
            classes_to_use: Optional[List[str]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pickle_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(PickleFolder, self).__init__(root, loader, ('.pkl',),
                                            transform=transform,
                                            target_transform=target_transform,
                                            is_valid_file=is_valid_file)
                                            
        if samples_per_class is not None or classes_to_use is not None:
            class_sample_count = defaultdict(int)
            filtered_samples = []
            
            for path, target in self.samples:
                if classes_to_use is not None and self.classes[target] not in classes_to_use:
                    continue
                if samples_per_class is not None and class_sample_count[target] >= samples_per_class:
                    continue

                filtered_samples.append((path, target))
                class_sample_count[target] += 1

            self.samples = filtered_samples        

        if total_samples is not None:
            class_sample_count = defaultdict(int)
            for _, target in self.samples:
                class_sample_count[target] += 1
            
            # calculate class distribution
            class_distribution = {k: v / len(self.samples) for k, v in class_sample_count.items()}
            print("Class distribution:")
            for class_id, class_frac in class_distribution.items():
                print(f"Class {class_id}: {class_frac * 100:.2f}%")

            resampled_samples = []
            for i in range(total_samples):
                # generate random idx between 0 and len(self.samples)
                idx = np.random.randint(0, len(self.samples))
                path, target = self.samples[idx]
                resampled_samples.append((path, target))

            self.samples = resampled_samples

            class_sample_count = defaultdict(int)
            for _, target in self.samples:
                class_sample_count[target] += 1
            
            # calculate class distribution
            class_distribution = {k: v / len(self.samples) for k, v in class_sample_count.items()}
            print("Class distribution after shuffling:")
            for class_id, class_frac in class_distribution.items():
                print(f"Class {class_id}: {class_frac * 100:.2f}%")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"latents": sample, "target": target}
    


