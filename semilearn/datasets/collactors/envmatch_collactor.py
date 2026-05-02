import torch

def envmatch_collate_fn(batch):
    """
    Custom collate function for MixColorDataset.
    Concatenates values of the same key across the batch into a single tensor.
    Handles missing keys by setting them to None only if all examples lack the key.
    Also handles nested dictionaries by recursively collating their contents.

    :param batch: List of dictionaries returned by the dataset's __getitem__ method.
    :return: A dictionary with keys from the dataset and concatenated values.
    """
    collated_batch = {}

    # Collect all possible keys from the batch
    all_keys = set().union(*(item.keys() for item in batch))

    for key in all_keys:
        values = [item.get(key, None) for item in batch]

        # If no sample has this key, set to None
        if all(v is None for v in values):
            collated_batch[key] = None
            continue

        # Filter out missing values
        present = [(i, v) for i, v in enumerate(values) if v is not None]
        idxs, present_vals = zip(*present)

        # Helper to collate a list of values
        def collate_list(vals):
            if isinstance(vals[0], torch.Tensor):
                return torch.cat([v.unsqueeze(0) for v in vals], dim=0)
            else:
                return torch.tensor(vals)

        # Nested dict handling
        if isinstance(present_vals[0], dict):
            sub_collated = {}
            sub_keys = set().union(*(v.keys() for v in present_vals))
            for sub_key in sub_keys:
                sub_vals = [v.get(sub_key, None) for v in present_vals]
                if all(sv is None for sv in sub_vals):
                    sub_collated[sub_key] = None
                else:
                    valid = [sv for sv in sub_vals if sv is not None]
                    sub_collated[sub_key] = collate_list(valid)
            collated_batch[key] = sub_collated
        else:
            collated_batch[key] = collate_list(present_vals)

        # For keys with some missing values, you may want to log or raise an error
        if len(present_vals) < len(values):
            missing = len(values) - len(present_vals)
            print(f"Warning: {missing} missing entries for key '{key}', collated only {len(present_vals)} items.")

    return collated_batch
