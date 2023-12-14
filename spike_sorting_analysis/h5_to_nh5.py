from typing import Union
import json
import numpy as np
import h5py


def h5_to_nh5(h5_path: str, nh5_path: str) -> str:
    """Converts an h5 file to an nh5 file.

    Args:
        h5_path (str): Path to the h5 file.
        nh5_path (str): Path to the nh5 file.
    """
    with h5py.File(h5_path, 'r') as h5_file:
        with open(nh5_path, 'wb') as nh5_file:
            header = {
                'datasets': [],
                'groups': []
            }
            all_groups_in_h5_file = _get_h5_groups(h5_file)
            for group in all_groups_in_h5_file:
                header['groups'].append({
                    'path': group.name,
                    'attrs': json.loads(_attrs_to_json(group))
                })
            all_datasets_in_h5_file = _get_h5_datasets(h5_file)
            position = 0
            for dataset in all_datasets_in_h5_file:
                header['datasets'].append({
                    'path': dataset.name,
                    'attrs': json.loads(_attrs_to_json(dataset)),
                    'dtype': _dtype_to_str(dataset),
                    'shape': _format_shape(dataset),
                    'position': int(position)
                })
                position += _get_dataset_byte_count(dataset)
            header_json = json.dumps(header).encode('utf-8')
            nh5_file.write(f'nh5|1|{len(header_json)}|'.encode('utf-8'))
            nh5_file.write(header_json)
            position = 0
            for dataset in all_datasets_in_h5_file:
                nh5_file.write(dataset[...].tobytes())
                position += _get_dataset_byte_count(dataset)

def _get_h5_groups(h5_file: h5py.File) -> list:
    """Returns a list of all groups in an h5 file.

    Args:
        h5_file (h5py.File): The h5 file.

    Returns:
        list: A list of all groups in the h5 file.
    """
    groups = []

    # include root group
    groups.append(h5_file)

    def _get_groups(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append(obj)
    h5_file.visititems(_get_groups)
    return groups

def _get_h5_datasets(h5_file: h5py.File) -> list:
    """Returns a list of all datasets in an h5 file.

    Args:
        h5_file (h5py.File): The h5 file.

    Returns:
        list: A list of all datasets in the h5 file.
    """
    datasets = []

    def _get_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(obj)
    h5_file.visititems(_get_datasets)
    return datasets

def _attrs_to_json(group: Union[h5py.Group, h5py.Dataset]) -> str:
    """Converts the attributes of an HDF5 group or dataset to a JSON-serializable format."""
    attrs_dict = {}
    for attr_name in group.attrs:
        value = group.attrs[attr_name]

        # Convert NumPy arrays to lists
        if isinstance(value, np.ndarray):
            value = value.tolist()
        # Handle other non-serializable types as needed
        if isinstance(value, np.int64):
            value = int(value)

        attrs_dict[attr_name] = value

    return json.dumps(attrs_dict)

def _dtype_to_str(dataset: h5py.Dataset) -> str:
    """Converts the dtype of an HDF5 dataset to a string."""
    dtype = dataset.dtype
    if dtype == np.dtype('int8'):
        return 'int8'
    elif dtype == np.dtype('uint8'):
        return 'uint8'
    elif dtype == np.dtype('int16'):
        return 'int16'
    elif dtype == np.dtype('uint16'):
        return 'uint16'
    elif dtype == np.dtype('int32'):
        return 'int32'
    elif dtype == np.dtype('uint32'):
        return 'uint32'
    elif dtype == np.dtype('float32'):
        return 'float32'
    elif dtype == np.dtype('float64'):
        return 'float64'
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')

def _format_shape(dataset: h5py.Dataset) -> list:
    """Formats the shape of an HDF5 dataset to a list."""
    shape = dataset.shape
    return [int(dim) for dim in shape]

def _get_dataset_byte_count(dataset: h5py.Dataset) -> int:
    """Returns the number of bytes in an HDF5 dataset."""
    dtype = dataset.dtype
    shape = dataset.shape
    shape_prod = np.prod(shape)
    if dtype == np.dtype('int8'):
        return shape_prod
    elif dtype == np.dtype('uint8'):
        return shape_prod
    elif dtype == np.dtype('int16'):
        return shape_prod * 2
    elif dtype == np.dtype('uint16'):
        return shape_prod * 2
    elif dtype == np.dtype('int32'):
        return shape_prod * 4
    elif dtype == np.dtype('uint32'):
        return shape_prod * 4
    elif dtype == np.dtype('float32'):
        return shape_prod * 4
    elif dtype == np.dtype('float64'):
        return shape_prod * 8
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
