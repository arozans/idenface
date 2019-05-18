from pathlib import Path

import tensorflow as tf
from tensorflow.python.data import TFRecordDataset

from src.data.common_types import DatasetSpec, DatasetStorageMethod
from src.data.tfrecord.reading.tfrecord_readers import AbstractReader, PairedDecodingReader, UnpairedDecodingReader, \
    PairedNotDecodingReader, UnpairedNotDecodingReader
from src.utils import utils


def assemble_dataset(input_data_dir: Path, dataset_spec: DatasetSpec) -> TFRecordDataset:
    def all_names_in_dir(dir):
        return [str(f) for f in dir.iterdir()][0]  # only one file atm

    files_to_assemble = all_names_in_dir(input_data_dir)

    utils.log('Assembling dataset from .tfrecord file(s): {}, dataset spec: {}'.format(files_to_assemble, dataset_spec))
    dataset = tf.data.TFRecordDataset(filenames=files_to_assemble)

    reader = resolve_reader(dataset_spec)

    dataset = dataset.map(reader.get_decode_op(), num_parallel_calls=64)

    return dataset


def resolve_reader(dataset_spec: DatasetSpec) -> AbstractReader:
    storage_method = dataset_spec.raw_data_provider.description.storage_method
    tf_readers = {
        (False, False): UnpairedNotDecodingReader,
        (True, False): PairedNotDecodingReader,
        (True, True): PairedDecodingReader,
        (False, True): UnpairedDecodingReader
    }
    encoded_dataset = storage_method == DatasetStorageMethod.ON_DISC or dataset_spec.encoding
    return tf_readers[(dataset_spec.paired, encoded_dataset)]()
