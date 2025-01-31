from keras.utils import to_categorical

import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from pathlib import Path


nClasses = 2


def get_label(df: pd.DataFrame, filename: Path):
    # Filename column can take two different names in csv files
    try:
        label = df[df.hdf5.str.match(filename.stem)]["label"]
    except AttributeError:
        label = df[df.h5.str.match(filename.stem)]["label"]

    if label.shape[0] == 0:
        msg = f"No label found for {fname}"
        raise ValueError(msg)

    return label


def load_img(filename: Path):
    with h5py.File(filename, "r") as f:
        try:
            dm_t = np.array(f["cand/ml/dm_time"])
            fq_t = np.array(f["cand/ml/freq_time"]).T
        except KeyError:
            # Try alternative candidate hdf5 format
            dm_t = np.array(f["data_dm_time"])
            fq_t = np.array(f["data_freq_time"]).T

    return np.stack((dm_t, fq_t), axis=-1) / 255.0


if __name__ == "__main__":
    data_dir = Path("data")

    ds_name = "train"
    label_name = "train"
    train_data = data_dir / ds_name
    train_labels = data_dir / "csv_labels" / f"{label_name}.csv"

    labels_df = pd.read_csv(train_labels)

    ds = None
    imgs, labels, fnames = [], [], []
    for idx, fname in enumerate(train_data.iterdir()):
        if idx % 500 == 0:
            print(f"Processing candidate {idx}")
        if fname.suffix != ".hdf5":
            continue

        imgs.append(load_img(filename=fname))
        labels.append(get_label(df=labels_df, filename=fname))
        fnames.append(str(fname.stem))

        if ds is None:
            ds = tf.data.Dataset.from_tensor_slices(
                (imgs, to_categorical(labels, nClasses), fnames)
            )
            imgs, labels, fnames = [], [], []

        if idx % 100 == 0 and idx > 0:
            ds = ds.concatenate(
                tf.data.Dataset.from_tensor_slices(
                    (imgs, to_categorical(labels, nClasses), fnames)
                )
            )
            imgs, labels, fnames = [], [], []

    if len(imgs) > 0 and len(labels) > 0:
        ds = ds.concatenate(
            tf.data.Dataset.from_tensor_slices(
                (imgs, to_categorical(labels, nClasses), fnames)
            )
        )

    ds.save(str(data_dir / "tf_ds" / label_name))
