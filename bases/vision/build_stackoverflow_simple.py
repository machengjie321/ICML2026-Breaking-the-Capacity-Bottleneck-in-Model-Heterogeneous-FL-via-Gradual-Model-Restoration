# build_stackoverflow_simple.py
import os
from typing import Dict

import torch


SAVE_DIR = "./datasets/stackoverflow/"
MAX_PER_USER = 50
MAX_LEN = 128
VOCAB_LIMIT = 10000
MAX_USERS_TRAIN = 6000
MAX_USERS_VAL = 1500
SEED = 31


def _import_tff():
    try:
        import tensorflow as tf
        import tensorflow_federated as tff
    except ImportError as exc:
        raise RuntimeError(
            "StackOverflow preprocessing requires TensorFlow and TensorFlow Federated. "
            "Install them first, then rerun the same command and preprocessing will continue automatically."
        ) from exc

    tf.random.set_seed(SEED)
    return tf, tff


def get_stackoverflow():
    _, tff = _import_tff()
    print("Loading StackOverflow dataset from TensorFlow Federated ...")
    return tff.simulation.datasets.stackoverflow.load_data()


def tf_string_to_tokens(s, vocab: Dict[str, int], max_len: int = MAX_LEN):
    text = s.numpy().decode("utf-8")
    tokens = text.split()
    ids = []
    for tok in tokens[:max_len]:
        if tok in vocab:
            ids.append(vocab[tok])
        elif len(vocab) < VOCAB_LIMIT:
            vocab[tok] = len(vocab)
            ids.append(vocab[tok])
        else:
            ids.append(vocab["<unk>"])
    return ids


def export_split(ds, save_path, vocab, max_per_user=MAX_PER_USER, max_len=MAX_LEN, max_users=None):
    client_ids = ds.client_ids
    print(f"{len(client_ids)} clients detected in split.")
    x_all, y_all, user_ids = [], [], []
    total = 0

    if max_users is not None:
        client_ids = client_ids[:max_users]

    for uid_int, cid in enumerate(client_ids):
        c_ds = ds.create_tf_dataset_for_client(cid)
        num = 0
        for ex in c_ds:
            ids = tf_string_to_tokens(ex["tokens"], vocab, max_len)
            x_all.append(torch.tensor(ids, dtype=torch.long))
            y_all.append(torch.tensor(0, dtype=torch.long))
            user_ids.append(uid_int)

            num += 1
            total += 1
            if max_per_user and num >= max_per_user:
                break

        if total and total % 5000 == 0:
            print(f"processed {total} samples ...")

    print(f"Finished {save_path}: {total} samples, vocab size={len(vocab)}, users={len(set(user_ids))}")
    torch.save({"x": x_all, "y": y_all, "user_ids": user_ids}, save_path)
    return vocab


def ensure_stackoverflow_preprocessed(save_dir=SAVE_DIR, force=False):
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(save_dir, "stackoverflow_train.pt")
    val_path = os.path.join(save_dir, "stackoverflow_val.pt")
    vocab_path = os.path.join(save_dir, "stackoverflow_vocab.pt")

    if not force and all(os.path.exists(p) for p in (train_path, val_path, vocab_path)):
        return {
            "train": train_path,
            "val": val_path,
            "vocab": vocab_path,
        }

    so_train, _, so_test = get_stackoverflow()
    vocab = {"<pad>": 0, "<unk>": 1}

    vocab = export_split(
        so_train,
        train_path,
        vocab,
        max_per_user=MAX_PER_USER,
        max_len=MAX_LEN,
        max_users=MAX_USERS_TRAIN,
    )
    vocab = export_split(
        so_test,
        val_path,
        vocab,
        max_per_user=MAX_PER_USER,
        max_len=MAX_LEN,
        max_users=MAX_USERS_VAL,
    )

    print(f"Final vocab size = {len(vocab)} (limit = {VOCAB_LIMIT})")
    torch.save(vocab, vocab_path)
    print(f"All saved to {save_dir}")

    return {
        "train": train_path,
        "val": val_path,
        "vocab": vocab_path,
    }


if __name__ == "__main__":
    ensure_stackoverflow_preprocessed()
