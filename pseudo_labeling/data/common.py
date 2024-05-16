"""
from ubteach
"""
import logging
from detectron2.data.common import MapDataset, AspectRatioGroupedDataset

class AspectRatioGroupedSemiSupDatasetTwoCrop(AspectRatioGroupedDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: a tuple containing two iterable generators. （labeled and unlabeled data)
               Each element must be a dict with keys "width" and "height", which will be used
               to batch data.
            batch_size (int):
        """

        self.label_dataset, self.unlabel_dataset = dataset
        self.batch_size_label = batch_size[0]
        self.batch_size_unlabel = batch_size[1]

        self._label_buckets = [[] for _ in range(2)]
        self._label_buckets_key = [[] for _ in range(2)]
        self._unlabel_buckets = [[] for _ in range(2)]
        self._unlabel_buckets_key = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):

        label_bucket, unlabel_bucket = [], []

        for d_label, d_unlabel in zip(self.label_dataset, self.unlabel_dataset):
            
            if len(label_bucket) != self.batch_size_label:
                w, h = d_label["width"], d_label["height"]
                label_bucket_id = 0 if w > h else 1
                label_bucket.append(d_label)
                label_bucket_key = self._label_buckets_key[label_bucket_id]
                label_bucket_key.append(d_label)

            if len(unlabel_bucket) != self.batch_size_unlabel*2:
                w, h = d_unlabel[0]["width"], d_unlabel[0]["height"]
                unlabel_bucket_id = 0 if w > h else 1
                unlabel_bucket.append(d_unlabel[0])
                unlabel_bucket_key = self._unlabel_buckets_key[unlabel_bucket_id]
                unlabel_bucket_key.append(d_unlabel[1])
                print("#########################")
                print(unlabel_bucket)
                print("-----------------------------")
                print(unlabeled_bucket_ley)



            print(stop)

    """
                label_bucket_id = 0 if w > h else 1
                label_bucket = self._label_buckets[label_bucket_id]
                label_bucket.append(d_label)
                label_buckets_key = self._label_buckets_key[label_bucket_id]
                label_buckets_key.append(d_label)

            if len(unlabel_bucket) != self.batch_size_unlabel:
                w, h = d_unlabel[0]["width"], d_unlabel[0]["height"]
                unlabel_bucket_id = 0 if w > h else 1
                unlabel_bucket = self._unlabel_buckets[unlabel_bucket_id]
                unlabel_bucket.append(d_unlabel)
                unlabel_buckets_key = self._unlabel_buckets_key[unlabel_bucket_id]
                unlabel_buckets_key.append(d_unlabel)

            # yield the batch of data until all buckets are full
            if (
                len(label_bucket) == self.batch_size_label
                and len(unlabel_bucket) == self.batch_size_unlabel
            ):
                # label_strong, label_weak, unlabed_strong, unlabled_weak
                yield (
                    label_bucket[:],
                    label_buckets_key[:],
                    unlabel_bucket[:],
                    unlabel_buckets_key[:],
                )
                del label_bucket[:]
                del label_buckets_key[:]
                del unlabel_bucket[:]
                del unlabel_buckets_key[:]
    """