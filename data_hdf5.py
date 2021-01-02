import os
import h5py
import numpy as np
import random


class HDF5DatasetGenerator:
    def __init__(self, db_path, batch_size, number=None):
        self.batchSize = batch_size
        self.db = h5py.File(db_path)
        self.numAudios = self.db["labels"].shape[0]
        self.indexes = []
        for i in range(self.numAudios):
            self.indexes.append(i)
        if number is not None:
            self.indexes = self.indexes[: number]

    def get_total_samples(self):
        return len(self.indexes)

    def get_random_sample(self):
        random.shuffle(self.indexes)
        i = self.indexes[0]
        audios = self.db["audios"][self.indexes[i]: self.indexes[i] + self.batchSize]
        labels = self.db["labels"][self.indexes[i]: self.indexes[i] + self.batchSize]
        audio_lens = self.db["audio_lens"][self.indexes[i]: self.indexes[i] + self.batchSize]
        label_lens = self.db["label_lens"][self.indexes[i]: self.indexes[i] + self.batchSize]
        return np.array(audios), np.array(labels), np.array(audio_lens), np.array(label_lens)

    def generator(self):
        random.shuffle(self.indexes)
        for i in range(0, len(self.indexes), self.batchSize):
            audios = self.db["audios"][self.indexes[i]: self.indexes[i] + self.batchSize]
            labels = self.db["labels"][self.indexes[i]: self.indexes[i] + self.batchSize]
            audio_lens = self.db["audio_lens"][self.indexes[i]: self.indexes[i] + self.batchSize]
            label_lens = self.db["label_lens"][self.indexes[i]: self.indexes[i] + self.batchSize]
            yield np.array(audios), np.array(labels), np.array(audio_lens), np.array(label_lens)

    def close(self):
        self.db.close()
