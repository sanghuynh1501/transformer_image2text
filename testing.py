import cv2
import numpy as np

import torch

from model import Speech_Transformer
from data_hdf5 import HDF5DatasetGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

min_loss = float('inf')
path = 'transformer_image2text_torch.pth'

letters = ['pad', '<', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P',
           'T', 'X', 'Y', '>']


def norm_tensor(tensor):
    return np.divide(
        np.subtract(
            tensor,
            np.min(tensor)
        ),
        np.subtract(
            np.max(tensor),
            np.min(tensor)
        )
    )


def tight_grid(images):
    images = np.array(images)
    images = np.pad(images, [[0, 0], [1, 1], [1, 1]], 'constant', constant_values=1)  # add borders
    if len(images.shape) != 3:
        raise Exception
    else:
        n, y, x = images.shape
    ratio = y / x
    if ratio > 1:
        ny = max(int(np.sqrt(n / ratio)), 1)
        nx = int(n / ny)
        nx += n - (nx * ny)
        extra = nx * ny - n
    else:
        nx = max(int(np.sqrt(n * ratio)), 1)
        ny = int(n / nx)
        ny += n - (nx * ny)
        extra = nx * ny - n
    tot = np.append(images, np.zeros((extra, y, x)), axis=0)
    img = np.block([[*tot[i * nx:(i + 1) * nx]] for i in range(ny)])
    return img


def decode(array):
    text = ''
    for i in range(len(array)):
        if len(letters) > array[i] > 0:
            text += letters[array[i]]

    return text


def predict(model, images):
    image = np.expand_dims(images[0], 0)

    text_input = np.ones((1, 1), np.int32)

    image = torch.from_numpy(image).float().cuda()
    output = torch.from_numpy(text_input).long().cuda()

    while True:
        final_output, _, _, _ = model(image, output)
        final_output = final_output[:1, -1:]
        final_output = torch.argmax(final_output, axis=-1)
        if final_output.item() == len(letters) - 1:
            break
        output = torch.cat((output, final_output), 1)

    return output.cpu().numpy()


def padding_data(images, labels, audio_len, text_len):
    max_audio_len = np.max(audio_len)
    max_text_len = np.max(text_len)

    images_batch = images[:, :max_audio_len, :]
    labels_batch = labels[:, :max_text_len]

    images_batch = np.expand_dims(images_batch, -1)

    return images_batch, labels_batch


def save_attention_heads(attentions, tag):
    image = tight_grid(norm_tensor(attentions))  # dim 0 of image_batch is now number of heads
    cv2.imwrite(tag + '.png', image * 255)


batch_size = 64

test_data = HDF5DatasetGenerator('ocr_test.hdf5', batch_size)

test_total = test_data.get_total_samples()

model = Speech_Transformer(256, len(letters)).to(device)

optimizer = torch.optim.Adam(model.parameters(), 1.0e-5, betas=(0.9, 0.98), eps=1e-9)

model.load_state_dict(torch.load(path))
model.eval()


test_images = np.array([])
test_labels = np.array([])
text_lengths = np.array([])

step = 1
model.eval()
with torch.no_grad():
    for images, labels, audio_lens, label_lens in test_data.generator():
        images_batch, labels_batch = padding_data(images / 255, labels, audio_lens, label_lens)

        test_images = images_batch
        test_labels = labels_batch
        text_lengths = label_lens

        text = predict(model, test_images)
        print('\nTest ' + str(step) + ' ============================================================================')
        print(decode(text[0]))
        print(decode(test_labels[0].astype(np.int32)))

        step += 1
