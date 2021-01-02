import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from model import Speech_Transformer
from data_hdf5 import HDF5DatasetGenerator

letters = ['pad', '<', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P',
           'T', 'X', 'Y', '>']

entropy_loss = nn.CrossEntropyLoss(reduction='none')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def save_attention_heads(attentions, tag):
    image = tight_grid(norm_tensor(attentions))  # dim 0 of image_batch is now number of heads
    # batch_plot_path = f'{tag}_{layer}'
    cv2.imwrite(tag + '.png', image * 255)


def masked_cross_entropy_error(targets, logits, mask_value=0):
    targets = targets.contiguous().view(-1, )
    logits = logits.contiguous().view(-1, logits.size()[-1])

    mask = targets.ne(mask_value).long()
    loss = entropy_loss(logits, targets)
    loss = loss * mask

    return torch.sum(loss) / torch.sum(mask)


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


def save_attention_heads(attentions, tag):
    image = tight_grid(norm_tensor(attentions))  # dim 0 of image_batch is now number of heads
    # batch_plot_path = f'{tag}_{layer}'
    cv2.imwrite(tag + '.png', image * 255)


batch_size = 64

train_data = HDF5DatasetGenerator('ocr_train.hdf5', batch_size, 100)
test_data = HDF5DatasetGenerator('ocr_test.hdf5', batch_size, 100)

train_total = train_data.get_total_samples()
test_total = test_data.get_total_samples()

model = Speech_Transformer(256, len(letters)).to(device)

optimizer = torch.optim.Adam(model.parameters(), 1.0e-5, betas=(0.9, 0.98), eps=1e-9)


def padding_data(images, labels, audio_len, text_len):
    max_audio_len = np.max(audio_len)
    max_text_len = np.max(text_len)

    images_batch = images[:, :max_audio_len, :]
    labels_batch = labels[:, :max_text_len]

    images_batch = np.expand_dims(images_batch, -1)

    return images_batch, labels_batch


def train_step(model, images, texts, optimizer, clip=1):
    # mel numpy
    text_inps = texts[:, :-1]
    text_reals = texts[:, 1:]

    # text torch
    text_inps = torch.from_numpy(text_inps).long().to(device)
    text_reals = torch.from_numpy(text_reals).long().to(device)

    # image torch
    images = torch.from_numpy(images).float().to(device)

    optimizer.zero_grad()

    dec_output, _, _, _ = model(images, text_inps)

    loss = masked_cross_entropy_error(text_reals, dec_output)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    return loss.item()


def test_step(model, images, texts):
    # mel numpy
    text_inps = texts[:, :-1]
    text_reals = texts[:, 1:]

    # text torch
    text_inps = torch.from_numpy(text_inps).long().to(device)
    text_reals = torch.from_numpy(text_reals).long().to(device)

    # image torch
    images = torch.from_numpy(images).float().to(device)

    dec_output, en_self_attns, de_self_attns, mutihead_attns = model(images, text_inps)

    save_attention_heads(en_self_attns[0].detach().cpu().numpy(), '_encoder_attention')
    save_attention_heads(de_self_attns[0].detach().cpu().numpy(), '_decoder_attention')
    save_attention_heads(mutihead_attns[0].detach().cpu().numpy(), '_multihead_attention')

    loss = masked_cross_entropy_error(text_reals, dec_output)

    return loss.item()


def decode_index(labels, length):
    text = ''
    labels = labels.astype(np.int64)
    for index in labels[:length]:
        if index < len(letters):
            text += letters[index]
    return text


def decode(array):
    text = ''
    for i in range(len(array)):
        if array[i] < len(letters):
            text += letters[array[i]]

    return text


def decode_index_argmax(labels, length):
    return decode_index(np.argmax(labels, -1), length)


def validate_step(model, images, texts, text_lengths):
    # mel numpy
    text_inps = texts[:, :-1]

    # text torch
    text_inps = torch.from_numpy(text_inps).long().to(device)

    # mel torch
    images = torch.from_numpy(images).float().to(device)

    dec_output, _, _, _ = model(images, text_inps)

    dec_output = F.log_softmax(dec_output, dim=2)

    dec_output = dec_output.cpu().numpy()

    text_length = text_lengths[0][0]
    # ctc_output = ctc_output[0, : text_length - 1]
    dec_output = dec_output[0, : text_length - 1]

    tar_text = decode_index(texts[0, 1:], text_length - 1)
    dec_text = decode_index_argmax(dec_output, text_length - 1)

    return tar_text, dec_text


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


min_loss = float('inf')
path = 'transformer_image2text_torch.pth'

model.load_state_dict(torch.load(path))
model.eval()

for epoch in range(1000):

    train_loss = 0
    test_loss = 0

    train_count = 0
    test_count = 0

    test_images = np.array([])
    test_labels = np.array([])
    text_lengths = np.array([])

    model.train()
    with tqdm(total=train_total) as pbar:
        for images, labels, image_lens, label_lens in train_data.generator():
            images_batch, labels_batch = padding_data(images / 255, labels, image_lens, label_lens)
            ta_loss = train_step(model, images_batch, labels_batch, optimizer)

            train_loss += ta_loss
            train_count += 1

            pbar.update(batch_size)

    model.eval()
    with torch.no_grad():
        with tqdm(total=test_total) as pbar:
            for images, labels, audio_lens, label_lens in test_data.generator():
                images_batch, labels_batch = padding_data(images / 255, labels, audio_lens, label_lens)
                te_loss = test_step(model, images_batch, labels_batch)

                test_loss += te_loss
                test_count += 1

                test_images = images_batch
                test_labels = labels_batch
                text_lengths = label_lens

                # text = predict(model, test_audios)
                # print('\n', decode(text[0]))
                # print('\n', decode(test_labels[0].astype(np.int32)))

                pbar.update(batch_size)

        if float(test_loss) / test_count < min_loss:
            torch.save(model.state_dict(), path)
            print('\nSaving checkpoint ')
            min_loss = float(test_loss) / test_count

        tar_text, dec_text = validate_step(model, test_images, test_labels, text_lengths)
        print('\ntar_text ' + str(tar_text))
        print('\ndec_text ' + str(dec_text))
        print('\n')

    if train_count == 0:
        train_count += 1
    if test_count == 0:
        test_count += 1

    print('\nEpoch {} Train Loss {:.4f} Test Loss {:.4f}'.format(epoch + 1, float(train_loss) / train_count,
                                                                 float(test_loss) / test_count))
    print('\n')
