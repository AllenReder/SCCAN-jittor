import random
import math
import numpy as np
import numbers
import collections
import cv2

import jittor as jt

manual_seed = 123
jt.set_global_seed(manual_seed)
np.random.seed(manual_seed)
random.seed(manual_seed)


class Compose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label, label2):
        for t in self.segtransform:
            image, label, label2 = t(image, label, label2)
        return image, label, label2


import time


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a jittor.Var of shape (C x H x W).
    def __call__(self, image, label, label2):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                              "[eg: data readed by cv2.imread()].\n")
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n")
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n")

        image = jt.array(image.transpose((2, 0, 1)))
        if not image.dtype == jt.float32:
            image = image.float32()
        label = jt.array(label)
        label2 = jt.array(label2)
        if not label.dtype == jt.int64:
            label = label.int64()
            label2 = label2.int64()
        return image, label, label2


class ToNumpy(object):
    # Converts jittor.Var of shape (C x H x W) to a numpy.ndarray (H x W x C).
    def __call__(self, image, label, label2):
        if not isinstance(image, jt.Var) or not isinstance(label, jt.Var):
            raise RuntimeError("segtransform.ToNumpy() only handle jittor.Var")

        image = image.numpy().transpose((1, 2, 0))
        if not image.dtype == np.uint8:
            image = image.astype(np.uint8)
        label = label.numpy()
        label2 = label2.numpy()
        if len(label.shape) == 3:
            label = label.transpose((1, 2, 0))
            label2 = label2.transpose((1, 2, 0))
        if not label.dtype == np.uint8:
            label = label.astype(np.uint8)
            label2 = label2.astype(np.uint8)
        return image, label, label2


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label, label2):
        if self.std is None:
            for i, m in enumerate(self.mean):
                image[i] = image[i] - m
        else:
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                image[i] = (image[i] - m) / s
        return image, label, label2


class UnNormalize(object):
    # UnNormalize tensor with mean and standard deviation along channel: channel = (channel * std) + mean
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label, label2):
        if self.std is None:
            for i, m in enumerate(self.mean):
                image[i] = image[i] + m
        else:
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                image[i] = image[i] * s + m
        return image, label, label2


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label, label2):

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        def find_new_hw(ori_h, ori_w, test_size):
            if ori_h >= ori_w:
                ratio = test_size * 1.0 / ori_h
                new_h = test_size
                new_w = int(ori_w * ratio)
            elif ori_w > ori_h:
                ratio = test_size * 1.0 / ori_w
                new_h = int(ori_h * ratio)
                new_w = test_size

            if new_h % 8 != 0:
                new_h = (int(new_h / 8)) * 8
            else:
                new_h = new_h
            if new_w % 8 != 0:
                new_w = (int(new_w / 8)) * 8
            else:
                new_w = new_w
            return new_h, new_w

        test_size = self.size
        new_h, new_w = find_new_hw(image.shape[0], image.shape[1], test_size)
        # new_h, new_w = test_size, test_size
        image_crop = cv2.resize(image, dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        back_crop = np.zeros((test_size, test_size, 3))
        # back_crop[:,:,0] = mean[0]
        # back_crop[:,:,1] = mean[1]
        # back_crop[:,:,2] = mean[2]
        back_crop[:new_h, :new_w, :] = image_crop
        image = back_crop

        s_mask = label
        new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
        s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_NEAREST)
        back_crop_s_mask = np.ones((test_size, test_size)) * 255
        back_crop_s_mask[:new_h, :new_w] = s_mask
        label = back_crop_s_mask

        s_mask2 = label2
        new_h2, new_w2 = find_new_hw(s_mask2.shape[0], s_mask2.shape[1], test_size)
        s_mask2 = cv2.resize(s_mask2.astype(np.float32), dsize=(int(new_w2), int(new_h2)),
                             interpolation=cv2.INTER_NEAREST)
        back_crop_s_mask2 = np.ones((test_size, test_size)) * 255
        back_crop_s_mask2[:new_h2, :new_w2] = s_mask2
        label2 = back_crop_s_mask2

        # image = cv2.resize(image, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        # label = cv2.resize(label, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)
        # label2 = cv2.resize(label2, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return image, label, label2


class test_Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label, label2):

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        def find_new_hw(ori_h, ori_w, test_size):
            if max(ori_h, ori_w) > test_size:
                if ori_h >= ori_w:
                    ratio = test_size * 1.0 / ori_h
                    new_h = test_size
                    new_w = int(ori_w * ratio)
                elif ori_w > ori_h:
                    ratio = test_size * 1.0 / ori_w
                    new_h = int(ori_h * ratio)
                    new_w = test_size

                if new_h % 8 != 0:
                    new_h = (int(new_h / 8)) * 8
                else:
                    new_h = new_h
                if new_w % 8 != 0:
                    new_w = (int(new_w / 8)) * 8
                else:
                    new_w = new_w
                return new_h, new_w
            else:
                return ori_h, ori_w

        test_size = self.size
        new_h, new_w = find_new_hw(image.shape[0], image.shape[1], test_size)
        if new_w != image.shape[0] or new_h != image.shape[1]:
            image_crop = cv2.resize(image, dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        else:
            image_crop = image.copy()
        back_crop = np.zeros((test_size, test_size, 3))
        back_crop[:new_h, :new_w, :] = image_crop
        image = back_crop

        s_mask = label
        new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
        if new_w != s_mask.shape[0] or new_h != s_mask.shape[1]:
            s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)),
                                interpolation=cv2.INTER_NEAREST)
        back_crop_s_mask = np.ones((test_size, test_size)) * 255
        back_crop_s_mask[:new_h, :new_w] = s_mask
        label = back_crop_s_mask

        s_mask2 = label2
        new_h2, new_w2 = find_new_hw(s_mask2.shape[0], s_mask2.shape[1], test_size)
        if new_w2 != s_mask.shape[0] or new_h2 != s_mask2.shape[1]:
            s_mask2 = cv2.resize(s_mask2.astype(np.float32), dsize=(int(new_w2), int(new_h2)),
                                 interpolation=cv2.INTER_NEAREST)
        back_crop_s_mask2 = np.ones((test_size, test_size)) * 255
        back_crop_s_mask2[:new_h2, :new_w2] = s_mask2
        label2 = back_crop_s_mask2

        return image, label, label2


class Direct_Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label, label2):
        test_size = self.size

        image = cv2.resize(image, dsize=(test_size, test_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label.astype(np.float32), dsize=(test_size, test_size), interpolation=cv2.INTER_NEAREST)
        label2 = cv2.resize(label2.astype(np.float32), dsize=(test_size, test_size), interpolation=cv2.INTER_NEAREST)

        return image, label, label2


class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise RuntimeError("segtransform.RandScale() scale param error.\n")
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise RuntimeError("segtransform.RandScale() aspect_ratio param error.\n")

    def __call__(self, image, label, label2):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        label2 = cv2.resize(label2, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label, label2


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """

    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        self.size = size
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise RuntimeError("crop size error.\n")
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise RuntimeError("crop type error: rand | center\n")
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise RuntimeError("padding in Crop() should be a number list\n")
            if len(padding) != 3:
                raise RuntimeError("padding channel is not equal with 3\n")
        else:
            raise RuntimeError("padding in Crop() should be a number list\n")
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise RuntimeError("ignore_label should be an integer number\n")

    def __call__(self, image, label, label2):
        h, w = label.shape

        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise RuntimeError("segtransform.Crop() need padding while padding argument is None\n")
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.ignore_label)
            label2 = cv2.copyMakeBorder(label2, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                        cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        raw_label = label
        raw_label2 = label2
        raw_image = image

        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        label2 = label2[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        raw_pos_num = np.sum(raw_label == 1)
        pos_num = np.sum(label == 1)
        crop_cnt = 0
        while (pos_num < 0.85 * raw_pos_num and crop_cnt <= 30):
            image = raw_image
            label = raw_label
            label2 = raw_label2
            if self.crop_type == 'rand':
                h_off = random.randint(0, h - self.crop_h)
                w_off = random.randint(0, w - self.crop_w)
            else:
                h_off = int((h - self.crop_h) / 2)
                w_off = int((w - self.crop_w) / 2)
            image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
            label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
            label2 = label2[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
            raw_pos_num = np.sum(raw_label == 1)
            pos_num = np.sum(label == 1)
            crop_cnt += 1
        if crop_cnt >= 50:
            image = cv2.resize(raw_image, (self.size[0], self.size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(raw_label, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)
            label2 = cv2.resize(raw_label2, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)

        if image.shape != (self.size[0], self.size[0], 3):
            image = cv2.resize(image, (self.size[0], self.size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)
            label2 = cv2.resize(label2, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)

        return image, label, label2


class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise RuntimeError("segtransform.RandRotate() scale param error.\n")
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise RuntimeError("padding in RandRotate() should be a number list\n")
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label, label2):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.ignore_label)
            label2 = cv2.warpAffine(label2, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=self.ignore_label)
        return image, label, label2


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, label2):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
            label2 = cv2.flip(label2, 1)
        return image, label, label2


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, label2):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
            label2 = cv2.flip(label2, 0)
        return image, label, label2


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label, label2):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label, label2


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label, label2):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label, label2


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label, label2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label, label2