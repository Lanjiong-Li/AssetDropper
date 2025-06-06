import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from typing import Literal, Tuple
import torch.utils.data as data
import numpy as np
import cv2
import torch

class AssetDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        size: Tuple[int, int] = (512, 512),
        txt_name: str = None,
    ):
        super(AssetDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.txt_name = txt_name

        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.transform2D = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        self.toTensor = transforms.ToTensor()

        image_names = []
        caption_names = []
        dataroot_names = []


        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}.txt")
        else:
            if txt_name is None:
                filename = os.path.join(dataroot_path, f"{phase}.txt")
            else:
                filename = os.path.join(dataroot_path, f"{txt_name}.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                
                image_name = line.strip()

                name_no_ext, _ = os.path.splitext(image_name)
                caption_name = name_no_ext + ".txt"

                image_names.append(image_name)
                caption_names.append(caption_name)
                dataroot_names.append(dataroot_path)

        self.image_names = image_names
        self.caption_names = caption_names
        self.dataroot_names = dataroot_names
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        self.clip_processor = CLIPImageProcessor()
    
    def _crop_and_resize_by_mask(
        self,
        image: Image.Image,
        mask: Image.Image,
        output_size=(512, 512)
    ) -> Tuple[Image.Image, Image.Image]:

        mask_np = np.array(mask.convert("L"))
        if mask_np.max() == 0:
            return image.resize(output_size), mask.resize(output_size)

        ys, xs = np.nonzero(mask_np)
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        box_width = max_x - min_x
        box_height = max_y - min_y
        box_size = max(box_width, box_height)

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        half_size = box_size // 2

        left = max(center_x - half_size, 0)
        upper = max(center_y - half_size, 0)
        right = min(center_x + half_size, image.width)
        lower = min(center_y + half_size, image.height)

        if right - left < box_size:
            if left == 0:
                right = min(left + box_size, image.width)
            else:
                left = max(right - box_size, 0)

        if lower - upper < box_size:
            if upper == 0:
                lower = min(upper + box_size, image.height)
            else:
                upper = max(lower - box_size, 0)

        crop_box = (left, upper, right, lower)

        cropped_image = image.crop(crop_box).resize(output_size, resample=Image.BICUBIC)
        cropped_mask = mask.crop(crop_box).resize(output_size, resample=Image.NEAREST)

        return cropped_image, cropped_mask

    def __getitem__(self, index):
        image_name = self.image_names[index]
        caption_name = self.caption_names[index]

        #1 image
        image = Image.open(os.path.join(self.dataroot, "Image", image_name))

        if image.mode == 'RGBA':
            white_bg = Image.new("RGB", image.size, (255, 255, 255))
            white_bg.paste(image, (0, 0), image)
            image = white_bg
        else:
            image = image.convert('RGB')

        image = image.resize((512, 512))

        mask_name_without_ext = os.path.splitext(image_name)[0]
        print(f"mask_name_without_ext:{mask_name_without_ext}")
        
        possible_ext = ['.jpg', '.png']

        for ext in possible_ext:
            test_path = os.path.join(self.dataroot, "Mask", mask_name_without_ext + ext)
            if os.path.exists(test_path):
                mask_path = test_path
                break

        if mask_path is None:
            raise FileNotFoundError(f"Missing Mask: {image_name}")

        #2 mask
        mask = Image.open(mask_path).resize((512,512))
        
        image, mask = self._crop_and_resize_by_mask(image, mask, output_size=(512, 512))
        
        #3 pattern
        pattern = self.toTensor(image)
        
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask_cv = np.array(mask.convert("L"))

        #4 masked_image for IP-Adapter
        masked_image_cv = cv2.bitwise_and(image_cv, image_cv, mask=mask_cv)
        masked_image = Image.fromarray(cv2.cvtColor(masked_image_cv, cv2.COLOR_BGR2RGB)).resize((512, 512))
        mask_img_trim = self.clip_processor(images=masked_image, return_tensors="pt").pixel_values

        #5 edgemap
        image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask_cv, kernel, iterations=3)
        sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
        gradient[eroded_mask == 0] = 0
        edgemap = Image.fromarray(gradient).resize((512, 512))

        mask = self.toTensor(mask)
        edgemap = self.toTensor(edgemap)
        mask = mask[:1]
        edgemap = edgemap[:1]
        
        pattern = self.norm(pattern)
        image = self.transform(image) #norm [-1, 1]
        
        #caption
        with open(f"{self.dataroot}/Caption/{caption_name}","r") as f:
            caption = f.readline().strip()

        result = {}
        
        result["image_name"] = image_name
        result["image"] = image 
        result["mask"] = mask
        result["edgemap"] = edgemap
        result["masked_image"] = mask_img_trim
        result["pattern"] = pattern 
        result["caption_pattern"] = f"The pattern is {caption}"
        result["caption_gen"] = f"A normalized square pattern of {caption}"

        return result

    def __len__(self):
        return len(self.image_names)