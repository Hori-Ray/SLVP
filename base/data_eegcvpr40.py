import os
import gc
import copy
import bisect
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open_clip

from base.utils import instantiate_from_config, get_device

def _to_float_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


def _average_img_feature_entries(entries):
    """
    支持:
    1. tensor[D]
    2. tensor[2,D]
    3. {"feat": tensor[2,D], "stat": tensor[6], "ctx_feat": tensor[3,D]}
    """
    first = entries[0]

    if isinstance(first, dict):
        feats, stats, ctx_feats = [], [], []

        for e in entries:
            feats.append(_to_float_tensor(e["feat"]))

            if "stat" in e:
                stats.append(_to_float_tensor(e["stat"]))
            else:
                stats.append(torch.zeros(6, dtype=torch.float32))

            if "ctx_feat" in e:
                ctx_feats.append(_to_float_tensor(e["ctx_feat"]))
            else:
                d = feats[-1].shape[-1]
                ctx_feats.append(torch.zeros(3, d, dtype=torch.float32))

        return {
            "feat": torch.stack(feats, dim=0).mean(dim=0),
            "stat": torch.stack(stats, dim=0).mean(dim=0),
            "ctx_feat": torch.stack(ctx_feats, dim=0).mean(dim=0),
        }

    feats = [_to_float_tensor(e) for e in entries]
    return torch.stack(feats, dim=0).mean(dim=0)


def load_eegcvpr40_data(config):
    exp_setting = config.get("exp_setting", "intra-subject")

    if exp_setting == "intra-subject":
        train_dataset = EEGCVPR40Dataset(config, mode="train")
        val_dataset = EEGCVPR40Dataset(config, mode="val")
        test_dataset = EEGCVPR40Dataset(config, mode="test")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["train_batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["data"]["val_batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["data"]["test_batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader

    elif exp_setting == "inter-subject":
        test_subjects = config["data"]["subjects"]
        all_subjects = [f"sub-{i:02d}" for i in range(1, 7)]
        train_subjects = [s for s in all_subjects if s not in test_subjects]

        test_dataset = EEGCVPR40Dataset(config, mode="test")

        train_cfg = copy.deepcopy(config)
        train_cfg["data"]["subjects"] = train_subjects

        train_dataset = EEGCVPR40Dataset(train_cfg, mode="train")
        val_dataset = EEGCVPR40Dataset(train_cfg, mode="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["train_batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["data"]["val_batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["data"]["test_batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader

    else:
        raise ValueError(f"Unsupported exp_setting: {exp_setting}")


class EEGCVPR40Dataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.name = config["name"]
        self.data_dir = config["data"]["data_dir"]
        self.subjects = config["data"]["subjects"]
        self.model_type = config["data"]["model_type"]
        self.timesteps = config["data"]["timesteps"]
        self.avg = config["data"].get(f"{mode}_avg", True)

        self.data_paths = [os.path.join(self.data_dir, s, f"{mode}.pt") for s in self.subjects]
        self.loaded_data = [self.load_data(p) for p in self.data_paths]

        # EEGCVPR40 各被试样本数不同，不能假设固定长度
        self.subject_lengths = [d["eeg"].shape[0] for d in self.loaded_data]
        self.subject_offsets = np.cumsum([0] + self.subject_lengths).tolist()
        self.trial_all_subjects = int(sum(self.subject_lengths))

        self.c = config["c"]
        self.uncertainty_aware = config["data"]["uncertainty_aware"]

        if self.uncertainty_aware:
            self.blur_transform = {}
            for shift, tag in zip([-self.c, 0, self.c], ["low", "medium", "high"]):
                blur_param = copy.deepcopy(config["data"]["blur_type"])
                blur_param["params"]["blur_kernel_size"] = blur_param["params"]["blur_kernel_size"] + shift
                self.blur_transform[tag] = instantiate_from_config(blur_param)
        else:
            self.blur_transform = instantiate_from_config(config["data"]["blur_type"])

        self.process_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        self.match_label = np.ones(self.trial_all_subjects, dtype=int)

        feat_dir = os.path.join(
            self.data_dir,
            "..",
            "Image_feature",
            config["data"]["blur_type"]["target"].rsplit(".", 1)[-1]
        )
        os.makedirs(feat_dir, exist_ok=True)
        features_filename = os.path.join(feat_dir, f"{self.name}_{mode}.pt")

        # 收集当前 mode 下真正需要的所有图片和文本 key
        self.required_images = sorted(list(set(
            str(img).replace("\\", "/")
            for d in self.loaded_data
            for img in d["img"]
        )))
        self.required_texts = sorted(list(set(
            str(txt)
            for d in self.loaded_data
            for txt in d["text"]
        )))

        cache_loaded = False

        if os.path.exists(features_filename):
            try:
                saved = torch.load(features_filename, weights_only=False)
                self.img_features = saved["img_features"]
                self.text_features = saved["text_features"]

                if self._cache_is_complete():
                    cache_loaded = True
                    print(f"[INFO] Loaded feature cache: {features_filename}")
                else:
                    print(f"[WARN] Incomplete or incompatible cache detected: {features_filename}")
                    print("[WARN] Removing and rebuilding cache...")
                    try:
                        os.remove(features_filename)
                    except OSError:
                        pass
                    cache_loaded = False

            except Exception as e:
                print(f"[WARN] Failed to load feature cache: {features_filename}")
                print(f"[WARN] Error: {repr(e)}")
                print("[WARN] Removing corrupted cache and rebuilding...")
                try:
                    os.remove(features_filename)
                except OSError:
                    pass
                cache_loaded = False

        if not cache_loaded:
            self._build_and_save_cache(features_filename)

    def _unwrap_singleton(self, x):
        """
        把 [[str]] / [str] / np.array([str], dtype=object) 这种嵌套解包成标量。
        """
        while True:
            if isinstance(x, (list, tuple)) and len(x) == 1:
                x = x[0]
                continue
            if isinstance(x, np.ndarray) and x.size == 1:
                x = x.reshape(-1)[0]
                continue
            break
        return x

    def _cache_entry_is_valid(self, entry):
        """
        检查当前图像缓存是否满足最终版 contextual adapter 的需要。
        """
        need_original = bool(
            self.config["data"].get("return_original_feature_for_calibration", False)
        )
        need_stats = bool(
            self.config["data"].get("return_awmb_stats_for_calibration", False)
        )
        need_context = bool(
            self.config["data"].get("return_awmb_context_for_teacher", False)
        )

        # 如果不需要最终版额外字段，老 tensor 缓存也可以用
        if not (need_original or need_stats or need_context):
            return isinstance(entry, torch.Tensor) or isinstance(entry, dict)

        # 最终版必须是 dict
        if not isinstance(entry, dict):
            return False

        if "feat" not in entry:
            return False

        feat = entry["feat"]
        if not isinstance(feat, torch.Tensor):
            return False

        # contextual adapter 需要 fused + original，即 [2,D]
        if need_original:
            if feat.dim() != 2 or feat.shape[0] < 2:
                return False

        if need_stats:
            if "stat" not in entry:
                return False
            stat = entry["stat"]
            if not isinstance(stat, torch.Tensor):
                return False
            if stat.numel() < 6:
                return False

        if need_context:
            if "ctx_feat" not in entry:
                return False
            ctx = entry["ctx_feat"]
            if not isinstance(ctx, torch.Tensor):
                return False
            if ctx.dim() != 2 or ctx.shape[0] < 3:
                return False

        return True

    def _cache_is_complete(self):
        # 文本缓存检查
        if not isinstance(self.text_features, dict):
            return False

        for t in self.required_texts:
            if t not in self.text_features:
                print(f"[CACHE MISS][TEXT] {t}")
                return False

        # 图像缓存检查
        if self.uncertainty_aware:
            if not isinstance(self.img_features, dict):
                return False

            for tag in ["low", "medium", "high"]:
                if tag not in self.img_features or not isinstance(self.img_features[tag], dict):
                    print(f"[CACHE MISS][TAG DICT] {tag}")
                    return False

                for img in self.required_images:
                    if img not in self.img_features[tag]:
                        print(f"[CACHE MISS][IMG][{tag}] {img}")
                        return False

                    if not self._cache_entry_is_valid(self.img_features[tag][img]):
                        print(f"[CACHE FORMAT INVALID][{tag}] {img}")
                        return False

        else:
            if not isinstance(self.img_features, dict):
                return False

            for img in self.required_images:
                if img not in self.img_features:
                    print(f"[CACHE MISS][IMG] {img}")
                    return False

                if not self._cache_entry_is_valid(self.img_features[img]):
                    print(f"[CACHE FORMAT INVALID] {img}")
                    return False

        return True

    def _build_and_save_cache(self, features_filename):
        device = get_device("auto")

        local_ckpt = self.config["data"].get("clip_ckpt", None)
        if local_ckpt is not None and os.path.exists(local_ckpt):
            pretrained_arg = local_ckpt
        else:
            pretrained_arg = "openai"

        self.vlmodel, _, _ = open_clip.create_model_and_transforms(
            self.model_type,
            device=f"cuda:{device}",
            pretrained=pretrained_arg
        )

        for p in self.vlmodel.parameters():
            p.requires_grad = False
        self.vlmodel.eval()

        if self.uncertainty_aware:
            self.img_features = {}

            for tag in ["low", "medium", "high"]:
                self.img_features[tag] = self.ImageEncoder(
                    self.required_images,
                    self.blur_transform[tag]
                )

            self.img_features["avg"] = {}
            for k in self.img_features["medium"]:
                entries = [
                    self.img_features[tag][k]
                    for tag in ["low", "medium", "high"]
                ]
                self.img_features["avg"][k] = _average_img_feature_entries(entries)

        else:
            self.img_features = self.ImageEncoder(
                self.required_images,
                self.blur_transform
            )

        self.text_features = self.TextEncoder(self.required_texts)

        tmp_features_filename = features_filename + ".tmp"

        torch.save(
            {
                "img_features": self.img_features,
                "text_features": self.text_features,
            },
            tmp_features_filename
        )

        os.replace(tmp_features_filename, features_filename)

        print(f"[INFO] Saved feature cache: {features_filename}")

        del self.vlmodel
        torch.cuda.empty_cache()
        gc.collect()

    def load_data(self, data_path):
        d = torch.load(data_path, weights_only=False)

        eeg = d["eeg"]
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg)
        elif not isinstance(eeg, torch.Tensor):
            eeg = torch.tensor(eeg)
        eeg = eeg.float()

        if eeg.dim() != 4:
            raise RuntimeError(f"{data_path} eeg must be 4D [N,R,C,T], got {eeg.shape}")

        label_arr = d["label"]
        if isinstance(label_arr, torch.Tensor):
            label_arr = label_arr.cpu().numpy()
        else:
            label_arr = np.asarray(label_arr)

        img_arr = np.asarray(d["img"], dtype=object)
        text_arr = np.asarray(d["text"], dtype=object)

        session_arr = d["session"]
        if isinstance(session_arr, torch.Tensor):
            session_arr = session_arr.cpu().numpy()
        else:
            session_arr = np.asarray(session_arr)

        times_arr = d["times"]
        if isinstance(times_arr, torch.Tensor):
            times_arr = times_arr.cpu().numpy()
        else:
            times_arr = np.asarray(times_arr)

        # 统一解包成标准字符串
        if self.avg:
            avg_data = {}
            avg_data["eeg"] = eeg.mean(dim=1)   # [N,C,T]
            avg_data["label"] = label_arr[:, 0].astype(np.int64)

            avg_data["img"] = np.array(
                [str(self._unwrap_singleton(v)).replace("\\", "/") for v in img_arr[:, 0]],
                dtype=object
            )
            avg_data["text"] = np.array(
                [str(self._unwrap_singleton(v)) for v in text_arr[:, 0]],
                dtype=object
            )

            avg_data["session"] = session_arr.astype(np.int64)
            avg_data["times"] = times_arr.astype(np.int64)
            d = avg_data
        else:
            rep = eeg.shape[1]
            _data = {}
            _data["eeg"] = eeg.reshape(-1, *eeg.shape[2:])
            _data["eeg_avg"] = eeg.mean(dim=1)

            _data["label"] = label_arr.reshape(-1).astype(np.int64)
            _data["img"] = np.array(
                [str(self._unwrap_singleton(v)).replace("\\", "/") for v in img_arr.reshape(-1)],
                dtype=object
            )
            _data["text"] = np.array(
                [str(self._unwrap_singleton(v)) for v in text_arr.reshape(-1)],
                dtype=object
            )

            _data["session"] = np.repeat(session_arr.astype(np.int64), rep)
            _data["times"] = times_arr.astype(np.int64)
            d = _data

        return d

    @torch.no_grad()
    def ImageEncoder(self, images, blur_transform):
        
        set_images = sorted(list(set(str(x).replace("\\", "/") for x in images)))

        self.vlmodel.eval()
        device = next(self.vlmodel.parameters()).device

        return_original = bool(
            self.config["data"].get("return_original_feature_for_calibration", False)
        )
        return_stats = bool(
            self.config["data"].get("return_awmb_stats_for_calibration", False)
        )
        return_context = bool(
            self.config["data"].get("return_awmb_context_for_teacher", False)
        )

        if return_context and not hasattr(blur_transform, "get_fused_stats_context_images"):
            raise RuntimeError(
                "return_awmb_context_for_teacher=True requires blur_transform "
                "to implement get_fused_stats_context_images(img). "
                "Please use base.inpating_data.AttentionWeightedMultiBlurWithStats."
            )

        image_batch_size = int(
            self.config["data"].get("feature_extract_batch_size", 4)
        )
        encode_batch_size = int(
            self.config["data"].get("feature_encode_batch_size", 8)
        )

        image_features_dict = {}

        for i in tqdm(range(0, len(set_images), image_batch_size), desc=f"ImageEncoder-{self.mode}"):
            batch_images = set_images[i:i + image_batch_size]

            all_inputs = []
            image_counts = []
            main_counts = []
            context_counts = []
            stats_list = []

            for img_rel in batch_images:
                img_path = os.path.join(
                    self.data_dir,
                    "..",
                    "Image_set_Resize",
                    img_rel
                )

                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")

                pil_img = Image.open(img_path).convert("RGB")

                if return_context:
                    fused_img, stats, context_pils = blur_transform.get_fused_stats_context_images(pil_img)
                elif return_stats and hasattr(blur_transform, "get_fused_and_stats"):
                    fused_img, stats = blur_transform.get_fused_and_stats(pil_img)
                    context_pils = []
                else:
                    fused_img = blur_transform(pil_img)
                    stats = np.zeros(6, dtype=np.float32)
                    context_pils = []

                pil_list = [fused_img]

                if return_original:
                    pil_list.append(pil_img)

                main_count = len(pil_list)

                if return_context:
                    pil_list.extend(context_pils)

                context_count = len(context_pils) if return_context else 0

                image_counts.append(len(pil_list))
                main_counts.append(main_count)
                context_counts.append(context_count)
                stats_list.append(torch.tensor(stats, dtype=torch.float32))

                for p in pil_list:
                    all_inputs.append(self.process_transform(p))

            feature_chunks = []

            for j in range(0, len(all_inputs), encode_batch_size):
                sub_inputs = torch.stack(
                    all_inputs[j:j + encode_batch_size]
                ).to(device, non_blocking=True)

                sub_features = self.vlmodel.encode_image(sub_inputs)
                sub_features = sub_features / sub_features.norm(dim=-1, keepdim=True)
                sub_features = sub_features.float().cpu()

                feature_chunks.append(sub_features)

                del sub_inputs
                del sub_features
                torch.cuda.empty_cache()

            batch_features = torch.cat(feature_chunks, dim=0)

            offset = 0
            for img_rel, cnt, main_cnt, ctx_cnt, stat in zip(
                    batch_images,
                    image_counts,
                    main_counts,
                    context_counts,
                    stats_list
            ):
                feat_all = batch_features[offset:offset + cnt]
                offset += cnt

                main_feat = feat_all[:main_cnt]

                if main_cnt == 1:
                    main_feat_out = main_feat[0]
                else:
                    main_feat_out = main_feat

                if ctx_cnt > 0:
                    ctx_feat = feat_all[main_cnt:main_cnt + ctx_cnt]
                else:
                    d = main_feat.shape[-1]
                    ctx_feat = torch.zeros(3, d, dtype=torch.float32)

                if return_original or return_stats or return_context:
                    image_features_dict[img_rel] = {
                        "feat": main_feat_out,
                        "stat": stat,
                        "ctx_feat": ctx_feat,
                    }
                else:
                    image_features_dict[img_rel] = main_feat_out

            del all_inputs
            del feature_chunks
            del batch_features
            torch.cuda.empty_cache()
            gc.collect()

        return image_features_dict

    @torch.no_grad()
    def TextEncoder(self, texts):
        set_text = sorted(list(set(str(t) for t in texts)))
        text_inputs = torch.cat([open_clip.tokenize(t) for t in set_text])

        device = next(self.vlmodel.parameters()).device
        text_inputs = text_inputs.to(device)
        text_features = self.vlmodel.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return {set_text[i]: text_features[i].float().cpu() for i in range(len(set_text))}

    def _locate_subject_and_trial(self, index):
        subject_idx = bisect.bisect_right(self.subject_offsets, index) - 1
        trial_index = index - self.subject_offsets[subject_idx]
        return subject_idx, trial_index

    def __getitem__(self, index):
        subject, trial_index = self._locate_subject_and_trial(index)

        eeg = self.loaded_data[subject]["eeg"][trial_index].float()

        if "eeg_avg" in self.loaded_data[subject]:
            eeg_mean = self.loaded_data[subject]["eeg_avg"][trial_index].float()
        else:
            eeg_mean = eeg

        label = int(self.loaded_data[subject]["label"][trial_index])
        img_path = str(self.loaded_data[subject]["img"][trial_index]).replace("\\", "/")
        text = str(self.loaded_data[subject]["text"][trial_index])
        session = int(self.loaded_data[subject]["session"][trial_index])

        match_label = int(self.match_label[index])

        if self.uncertainty_aware:
            if self.mode == "train":
                if match_label == 0:
                    tag = "low"
                elif match_label == 2:
                    tag = "high"
                else:
                    tag = "medium"
            else:
                tag = "medium"

            if img_path not in self.img_features[tag]:
                raise KeyError(
                    f"Missing img feature key: {img_path}\n"
                    f"mode={self.mode}, tag={tag}, subject={subject}, trial_index={trial_index}\n"
                    f"Hint: delete data/eegcvpr40/Image_feature and rebuild."
                )

            img_entry = self.img_features[tag][img_path]

        else:
            if img_path not in self.img_features:
                raise KeyError(
                    f"Missing img feature key: {img_path}\n"
                    f"mode={self.mode}, subject={subject}, trial_index={trial_index}\n"
                    f"Hint: delete data/eegcvpr40/Image_feature and rebuild."
                )

            img_entry = self.img_features[img_path]

      
        if isinstance(img_entry, dict):
            img_features = img_entry["feat"]
            img_stats = img_entry.get("stat", torch.zeros(6, dtype=torch.float32))
            ctx_features = img_entry.get("ctx_feat", None)
        else:
            img_features = img_entry
            img_stats = torch.zeros(6, dtype=torch.float32)
            ctx_features = None

        if not isinstance(img_stats, torch.Tensor):
            img_stats = torch.tensor(img_stats, dtype=torch.float32)
        else:
            img_stats = img_stats.float()

        if ctx_features is None:
            if isinstance(img_features, torch.Tensor):
                d = img_features.shape[-1]
            else:
                d = torch.tensor(img_features).shape[-1]
            ctx_features = torch.zeros(3, d, dtype=torch.float32)
        else:
            if not isinstance(ctx_features, torch.Tensor):
                ctx_features = torch.tensor(ctx_features, dtype=torch.float32)
            else:
                ctx_features = ctx_features.float()

        sample = {
            "idx": index,
            "eeg": eeg[:, self.timesteps[0]:self.timesteps[1]],
            "label": label,
            "img_path": img_path,
            "img": "None",
            "img_features": img_features,
            "img_stats": img_stats,
            "ctx_features": ctx_features,
            "text": text,
            "text_features": self.text_features[text],
            "session": session,
            "subject": subject,
            "eeg_mean": eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
        }

        return sample

    def __len__(self):
        return self.trial_all_subjects