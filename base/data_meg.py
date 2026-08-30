import torch,os
import copy
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import logging
import open_clip
import gc
from tqdm import tqdm
import itertools

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from base.utils import instantiate_from_config, get_device 


def load_meg_data(config):
    exp_setting = config.get('exp_setting', 'intra-subject')
    
    if exp_setting == 'intra-subject':
        test_dataset = MEGDataset(config,mode='test')
        print('init test_dataset success')
        train_dataset = MEGDataset(config,mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False,num_workers=25, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=32, pin_memory=True)
        return train_loader, test_loader,test_loader
    
    elif exp_setting == 'inter-subject':
        subjects = config['data']['subjects']
        test_dataset = MEGDataset(config,mode='test')
        print('init test_dataset success')
        
        all_subjects = [f'sub-{i:02}' for i in range(1, 5)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))
        leave_one_subjects_config = config
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects
        val_dataset = MEGDataset(leave_one_subjects_config,mode='test')
        print('init val_dataset success')
        train_dataset = MEGDataset(leave_one_subjects_config,mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False,num_workers=25)#, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['val_batch_size'], shuffle=False, drop_last=False,num_workers=32)#, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=32)#, pin_memory=True)
        return train_loader, val_loader, test_loader

def _to_float_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


def _average_img_feature_entries(entries):
    """
    兼容:
    1. tensor[D]
    2. tensor[2,D]
    3. {"feat": tensor[2,D], "stat": tensor[6], "ctx_feat": tensor[3,D]}
    """
    first = entries[0]

    if isinstance(first, dict):
        feats = []
        stats = []
        ctx_feats = []

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
    
class MEGDataset(Dataset):
    def __init__(self, config, mode):
        self.config= config
        self.data_dir = config['data']['data_dir']
        # self.img_directory = os.path.join(self.data_dir,'../','Image_set_Resize',f'{mode}_images')
        # self.all_class_names = [d.split('_',1)[-1] for d in os.listdir(self.img_directory) if os.path.isdir(os.path.join(self.img_directory, d))]
        # self.all_class_names.sort()
        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = None
        if self.selected_ch == "None":
            self.selected_ch = self.channels
    
        self.avg = config['data'][f"{mode}_avg"]

        self.blur_type = config['data']['blur_type']

        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode=='train' else 200
        self.per_trials = 1 if self.mode=='train' else 12

        self.data_paths = [os.path.join(self.data_dir,subject,f'{mode}.pt') for subject in self.subjects]
        self.loaded_data= [self.load_data(data_path) for data_path in self.data_paths]
        
        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject*len(self.subjects)

        data_dir = os.path.join(self.data_dir,'../Image_feature',f"{config['data']['blur_type']['target'].rsplit('.',1)[-1]}")
        os.makedirs(data_dir,exist_ok=True)
        features_filename = os.path.join(data_dir,f"{self.name}_{mode}.pt")

        pretrain_map= {
                'RN50':{'pretrained':'openai','resize':(224,224)}, #1024 
                'RN101':{'pretrained':'openai','resize':(224,224)}, #512
                'ViT-B-16':{'pretrained':'laion2b_s34b_b88k','resize':(224,224)}, #512
                'ViT-B-32':{'pretrained':'laion2b_s34b_b79k','resize':(224,224)}, #512
                'ViT-L-14':{'pretrained':'laion2b_s32b_b82k','resize':(224,224)}, #768
                'ViT-H-14':{'pretrained':'laion2b_s32b_b79k','resize':(224,224)}, #1024
                'ViT-g-14':{'pretrained':'laion2b_s34b_b88k','resize':(224,224)}, #1024
                'ViT-bigG-14':{'pretrained':'laion2b_s39b_b160k','resize':(224,224)}, #1280
            }
        self.c = config['c']

       
        if self.config['data']['uncertainty_aware']:
            self.blur_transform = {}

            base_blur_param = copy.deepcopy(config['data']['blur_type'])
            base_kernel_size = int(base_blur_param['params']['blur_kernel_size'])

            for shift, tag in zip([-self.c, 0, self.c], ['low', 'medium', 'high']):
                blur_param = copy.deepcopy(config['data']['blur_type'])
                blur_param['params']['blur_kernel_size'] = base_kernel_size + shift
                self.blur_transform[tag] = instantiate_from_config(blur_param)
        else:
            self.blur_transform = instantiate_from_config(config['data']['blur_type'])

        process_term = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ]

        self.process_transform = transforms.Compose(process_term)
        self.match_label = np.ones(self.trial_all_subjects, dtype=int)

       
        cache_loaded = False

        if os.path.exists(features_filename):
            try:
                saved_features = torch.load(features_filename, weights_only=False)
                self.img_features = saved_features['img_features']
                cache_loaded = True
                print(f"[INFO] Loaded feature cache: {features_filename}")

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
            device = get_device('auto')

            
            local_ckpt = "/mnt/i/slvp/pretrained/open_clip_pytorch_model.bin"

            if os.path.exists(local_ckpt):
                pretrained_source = local_ckpt
            else:
                pretrained_source = pretrain_map[self.model_type]['pretrained']

            self.vlmodel, self.preprocess, _ = open_clip.create_model_and_transforms(
                self.model_type,
                device=f"cuda:{device}",
                pretrained=pretrained_source
            )

            for param in self.vlmodel.parameters():
                param.requires_grad = False

            self.vlmodel.eval()

            if self.config['data']['uncertainty_aware']:
                self.img_features = {}

                for tag in ['low', 'medium', 'high']:
                    self.img_features[tag] = self.ImageEncoder(
                        self.loaded_data[0]['img'],
                        self.blur_transform[tag]
                    )

                self.img_features['avg'] = {}

                for k in self.img_features['medium']:
                    entries = [
                        self.img_features[tag][k]
                        for tag in ['low', 'medium', 'high']
                    ]
                    self.img_features['avg'][k] = _average_img_feature_entries(entries)

            else:
                self.img_features = self.ImageEncoder(self.loaded_data[0]['img'])

            tmp_features_filename = features_filename + ".tmp"

            torch.save(
                {
                    'img_features': self.img_features,
                },
                tmp_features_filename
            )

            os.replace(tmp_features_filename, features_filename)

            print(f"[INFO] Saved feature cache: {features_filename}")

            del self.vlmodel
            torch.cuda.empty_cache()
            gc.collect()

    def load_data(self,data_path):
        logging.info(f"----load {data_path.rsplit('1000HZ',1)[-1]}----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg']=torch.from_numpy(loaded_data['eeg'])
        
        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:,:,selected_idx]
        if self.avg:
            avg_data={}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            #avg_data['label'] = loaded_data['label'][:,0]
            avg_data['img'] = np.array(loaded_data['img'])#[:,0]
            #avg_data['text'] = loaded_data['text'][:,0]
                
            #avg_data['session'] = loaded_data['session']
            #avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1,*loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            #_data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            # _data['text'] = loaded_data['text'].reshape(-1)
            # _data['session'] = loaded_data['session'].reshape(-1)
            # _data['times'] = loaded_data['times']
            loaded_data = _data
        
        
        for k,v in loaded_data.items():
            if k in ['eeg','label','img','text','session']:
                logging.info(f"{k}: {v.shape}")
        return loaded_data

    @torch.no_grad()
    def ImageEncoder(self, images, blur_transform=None):
        """
        最终版 contextual adapter 需要缓存:
            {
                "feat": tensor [2,D],      # [AWMB fused, original]
                "stat": tensor [6],       # AWMB statistics
                "ctx_feat": tensor [3,D], # [low, mid, high]
            }
        """
        if blur_transform is None:
            blur_transform = self.blur_transform

        self.vlmodel.eval()

        return_original = bool(
            self.config['data'].get('return_original_feature_for_calibration', False)
        )
        return_stats = bool(
            self.config['data'].get('return_awmb_stats_for_calibration', False)
        )
        return_context = bool(
            self.config['data'].get('return_awmb_context_for_teacher', False)
        )

        if return_context and not hasattr(blur_transform, "get_fused_stats_context_images"):
            raise RuntimeError(
                "return_awmb_context_for_teacher=True requires blur_transform "
                "to implement get_fused_stats_context_images(img). "
                "Please use base.inpating_data.AttentionWeightedMultiBlurWithStats."
            )

        set_images = list(set(images))
        set_images.sort()

        image_batch_size = int(self.config['data'].get('feature_extract_batch_size', 4))
        encode_batch_size = int(self.config['data'].get('feature_encode_batch_size', 8))

        image_features_dict = {}

        for i in tqdm(range(0, len(set_images), image_batch_size)):
            batch_images = set_images[i:i + image_batch_size]
            device = next(self.vlmodel.parameters()).device

            all_inputs = []
            image_counts = []
            main_counts = []
            context_counts = []
            stats_list = []

            for img_path in batch_images:
                pil_img = Image.open(
                    os.path.join(self.data_dir, '../Image_set_Resize', img_path)
                ).convert("RGB")

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
                sub_inputs = torch.stack(all_inputs[j:j + encode_batch_size]).to(
                    device,
                    non_blocking=True
                )

                sub_features = self.vlmodel.encode_image(sub_inputs)
                sub_features = sub_features / sub_features.norm(dim=-1, keepdim=True)
                sub_features = sub_features.float().cpu()

                feature_chunks.append(sub_features)

                del sub_inputs
                del sub_features
                torch.cuda.empty_cache()

            batch_image_features = torch.cat(feature_chunks, dim=0)

            offset = 0

            for img_path, cnt, main_cnt, ctx_cnt, stat in zip(
                    batch_images,
                    image_counts,
                    main_counts,
                    context_counts,
                    stats_list
            ):
                feat_all = batch_image_features[offset:offset + cnt]
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

                if return_stats or return_original or return_context:
                    image_features_dict[img_path] = {
                        "feat": main_feat_out,
                        "stat": stat,
                        "ctx_feat": ctx_feat,
                    }
                else:
                    image_features_dict[img_path] = main_feat_out

            del all_inputs
            del feature_chunks
            del batch_image_features
            torch.cuda.empty_cache()
            gc.collect()

        return image_features_dict
    
    @torch.no_grad()
    def Textencoder(self, text):   
        set_text = list(set(text))
        text_inputs = torch.cat([open_clip.tokenize(f"This is a {t}.") for t in set_text])
        device = next(self.vlmodel.parameters()).device
        text_inputs =  text_inputs.to(device)
        text_features = self.vlmodel.encode_text(text_inputs)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        text_features_dict = {set_text[i]:text_features[i].float().cpu() for i in range(len(set_text))}
        return text_features_dict

    def __getitem__(self, index):

        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()

        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][
                trial_index // self.per_trials
                ].float()

        img_path = self.loaded_data[subject]['img'][trial_index]

        img = 'None'

        match_label = self.match_label[index]

        if self.config['data']['uncertainty_aware']:
            if self.mode == 'train':
                if match_label == 0:
                    tag = 'low'
                elif match_label == 2:
                    tag = 'high'
                else:
                    tag = 'medium'
            else:
                tag = 'medium'

            img_entry = self.img_features[tag][img_path]
        else:
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
            'idx': index,

            # MEG 这里仍然叫 eeg，方便 main.py 复用
            'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],

            'img_path': img_path,
            'img': img,
            'img_features': img_features,
            'img_stats': img_stats,
            'ctx_features': ctx_features,
            'subject': subject,
            'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
        }

        return sample
    
    def __len__(self):
        return self.trial_all_subjects
    