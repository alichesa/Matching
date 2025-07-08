# Matching
```
    parser.add_argument('--megadepth_root_path', type=str, default='/mnt/usb/zdzdata/MegaDepth_v1',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='/mnt/usb/zdzdata/coco_20k/coco_20k',
                        help='Path to the synthetic dataset root directory.')
```

```
            TRAIN_BASE_PATH = f"{megadepth_root_path}/index"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/train/phoenix/S6/zl548/MegaDepth_v1"
            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

```

![image](https://github.com/user-attachments/assets/75ec174f-9966-428d-aff1-a9c58c245a9a)
