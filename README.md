# TimePoint
TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning. ICML 2025.

[Ron Shapira Weber](https://ronshapiraweber.github.io/), Shahar Ben Ishay, Andrey Lavrinenko, [Shahaf E. Finder](https://shahaffind.github.io/), and [Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/)

[![arXiv](https://img.shields.io/badge/arXiv-2505.23475-b31b1b.svg?style=flat)](https://arxiv.org/abs/2505.23475)


## How to use
### TimePoint

```python
from TimePoint.models.timepoint import TimePoint

# Pre-trained model paths
# Trained on SynthAlign 
model_path = "../TimePoint/models/pretrained_weights/synth_only.pth"
# Trained on SynthAlign + Fine-tuned on UCR
model_path = "../TimePoint/models/pretrained_weights/synth_and_ucr.pth"

# init params
encoder_dims = [128, 128, 256, 256]
encoder_type = "wtconv" # dense, wtconv

# init model and load weights
descriptor_dim = 256
device = "cuda"
timepoint = TimePoint(input_channels=1,
                      encoder_dims=encoder_dims,
                      descriptor_dim=descriptor_dim,
                      encoder_type=encoder_type
                      )

timepoint.load_state_dict(torch.load(model_path))

# Dummy input (N, C, L)
X_batch = torch.rand(10, 1, 512)
# (N, L), (N, 256, L)
detection_proba, descriptors = timepoint(X_batch)
```
Can also get topK keypoints
```python
kp_percentage = 0.1 # 10% of sequence length
# Non-Maximum Suppression (NMS)
nms_window = 5
# (1, num_kp), (1, 256, num_kp), (1, 256, L)
sorted_topk_indices, detection_proba, descriptors = timepoint.get_topk_points(X_batch, kp_percentage, nms_window)
```

### SynthAlign
```python
# Patterns: sinewave composition, block, sawtooth, RBF
probas = [0.6, 0.15, 0.05, 0.2]
data_length = 512
# dataset "size"
N_data = 100
transform = None
# Init
synth_dataset = SynthAlign(
        data_types=[SynthAlign.sine_wave_composition, SynthAlign.block_wave,
                    SynthAlign.sawtooth_wave, SynthAlign.radial_basis_function],
        probs=probas,
        data_length=data_length,
        total_samples=N_data,
        cache_size=0,
        transform=transform
        )
# Get data
data_dict = synth_dataset[i]
# (1, L), (L,)
X_i, keypoints_i = data_dict["signals"], data_dict["keypoints"]

```

# TODO List
- [x] Pretrained models
- [x] SynthAlign dataset
- [x] Example Usage notebook
- [ ] Requirements file
- [ ] Training code
- [ ] Hugging Face support

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{weber:2025:timepoint,
  title     = {TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning},
  author    = {Shapira Weber, Ron and Ben Ishay, Shahar and Lavrinenko, Andrey and Finder, Shahaf E and Freifeld, Oren},
  booktitle = {International Conference on Machine Learning},
  year      = {2025},
}
```
