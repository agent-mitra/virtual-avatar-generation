# Virtual avatar generation models as world navigators
Code repository for the paper "Virtual avatar generation models as world navigators". \
[Sai Mandava](https://www.linkedin.com/in/sai-mandava-7237abb7/)
<table>
  <tr>
    <td>
      <a href="https://arxiv.org/abs/2112.04477">
        <img src="https://img.shields.io/badge/arXiv-2112.04477-00ff00.svg" alt="arXiv">
      </a>
    </td>
    <td>
      <a href="www.google.com">Project Page</a>
    </td>
  </tr>
</table>

This code repository provides code implementation for our paper above, with installation, demo code, dataset preparation, and training recipe.

[![](https://dcbadge.limes.pink/api/server/84EPT97nf4)](https://discord.gg/84EPT97nf4)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/FM.svg?style=social&label=@SABR_ai)](https://x.com/SABR_ai)

> 🔗 [Demo](https://www.youtube.com/watch?v=cnvNPWoYZz4) • [Installation](#Installation) • [Train](#Training) • [FAQ](#FAQ) • [SABR use cases interest form](https://docs.google.com/forms/d/1KzvpX2mffBTncmV3d9Z9JEj5_OnGkSMTLyQkPFfCBtw/edit) • [Data & Benchmark Request](#Training-Data-and-Benchmark)

## Installation
First, download the [weights_download.zip](https://drive.google.com/file/d/1RWyyXyUMf97JSvBMJRrv3dvBbTDUlPb-/view?usp=sharing) file and put it in your root directory. Then, you may install the dependencies as:
```
bash Mambaforge-Linux-x86_64.sh
chmod +x ./install.sh
source install.sh
mamba activate sabr
```

If you are using a cloud service (e.g AWS, Cloudflare) to store data, fill out the .env file with your relevant credentials.

## Data Collection
First, put all your videos in a folder in the root directory.
<details>
  <summary>Step-by-step instructions</summary>

```bash
cd scripts/collect_mocap_data
python collect_data.py --videoDir <your_video_folder>
```
</details>
<br>

The data collection pipeline will automatically collect mocap data, context data, and raw environment data from your videos.
Optionally, you can uncomment the relevant lines in collect_data.py to auto upload all of this data to a cloud service of your choice.

## Training
Fill out the file paths for your training data in model/architecture/dataset/video_dataset.py. Then, you can run the following:
```
cd model/architecture
torchrun --nproc_per_node=8 train.py
```

Depending on the number of GPUs you have, you can modify the command accordingly.

## Inference
```
python inference.py --ckpt <your_model_ckpt_path> --mode 0 --videoDir <folder_of_your_videos>
```

## FAQ

#### How much VRAM do I need?
* Because SABR-Climb is a full pipeline system, you will need an 80GB GPU to do inference over 1 video. The main bottleneck here is the context model we use to annotate the route. After training the model, you can skip the context model and just see what it outputs for raw inpainted frames. For doing this, you would need a 40GB GPU.

#### I keep having issues with OpenGL. What do I do?
* sudo add-apt-repository ppa:graphics-drivers/ppa
* sudo apt-get install nvidia-driver-525
* sudo apt-get install libnvidia-gl-535 libnvidia-compute-535 libnvidia-extra-535 nvidia-compute-utils-535 libnvidia-decode-535 libnvidia-encode-535 nvidia-utils-535
* sudo reboot

#### I still see memory taken up after training finishes. What do I do?
* run [killall python](https://discuss.pytorch.org/t/pytorch-doesnt-free-gpus-memory-of-it-gets-aborted-due-to-out-of-memory-error/13775/15)

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [DiT](https://github.com/facebookresearch/DiT)
- [deep sort](https://github.com/nwojke/deep_sort)
- [SMPL-X](https://github.com/vchoutas/smplx)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)

## Training Data and Benchmark

If you are a researcher or part of a research organization and would like access to the training data and benchmark or contribute in any way, please contact the author via email.

## © License

This repo is licensed under the MIT License. The SMPL model is under a different license so please follow that.

## 📱 Community

- [Twitter](https://x.com/SABR_ai)
- [Discord](https://discord.gg/84EPT97nf4)

## SABR Platforms (Early Access)

We’re planning on taking some of this foundational research and working with people to see what kinds of use cases might be actually useful to them - individual or enterprise.

If you are interested in seeing applications of this simulation technology, you can fill this form out [here](https://docs.google.com/forms/d/e/1FAIpQLSeyUqgZ2683CStJkeR6_gP6uaM16QC9YrJKw7wRA1QzcYTvnA/viewform?usp=sf_link).

## Citation
If you find this code useful for your research or the use data generated by our method, please consider citing the following paper:
```bibtex
@inproceedings{mandava2024virtualavatarnavigation,
  title={Virtual avatar generation models as world navigators},
  author={Mandava, Sai},
  booktitle={SABR},
  year={2024}
}
```