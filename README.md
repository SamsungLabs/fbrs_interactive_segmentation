## Interactive Segmentation using f-BRS

<img src="https://media2.giphy.com/media/ZfS8bjls6D7BI4qlg5/giphy.gif?cid=790b7611fb7f28a26c645d7c6d61f05f1a3784b41640215e&rid=giphy.gif&ct=g" width="300" height="300"/>

This fork makes the f-BRS library easier to work with for your own application. Here are the steps to use it:

Clone this repository to your workspace:

```bash
git clone https://github.com/cviss-lab/fbrs_interactive_segmentation.git
```

Then set up the environment. This framework is built using Python 3.6 and relies on the PyTorch 1.4.0+. CUDA is optional but highly recommended. The following command installs all necessary packages:

```.bash
pip3 install -r requirements.txt
```

In your own project, insert the following code to interactively segment an image. You must specify the image coordinates and type of seed points (positive or negative).

```python
import sys
sys.path.append('path-to-fbrs-repo') # append system path to library
from fbrs_interactive_segmentation import fbrs_predict # import library to your project

checkpoint = 'resnet34_dh128_sbd' # Download a pretrained model from below, 
                                  # and place it in the weights folder in this library
engine = fbrs_predict.fbrs_engine(checkpoint) # Initialize model
image = cv2.imread('path-to-image') # Load your image 
                                    # make sure to downscale image first to remove high frequency components
x_coord = [] # x image coordinates seed pts
y_coord = [] # y image coordinates seed pts
is_pos = []  # Either 1 or 0 for + or - seed pts
mask_pred = engine.predict(x_coord, y_coord, is_pos, image) # Get segmentation mask
```

The file demo.py shows an example of how to use this library. 

If the pretrained models are not sufficient, consider using the tools in the original fork to train a .pth model using your own data/parameters.



## f-BRS: Rethinking Backpropagating Refinement for Interactive Segmentation [[Paper]](https://arxiv.org/abs/2001.10331) [[PyTorch]](https://github.com/saic-vul/fbrs_interactive_segmentation/tree/master) [[MXNet]](https://github.com/saic-vul/fbrs_interactive_segmentation/tree/mxnet) [[Video]](https://youtu.be/ArcZ5xtyMCk)

This repository provides code for training and testing state-of-the-art models for interactive segmentation with the official PyTorch implementation of the following paper:

> **f-BRS: Rethinking Backpropagating Refinement for Interactive Segmentation**<br>
> [Konstantin Sofiiuk](https://github.com/ksofiyuk), [Ilia Petrov](https://github.com/ptrvilya), [Olga Barinova](https://github.com/OlgaBarinova), [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)<br>
> Samsung AI Center Moscow <br>
> https://arxiv.org/abs/2001.10331

Please see [the video](https://youtu.be/ArcZ5xtyMCk) below explaining how our algorithm works:
<p align="center">
  <a href="https://youtu.be/ArcZ5xtyMCk">
  <img src="./images/fbrs_video_preview.gif" alt="drawing" width="70%"/>
  </a>
</p>

We also have full MXNet implementation of our algorithm, you can check [mxnet branch](https://github.com/saic-vul/fbrs_interactive_segmentation/tree/mxnet).

## News
* [2021-02-16] We have presented a new paper (+code) on interactive segmentation: [Reviving Iterative Training with Mask Guidance for Interactive Segmentation](https://github.com/saic-vul/ritm_interactive_segmentation). A simpler approach with new SoTA results and without any test-time optimization techniques.


## Datasets

We train all our models on SBD dataset and evaluate them on GrabCut, Berkeley, DAVIS, SBD and COCO_MVal datasets. We additionally provide the results of models that trained on combination of [COCO](http://cocodataset.org) and [LVIS](https://www.lvisdataset.org) datasets.

Berkeley dataset consists of 100 instances (96 unique images) provided by [[K. McGuinness, 2010]][McGuinness2010].
We use the same 345 images from DAVIS dataset for testing as [[WD Jang, 2019]][BRS], ground-truth mask for each image is a union of all objects' masks.
For testing on SBD dataset we evaluate our algorithm on every instance in the test set separately following the protocol of [[WD Jang, 2019]][BRS].

To construct COCO_MVal dataset we sample 800 object instances from the validation set of [COCO 2017](http://cocodataset.org). Specifically, we sample 10 unique instances from each of the 80 categories. The only exception is the toaster object class, which has only 9 unique instances in instances_val2017 annotation. So to get 800 masks one of the classes contains 11 objects. We provide this dataset for downloading so that everyone can reproduce our results.

[BRS]: http://openaccess.thecvf.com/content_CVPR_2019/papers/Jang_Interactive_Image_Segmentation_via_Backpropagating_Refinement_Scheme_CVPR_2019_paper.pdf
[McGuinness2010]: https://www.sciencedirect.com/science/article/abs/pii/S0031320309000818

| Dataset | Description |      Download Link        |
|---------|-------------|:-------------------------:|
|SBD      |  8498 images with 20172 instances for training and <br> 2857 images with 6671 instances for testing |  [official site][SBD]     |
|Grab Cut |  50 images with one object each   |  [GrabCut.zip (11 MB)][GrabCut]   |
|Berkeley  |  96 images with 100 instances     |  [Berkeley.zip (7 MB)][Berkeley] |
|DAVIS    |  345 images with one object each  |  [DAVIS.zip (43 MB)][DAVIS]       |
|COCO_MVal | 800 images with 800 instances | [COCO_MVal.zip (127 MB)][COCO_MVal] |

[GrabCut]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip
[Berkeley]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip
[DAVIS]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/DAVIS.zip
[SBD]: http://home.bharathh.info/pubs/codes/SBD/download.html
[COCO_MVal]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/COCO_MVal.zip

Don't forget to change the paths to the datasets in [config.yml](config.yml) after downloading and unpacking.

## Weights

### Pretrained models
We provide pretrained models with different backbones for interactive segmentation. The evaluation results are different from the ones presented in our paper, because we have retrained all models on the new codebase presented in this repository. We greatly accelerated the inference of the RGB-BRS algorithm - now it works from 2.5 to 4 times faster on SBD dataset compared to the timings given in the paper. Nevertheless, the new results sometimes are even better.

Note that all ResNet models were trained using [MXNet branch](https://github.com/saic-vul/fbrs_interactive_segmentation/tree/mxnet) and then converted to PyTorch (they have equivalent results). We provide the [script](./scripts/convert_weights_mx2pt.py) that was used to convert the models. HRNet models were trained using PyTorch.

You can find model weights and test results in the tables below:

<table>
  <tr>
    <th>Backbone</th>
    <th>Train Dataset</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>ResNet-34</td>
    <td>SBD</td>
    <td><a href="https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/resnet34_dh128_sbd.pth">resnet34_dh128_sbd.pth (GitHub, 89 MB)</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>SBD</td>
    <td><a href="https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/resnet50_dh128_sbd.pth">resnet50_dh128_sbd.pth (GitHub, 120 MB)</a></td>
  </tr>
  <tr>
    <td>ResNet-101</td>
    <td>SBD</td>
    <td><a href="https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/resnet101_dh256_sbd.pth">resnet101_dh256_sbd.pth (GitHub, 223 MB)</a></td>
  </tr>
  <tr>
    <td>HRNetV2-W18+OCR</td>
    <td>SBD</td>
    <td><a href="https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/hrnet18_ocr64_sbd.pth">hrnet18_ocr64_sbd.pth (GitHub, 39 MB)</a></td>
  </tr>
  <tr>
    <td>HRNetV2-W32+OCR</td>
    <td>SBD</td>
    <td><a href="https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/hrnet32_ocr128_sbd.pth">hrnet32_ocr128_sbd.pth (GitHub, 119 MB)</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>COCO+LVIS</td>
    <td><a href="https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/resnet50_dh128_lvis.pth">resnet50_dh128_lvis.pth (GitHub, 120 MB)</a></td>
  </tr>
  <tr>
    <td>HRNetV2-W32+OCR</td>
    <td>COCO+LVIS</td>
    <td><a href="https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/hrnet32_ocr128_lvis.pth">hrnet32_ocr128_lvis.pth (GitHub, 119 MB)</a></td>
  </tr>
</table>

<table align="center">
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2"><span style="font-weight:bold">BRS</span><br><span style="font-weight:bold">Type</span></th>
    <th colspan="2">GrabCut</th>
    <th colspan="2">Berkeley</th>
    <th colspan="2">DAVIS</th>
    <th colspan="2">SBD</th>
    <th colspan="2">COCO_MVal</th>
  </tr>
  <tr>
    <td>NoC<br>85%</td>
    <td>NoC<br>90%</td>
    <td>NoC<br>85%</td>
    <td>NoC<br>90%</td>
    <td>NoC<br>85%</td>
    <td>NoC<br>90%</td>
    <td>NoC<br>85%</td>
    <td>NoC<br>90%</td>
    <td>NoC<br>85%</td>
    <td>NoC<br>90%</td>
  </tr>
  <tr>
    <td rowspan="2">ResNet-34<br>(SBD)</td>
    <td>RGB-BRS</td>
    <td>2.04</td>
    <td>2.50</td>
    <td>2.22</td>
    <td>4.49</td>
    <td>5.34</td>
    <td>7.91</td>
    <td>4.19</td>
    <td>6.83</td>
    <td>4.16</td>
    <td>5.52</td>
  </tr>

  <tr>
    <td>f-BRS-B</td>
    <td>2.06</td>
    <td>2.48</td>
    <td>2.40</td>
    <td>4.17</td>
    <td>5.34</td>
    <td>7.73</td>
    <td>4.47</td>
    <td>7.28</td>
    <td>4.31</td>
    <td>5.79</td>
  </tr>

  <tr>
    <td rowspan="2">ResNet-50<br>(SBD)</td>
    <td>RGB-BRS</td>
    <td>2.16</td>
    <td>2.56</td>
    <td>2.17</td>
    <td>4.27</td>
    <td>5.27</td>
    <td>7.51</td>
    <td>4.00</td>
    <td>6.59</td>
    <td>4.12</td>
    <td>5.61</td>
  </tr>

  <tr>
    <td>f-BRS-B</td>
    <td>2.20</td>
    <td>2.64</td>
    <td>2.17</td>
    <td>4.22</td>
    <td>5.44</td>
    <td>7.81</td>
    <td>4.55</td>
    <td>7.45</td>
    <td>4.31</td>
    <td>6.26</td>
  </tr>

  <tr>
    <td rowspan="2">ResNet-101<br>(SBD)</td>
    <td>RGB-BRS</td>
    <td>2.10</td>
    <td>2.46</td>
    <td>2.34</td>
    <td>3.91</td>
    <td>5.19</td>
    <td><b>7.23</b></td>
    <td>3.78</td>
    <td><b>6.28</b></td>
    <td>3.98</td>
    <td>5.45</td>
  </tr>

  <tr>
    <td>f-BRS-B</td>
    <td>2.30</td>
    <td>2.68</td>
    <td>2.61</td>
    <td>4.22</td>
    <td>5.32</td>
    <td><b>7.35</b></td>
    <td>4.20</td>
    <td>7.10</td>
    <td>4.11</td>
    <td>5.91</td>
  </tr>

  <tr>
    <td rowspan="2">HRNet-W18+OCR<br>(SBD)</td>
    <td>RGB-BRS</td>
    <td>1.68</td>
    <td><b>1.94</b></td>
    <td>1.99</td>
    <td>3.81</td>
    <td>5.49</td>
    <td>7.98</td>
    <td>4.19</td>
    <td>6.84</td>
    <td>3.62</td>
    <td><b>5.04</b></td>
  </tr>

  <tr>
    <td>f-BRS-B</td>
    <td>1.86</td>
    <td>2.18</td>
    <td>2.07</td>
    <td>3.96</td>
    <td>5.62</td>
    <td>8.08</td>
    <td>4.70</td>
    <td>7.65</td>
    <td>3.87</td>
    <td>5.57</td>
  </tr>

  <tr>
    <td rowspan="2">HRNet-W32+OCR<br>(SBD)</td>
    <td>RGB-BRS</td>
    <td>1.80</td>
    <td>2.16</td>
    <td>2.00</td>
    <td><b>3.58</b></td>
    <td>5.40</td>
    <td>7.59</td>
    <td>3.87</td>
    <td><b>6.33</b></td>
    <td>3.61</td>
    <td><b>5.12</b></td>
  </tr>

  <tr>
    <td>f-BRS-B</td>
    <td>1.78</td>
    <td>2.16</td>
    <td>2.13</td>
    <td><b>3.69</b></td>
    <td>5.54</td>
    <td>7.62</td>
    <td>4.31</td>
    <td>7.08</td>
    <td>3.82</td>
    <td>5.44</td>
  </tr>

  <tr>
    <td class="divider" colspan="12"><hr /></td>
  </tr>

  <tr>
    <td rowspan="2">ResNet-50<br>(COCO+LVIS)</td>
    <td>RGB-BRS</td>
    <td>1.54</td>
    <td>1.76</td>
    <td>1.56</td>
    <td>2.70</td>
    <td>4.93</td>
    <td><b>6.22</b></td>
    <td>4.04</td>
    <td><b>6.85</b></td>
    <td>2.41</td>
    <td><b>3.47</b></td>
  </tr>

  <tr>
    <td>f-BRS-B</td>
    <td>1.52</td>
    <td>1.74</td>
    <td>1.56</td>
    <td>2.61</td>
    <td>4.94</td>
    <td><b>6.36</b></td>
    <td>4.29</td>
    <td><b>7.20</b></td>
    <td>2.34</td>
    <td><b>3.43</b></td>
  </tr>

  <tr>
    <td rowspan="2">HRNet-W32+OCR<br>(COCO+LVIS)</td>
    <td>RGB-BRS</td>
    <td>1.54</td>
    <td><b>1.60</b></td>
    <td>1.63</td>
    <td><b>2.59</b></td>
    <td>5.06</td>
    <td>6.34</td>
    <td>4.18</td>
    <td>6.96</td>
    <td>2.38</td>
    <td>3.55</td>
  </tr>

  <tr>
    <td>f-BRS-B</td>
    <td>1.54</td>
    <td><b>1.69</b></td>
    <td>1.64</td>
    <td><b>2.44</b></td>
    <td>5.17</td>
    <td>6.50</td>
    <td>4.37</td>
    <td>7.26</td>
    <td>2.35</td>
    <td>3.44</td>
  </tr>

</table>

## License

The code is released under the MPL 2.0 License. MPL is a copyleft license that is easy to comply with. You must make the source code for any of your changes available under MPL, but you can combine the MPL software with proprietary code, as long as you keep the MPL code in separate files.


## Citation

If you find this work is useful for your research, please cite our paper:
```
@article{fbrs2020,
  title={f-BRS: Rethinking Backpropagating Refinement for Interactive Segmentation},
  author={Konstantin Sofiiuk, Ilia Petrov, Olga Barinova, Anton Konushin},
  journal={arXiv preprint arXiv:2001.10331},
  year={2020}
}
```
