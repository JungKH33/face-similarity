<!-- HEADER START -->
<!-- src: https://github.com/kyechan99/capsule-render -->
<p align="center"><a href="#">
    <img width="100%" height="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:002868,50:0BEC31, 100:0050F7, &height=220&section=header&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=40&text=Face%20Similarity%20Task" alt="header" />
</a></p>

<p align="center"><a href="https://github.com/JungKH33/face-similarity"><img src="./assets/logo.png", width=50%, height=50%, alt="logo"></a></p>


<p align="center">This project was carried out by <b><a href="https://github.com/yonsei-YAI">YAI 13th</a></b>, in cooperation with <b><a href="https://www.aipark.ai/">AIpark</a></b>.</p>

<p align="center">
<br>
<a href="mailto:melon345@yonsei.ac.kr">
    <img src="https://img.shields.io/badge/-Gmail-D14836?style=flat-square&logo=gmail&logoColor=white" alt="Gmail"/>
</a>
<a href="[https://binne.notion.site/YAIXPOZALabs-1-e679cf3cf3854ef69bfabc5a377cbea2](https://www.notion.so/binne/YAI-X-POZAlabs-852ef538af984d99abee33037751547c)">
    <img src="https://img.shields.io/badge/-Project%20Page-000000?style=flat-square&logo=notion&logoColor=white" alt="NOTION"/>
</a>
<a href="https://www.notion.so/y-ai/7b3780c8172c425d87f6174ed159d99a?pvs=4">
    <img src="https://img.shields.io/badge/-Full%20Report-dddddd?style=flat-square&logo=latex&logoColor=black" alt="REPORT"/>
</a>
</p>
<br>
<hr>
<!-- HEADER END -->

# Improving performance of Face Recogition for virtual human generation 👩‍💼
YAI x AIpark  <br>Face recognition model for virtual human generation<br>

# Members 👋
<b> <a href="https://github.com/JungKH33">정경훈</a></b>&nbsp; :&nbsp; YAI 13th&nbsp; /&nbsp; melon345@yonsei.ac.kr<br>
<b>  <a href="https://github.com/jmjmfasdf">서정민</a></b>&nbsp; :&nbsp; YAI 13th&nbsp; /&nbsp; jmme425@yonsei.ac.kr  <br>
<b> <a href="https://github.com/Nugu-ai">박승호</a></b>&nbsp; :&nbsp; YAI 13th&nbsp; /&nbsp; gomi0904@yonsei.ac.kr <br>
<b> <a href="https://github.com/Tim3s">홍지우</a></b>&nbsp; :&nbsp; YAI 13th&nbsp; /&nbsp; jiwoo0729@yonsei.ac.kr <br>
<hr>

# Getting Started 🔥
As we modify Deepface library to implement Adaface, please use our Deepface library.
```
$ git clone ~
$ cd deepface
$ pip install e.
```
For more details, please refer to our baseline [Deepface](https://github.com/serengil/deepface)

# Building on DeepFace for experiments 🏗️🔬

## 0. DeepFace - [Link](https://github.com/serengil/deepface)
Deepface is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python. It is a hybrid face recognition framework wrapping state-of-the-art models: VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace, Dlib, SFace and GhostFaceNet.

## 1. AdaFace - [Link](https://github.com/mk-minchul/AdaFace)
[AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.00964)

AdaFace is a novel approach for face recognition in low-quality datasets. AdaFace enhances the margin-based loss function by adapting to image quality, assigning varied importance to easy and hard samples based on their quality. By approximating image quality through feature norms, AdaFace outperforms state-of-the-art methods on multiple datasets, including IJB-B, IJB-C, IJB-S, and TinyFace.

## 2. Implement Adaface on Deepface by using ONNX
TBD

# Dataset 😊
We made our own datasets to evaluate various combination of model and backend(detection). To evaluate the robustness of each features, we prepared datasets that can represent key features affect face similarity. We picked 6 to 14 identities considering gender, race, age from the datasets below.

## 1. Digiface
Digiface is a collection of over one million diverse synthetic face images for face recognition. There are 720k images with 10K identities containing various face angle, accessories, emotion and hairstyle. We chose 6 identities for fast inference.

## 2. Hairstyle
We collected various hairstyle images generated from several GAN based models([Styleyourhair](https://github.com/Taeu/Style-Your-Hair), [Barbershop](https://github.com/ZPdesu/Barbershop)). This dataset contains one original image and several images based on oringal image.

## 3. Makeup
We collected various makeup images of same identity generated from GAN based models([LADN](https://github.com/wangguanzhi/LADN)). This dataset contains one original image and several makeup images based on original image. In addition, these dataset has subfolders for each identity, one is similar makeup images compared to original image, the other subfolder contains less similar images compared with original images.

## 4. Rotation
[Multi-PIE](https://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html) contains pictures of multiple angle of same identity. 


# Metrics 📋
To find best model combination, we made inference metric of our own. You can test our metric on the datasets above. The model combination we tested is shown as below.
```
model = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "SFace",
  "AdaFace"
]

backends = [
  'opencv', 
  'ssd', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]
```

The detail of how to use our code will be upload soon.


## Skills
Frameworks <br><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> 

## Citations
```bibtex
@misc{dai2019transformerxl,
      title={Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context}, 
      author={Zihang Dai and Zhilin Yang and Yiming Yang and Jaime Carbonell and Quoc V. Le and Ruslan Salakhutdinov},
      year={2019},
      eprint={1901.02860},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```bibtex
@misc{https://doi.org/10.48550/arxiv.1905.10887,
  doi = {10.48550/ARXIV.1905.10887},
  url = {https://arxiv.org/abs/1905.10887},
  author = {Ravuri, Suman and Vinyals, Oriol},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Classification Accuracy Score for Conditional Generative Models},
  publisher = {arXiv},
  year = {2019},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@inproceedings{hyun2022commu,
  title={Com{MU}: Dataset for Combinatorial Music Generation},
  author={Lee Hyun and Taehyun Kim and Hyolim Kang and Minjoo Ki and Hyeonchan Hwang and Kwanho Park and Sharang Han and Seon Joo Kim},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
}
```

