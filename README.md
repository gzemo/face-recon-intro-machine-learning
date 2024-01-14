# Face Recognition, Introduction to Machine Learning
---
*Face Recognition final project for the Introduction to Machine Learning course, University of Trento a.y. 2022-2023*

**Credits**: for model implementation and pre-trained architectures download please see the official *Deepface* (Serengil, S. I., & Ozpinar, A, 2020) repository: https://github.com/serengil/deepface

**Complete project report**: see `report.pdf`.

**Disclaimer**: the code comes as it is with no futher guarantee. For any high end face recognition pipeline it is strongly suggested to directly rely on the aforementioned repository.

---
## General description
The present work to address the Introduction to Machine Learning course final project competition on face recognition. In order to do so a **Tensorflow**-based face recognition sistem is presented to address face detection and image embedding extraction to further match the resulting representation to those belonging to a given test set according to similarity scores. 

Originally each group had been asked to submit their result to a private server in oder to evaluate the current model performance: this feature is disabled.

## Implementation

### Face detection
- Yolov8 (Redmon et al., 2016) pretrained explicitly on face detection (from https://docs.ultralytics.com/)

### Face embedding extraction
- FaceNet (Schroff et al., 2015)
- FaceNet512
- ArcFace (Deng et al., 2019)

### To be completed
Currently, no output is return to the user. An outcome rensponse is under-development

### How to run
- installation of dependencies find on the `requirements.txt`
- specify the parameters according to your local path in the `main.py` and execute in a dedicated shell


## References
Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face recognition. In Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition (pp. 4690-4699).

Huang, G. B., Mattar, M., Berg, T., & Learned-Miller, E. (2008, October). Labeled faces in the wild: A database for studying face
recognition in unconstrained environments. In Workshop on faces in Real-Life Images: detection, alignment, and recognition.

Klare, B. F., Klein, B., Taborsky, E., Blanton, A., Cheney, J., Allen, K., ... & Jain, A. K. (2015). Pushing the frontiers of unconstrained
face detection and recognition: Iarpa janus benchmark a. In Proceedings of the IEEE conference on computer vision and
pattern recognition (pp. 1931-1939).

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of
the IEEE conference on computer vision and pattern recognition (pp. 779-788).

Schroff, F., Kalenichenko, D., & Philbin, J. (2015). Facenet: A unified embedding for face recognition and clustering. In Proceedings
of the IEEE conference on computer vision and pattern recognition (pp. 815-823).

Serengil, S. I., & Ozpinar, A. (2020, October). Lightface: A hybrid deep face recognition framework. In 2020 Innovations in Intelligent
Systems and Applications Conference (ASYU) (pp. 1-5). IEEE.

Wolf, L., Hassner, T., & Maoz, I. (2011, June). Face recognition in unconstrained videos with matched background similarity. In
CVPR 2011 (pp. 529-534). IEEE.

Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks.
IEEE signal processing letters, 23(10), 1499-1503.


---
Tested on:
- Ubuntu 22.04 LTS
- python 3.9


