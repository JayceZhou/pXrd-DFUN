## This is the official implementation of the paper "Towards Class Imbalance and Uncertainty in Powder XRD Analysis: A Dual-Channel Fusion Network for Space Group Classification"

### overview
Accurate and automated identification of the 230 crystal space groups from pXRD patterns is a critical but challenging task in materials science. Standard deep learning models often fail when applied to real-world experimental data due to noise, severe class imbalance, and the gap between simulated and experimental domains. Furthermore, their "black box" nature limits their trustworthiness in scientific applications. DFUN is designed to address these challenges with a complete and reliable framework.

### Project directory structure
```
📦DFUN
 ┣ 📂generation
 ┣ 📂models
 ┃ ┣ 📂modules
 ┣ 📂utils
 ┣ 📜api_results_cache.pkl
 ┣ 📜gen.ipynb
 ┣ 📜mpids.npy
 ┣ 📜train.py
 ┗ 📜uncertainty_train.py
```
### How to Run
  - Experimental environment: Python3.9.20/pytorch2.5.1
  - Generation:
  ```
  generation.py
  ```
  - Evaluation: quick start
  ```
  train.py
  uncertainty_train.py
  ```

### Acknowledgements

We would like to express our sincere gratitude to the related works and open-source codes that have served as inspiration for our project:

- Oviedo, F., Ren, Z., Sun, S., Settens, C., Liu, Z., Hartono, N. T. P., ... & Buonassisi, T. (2019). Fast and interpretable classification of small X-ray diffraction datasets using data augmentation and deep neural networks. npj Computational Materials, 5(1), 60. [[paper]](https://doi.org/10.1038/s41524-019-0196-x) [[code]](https://github.com/PV-Lab/autoXRD).
- Salgado, J. E., Lerman, S., Du, Z., Xu, C., & Abdolrahim, N. (2023). Automated classification of big X-ray diffraction data using deep learning models. npj Computational Materials, 9(1), 214. [[paper]](https://www.nature.com/articles/s41524-023-01164-8) [[code]](https://github.com/AGI-init/XRDs).
- Adachi, M. Mixture-of-Experts Ensemble with Hierarchical Deep Metric Learning for Spectroscopic Identification. [[paper]](https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_9.pdf) [[code]](https://github.com/ma921/XRDidentifier?tab=readme-ov-file).
- opXRD database [https://xrd.aimat.science](https://xrd.aimat.science).
- RRUFF database [https://rruff.info/](https://rruff.info/)
- Materials Project database [https://next-gen.materialsproject.org/](https://next-gen.materialsproject.org/)
