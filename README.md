# Federated Foundation Model for Cardiac CT Imaging

Malte Tölle, Philipp Garthe, Clemens Scherer, Jan Moritz Seliger, Andreas Leha, Nina Krüger, Stefan Simm, Simon Martin, Sebastian Eble, Halvar Kelm, Moritz Bednorz, Florian André, Peter Bannas, Gerhard Diller, Norbert Frey, Stefan Groß, Anja Hennemuth, Lars Kaderali, Alexander Meyer, Eike Nagel, Stefan Orwat, Moritz Seiffert, Tim Friede, Tim Seidler, Sandy Engelhardt

Paper link: 

## Abstract

Federated learning (FL) is a renowned technique for utilizing decentralized data while preserving privacy. 
    However, real-world applications often involve inherent challenges such as partially labeled datasets, where not all clients possess expert annotations of all labels of interest, leaving large portions of unlabeled data unused. %inter-observer variability, and data quality across different clients. 
    In this study, we conduct the largest federated cardiac CT imaging analysis to date, focusing on partially labeled datasets ($n=8,124$) of Transcatheter Aortic Valve Implantation (TAVI) patients over eight hospital clients. %RETROSPECTIVE CLINICAL ROUTINE DATA
    Transformer architectures, which are the major building blocks of current  foundation models, have shown superior performance when trained on larger cohorts than traditional CNNs. 
    However, when trained on small task-specific labeled sample sizes, it is currently not feasible to exploit their underlying attention mechanism for improved performance. 
    Therefore, we developed a two-stage semi-supervised learning strategy that distills knowledge from several task-specific CNNs (landmark detection and segmentation of calcification) into a single transformer model by utilizing large amounts of unlabeled data typically residing unused in hospitals to mitigate these issues.
    This method not only improves the predictive accuracy and generalizability of transformer-based architectures but also facilitates the simultaneous learning of all partial labels within a single transformer model across the federation.
    Additionally, we show that our transformer-based model extracts more meaningful features for further downstream tasks than the UNet-based one by only training the last layer to also solve segmentation of coronary arteries.
    We make the code and weights of the final model openly available, which can serve as a foundation model for further research in cardiac CT imaging.

## BibTeX

```
@misc{toelle2024cardic-ct,
    title={Federated Foundation Model for Cardiac CT Imaging},
    author={Tölle, Malte and Garthe, Philipp and Scherer, Clemens and Seliger, Jan Moritz and Leha, Andreas and Krüger, Nina and Simm, Stefan and Martin, Simon and Eble, Sebastian and Kelm, Halvar and Bednorz, Moritz and André, Florian and Bannas, Peter and Diller, Gerhard and Frey, Norbert and Groß, Stefan and Hennemuth, Anja and Kaderali, Lars and Meyer, Alexander and Nagel, Eike and Orwat, Stefan and Seiffert, Moritz and Friede, Tim and Seidler, Tim and Engelhardt, Sandy},
    year={2024},
    doi={arXiv}
}
```

## Run Instructions

### Train Federated

With the `train_federated.py`-script multiple trainings are possible. This is determined with the `--task`-flag: 0=KD, 1=HPS, 2=MS, 3=Calcification. Training and testing can be performed on any location combination wanted with the `--locations`-flag. `num_rounds` on each client, federated `epochs`, `batch_size`, and per-client `test_ratio` can be set. The network can either be conditioned on a segmentation (`--condition_on_seg`) or the segmentation can be trained simultaneously with another head (`--output_seg`). If a SWIN-UNETR shall be used as model, the `--patches`-flag can be set out of memory requirements. A pretrained ckeckpoint can be loaded with the `--ckpt`-flag.

```
python train_federated.py \
    --task [0,1,2,3] \
    --locations [loc1, loc2, ...] \
    --mode [train, test] \
    --num_rounds 1 \
    --epochs 10 \
    --batch_size 4 \
    --test_ratio 0.2 \
    --test_on_gobal_updates \
    --test_on_local_updates \
    --exp_name federated_point_detection \
    --ckpt ckpt.pt \
    --condition_on_seg \
    --patches \
    --model_type swin_unetr
```

### Download Pretrained Checkpoints

The checkpoints will be made available upon acceptance.

### Prediction on local data

```
python tavi_predictor.py --fname ct.nii.gz --tmp_dir ./tmp
```

## Contact

Malte Tölle<br>
[malte.toelle@med.uni-heidelberg.de](mailto:malte.toelle@med.uni-heidelberg.de)<br>
[@maltetoelle](https://x.com/maltetoelle)<br>

Group Artificial Intelligence in Cardiovascular Medicine (AICM)<br>
Heidelberg University Hospital<br>
Im Neuenheimer Feld 410, 69120 Heidelberg, Germany