# Revisiting spatio-temporal layouts for compositional action recognition

[![Conference](https://img.shields.io/badge/BMVC%20Oral-2021-purple.svg?style=for-the-badge&color=f1e3ff&labelColor=purple)](https://www.bmvc2021-virtualconference.com/assets/papers/0974.pdf)    [![arXiv](https://img.shields.io/badge/arXiv-2111.01936-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2111.01936)

Codebase for [Revisiting spatio-temporal layouts for compositional action recognition](https://arxiv.org/abs/2111.01936).

## Dependencies

If you use [Poetry](https://python-poetry.org/), running ```poetry install``` inside the project should suffice.

## Preparing the data

### Something-Something and Something-Else

You need to download the [data splits and labels](https://github.com/joaanna/something_else/tree/master/code/dataset_splits), the [annotations](https://drive.google.com/drive/folders/1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ), and the [video sizes](https://drive.google.com/file/d/1ANaDAxXoA63CA9zXalnmaskqfO4cftW4/view?usp=sharing). Make sure that the annotations for the split you want to create datasets for are in a single directory. Then, use ```create_something_datasets.py``` to create the training and test datasets as:

```python
python src/create_something_datasets.py --train_data_path "data/path-to-the-train-file.json" --val_data_path "data/path-to-the-val-file.json" --annotations_path "data/all-annotations-for-the-split/"
```

### Action-Genome

You need to download the [Action Genome data](https://drive.google.com/drive/folders/1LGGPK_QgGbh9gH9SDFv_9LIhBliZbZys) and the [Charades data](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip). Then, use ```create_action_genome_datasets.py``` to create the training and test datasets as:

```python
python src/create_action_genome_datasets.py --action_genome_path "data/path-to-action-genome" --charades_path "data/path-to-charades" --save_datasets_path "data/directory-where-the-data-will-be-saved"
```

## Training and Inference

The codebase currently only supports training and inference of STLT models. Refer to the ```train_stlt.py``` and the ```inference_stlt.py``` scripts. More detailed instructions will be released soon.

## Model Zoo

| Model | Dataset | Download |
| :--- | :--- | :--- |
| STLT | Something-Else Compositional Split Detections | [Link](https://drive.google.com/file/d/1mSwN68F6UgaZsJ91hFt9up4XPlnP8ouz/view?usp=sharing) |
| STLT | Something-Else Compositional Split Oracle | [Link](https://drive.google.com/file/d/1PSIEgGhE9XwLwW-XMvWiRUZZhCVeSxbT/view?usp=sharing) |
| STLT backbone | Something-Else Detections Base split | [Link](https://drive.google.com/file/d/1CnQ89ipXjFkSkWIxaMzDVszlQ3LJ1ctK/view?usp=sharing) |
| STLT | Something-Else 10-shot Detections | [Link](https://drive.google.com/file/d/1W3ezhdTW7xLurSfiW36QIzeR21Zcatmt/view?usp=sharing) |
| STLT | Something-Else 5-shot Detections | [Link](https://drive.google.com/file/d/1V98gUlQitPB6uQ0pYBfCc9_vYXcsvajO/view?usp=sharing) |
| STLT backbone | Something-Else Oracle Base split | [Link](https://drive.google.com/file/d/1hIuDy8tmD83ibY0oT1fhQBpUSs8qMsWK/view?usp=sharing) |
| STLT | Something-Else 10-shot Oracle | [Link](https://drive.google.com/file/d/10YkKPXNrjQkMIxrFSMb4lR_csgnLYSyb/view?usp=sharing) |
| STLT | Something-Else 5-shot Oracle | [Link](https://drive.google.com/file/d/1_4yxNvgMT_mzKAveXdOzTYo53ZOKwT22/view?usp=sharing) |
| STLT | Something-Something V2 Regular Split | [Link](https://drive.google.com/file/d/1aBMpqpJ2H6prF5hfBaZ5u9iyslL1jSdv/view?usp=sharing) |
| STLT | Action Genome Oracle | [Link](https://drive.google.com/file/d/1xsQcosB-kNHW9WXRseR7jgMs_B8WsaG8/view?usp=sharing) |
| STLT | Action Genome Detections | [Link](https://drive.google.com/file/d/1L1mfNIhI938UyM3lIfHFrmx7k5DZzf7m/view?usp=sharing) |

More models trained on Something-Else and Charades/Action Genome will be released soon.

## Citation

If you find our code useful for your own research please use the following BibTeX entry.

```tex
@article{radevski2021revisiting,
  title={Revisiting spatio-temporal layouts for compositional action recognition},
  author={Radevski, Gorjan and Moens, Marie-Francine and Tuytelaars, Tinne},
  journal={arXiv preprint arXiv:2111.01936},
  year={2021}
}
```
