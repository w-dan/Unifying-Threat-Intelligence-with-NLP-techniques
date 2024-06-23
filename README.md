# Unifying Threat Intelligence with NLP Techniques

![UPM Logo](https://www.upm.es/sfs/Rectorado/Gabinete%20del%20Rector/Logos/UPM/Logotipo%20con%20Leyenda/LOGOTIPO%20leyenda%20color%20PDF%20p_.png)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)


This project aims to automate and enhance the mapping of unstructured threat intelligence APT reports to the standardized [STIX 2](https://oasis-open.github.io/cti-documentation/stix/intro) format using [Natural Language Processing](https://github.com/topics/awesome-nlp) techniques.

The project's objective is to enhance threat intelligence processes in three different ways:
* automatically and collectively analyze reports of multiple types
* improving the quality and comprehensiveness of the information
* ensuring transparency and reliability, particularly for non-expert entities involved in subsequent decision-making



# Contents
* `mapper`: Contains the main script and logic for processing threat intelligence reports. The script generates a STIX 2 file.
* `models`: Directory for storing trained models (currently BERT).
* `models-training`: Scripts and notebooks documenting training experiments.
* `preprocessing-and-EDA`: Data preprocessing and exploratory data analysis scripts.
* `samples`: Example data and sample PDFs for testing.
* `scrapers`: Web scrapers for collecting threat intelligence data.



## Folder structure
```txt
.
├── LICENSE
├── README.md
├── mapper
│   ├── __pycache__
│   │   ├── constants.cpython-310.pyc
│   │   └── utils.cpython-310.pyc
│   ├── constants.py
│   ├── mapper.py
│   └── utils.py
├── models
│   ├── README.md
│   └── bert_multitag
│       ├── config.json
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.txt
├── models-training
│   ├── README.md
│   └── bert_multitag.ipynb
├── preprocessing-and-EDA
│   └── README.md
├── requierements.txt
├── samples
│   └── sample.pdf
└── scrapers
    └── blackberry_insert.py
```

# Recommended use
It is recommended to create a virtual environment with your environment manager of choice to avoid conflicts between dependencies.

Required packages and libraries can be installed running:
> pip install -r requirements.txt

The main script, `mapper.py`, can be found under the `mapper` folder.

## Contribute

### Commit && PR Messages

```txt
[MODULE][FIX|ADD|DELETE] Summary of modifications
```
Where each of the folders under the repository's root directory may be understood as a module.


#### Example

```txt
* [README][ADD] Execution && Component diagram
* [MODELS][ADD] A new model
* [README][DELETED]
* [MAPPER][FIX] 
```