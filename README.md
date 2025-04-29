# Monster MIDI Dataset
## Giant searchable raw MIDI dataset for MIR and Music AI purposes

![Monster-MIDI-Dataset-Logo (8)](https://github.com/asigalov61/Monster-MIDI-Dataset/assets/56325539/d5648673-97c1-40e3-ad57-c03c639592a3)

***

## Monster MIDI Dataset GPU Search and Filter

### [NEW] Monster GPU/CPU Search and Filter stand-alone Python module with improved matching

#### Installation

##### Install requirements

```sh
!git clone --depth 1 https://github.com/asigalov61/Monster-MIDI-Dataset

!pip install cupy-cuda12x
!pip install numpy==1.26.4
!pip install huggingface_hub
!pip install hf-transfer
!pip install ipywidgets
!pip install tqdm
```

##### Import modules

```python
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import hf_hub_download

%cd ./Monster-MIDI-Dataset
import monster_search_and_filter
%cd ..
```

##### Download and unzip Monster MIDI dataset

```python
hf_hub_download(repo_id='projectlosangeles/Monster-MIDI-Dataset',
                repo_type='dataset',
                filename='Monster-MIDI-Dataset-Ver-1-0-CC-BY-NC-SA.zip',
                local_dir='./Monster-MIDI-Dataset/'
                )
```

```sh
%cd ./Monster-MIDI-Dataset
!unzip -o Monster-MIDI-Dataset-Ver-1-0-CC-BY-NC-SA.zip > /dev/null
%cd ..
```

##### Run the search

```python
sigs_data_path = './Monster-MIDI-Dataset/SIGNATURES_DATA/MONSTER_SIGNATURES_DATA.pickle'

sigs_data = monster_search_and_filter.load_pickle(sigs_data_path)
sigs_dicts = monster_search_and_filter.load_signatures(sigs_data)

# Please note that you will need at least 80GB RAM or VRAM to run the search
X, global_union = monster_search_and_filter.precompute_signatures(sigs_dicts)

# IO dirs will be created on the first function run
# Make sure to put your master MIDIs into created Master-MIDI-Dataset dir
monster_search_and_filter.search_and_filter(sigs_dicts, X, global_union)
```

### [LEGACY]

[![Open In Colab][colab-badge]][colab-notebook1]

[colab-notebook1]: <https://colab.research.google.com/github/asigalov61/Monster-MIDI-Dataset/blob/main/Monster_MIDI_Dataset_GPU_Search_and_Filter.ipynb>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

### Search, filter and explore Monster MIDI Dataset :)

#### PLEASE NOTE: Google Colab Pro or Pro+ subscription/A100 GPU is required to use the provided colab/code because of the size of the dataset and its data files

***

## Monster MIDI Dataset Sample Search Results

### Here are the [Monster MIDI Dataset Sample Search Results](https://huggingface.co/datasets/projectlosangeles/Monster-MIDI-Dataset/blob/main/Monster_MIDI_Dataset_Search_Results_Ver_1_0_CC_BY_NC_SA.zip)

### It takes about one hour on A100 GPU to do a full search on 285 source MIDIs

### Please also check out [Quad Music Transformer](https://github.com/asigalov61/Quad-Music-Transformer) which was trained using these sample search results

***

## Monster Music Transformer

### Here is the large model trained on the full Monster MIDI Dataset to demo the dataset in action :)

[![Open In Colab][colab-badge]][colab-notebook2]

[colab-notebook2]: <https://colab.research.google.com/github/asigalov61/Monster-MIDI-Dataset/blob/main/Monster_Music_Transformer.ipynb>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

### Model was trained on full Monster MIDI Dataset for 65 hours (1 epoch) @ 4 batches on a single H100 GPU
### This model can be used for music generation/composition or for (dataset) embeddings exploration

***

### Enjoy and please CC BY-NC-SA :)

***

### Project Los Angeles
### Tegridy Code 2025
