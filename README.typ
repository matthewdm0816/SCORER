#set text(
  font: "New Computer Modern",
  // size: 10pt,
)
#set rect(radius: (
    left: 3pt,
    top-right: 3pt,
    bottom-right: 3pt,
  ),)
#show raw: it => [
  #set text(font: "Source Code Pro", ligatures: true)
  #it
]
#show link: underline
#set heading(numbering: "1.", bookmarked: true)
#let title = [Assigment 3: Image Change Captioning]
#set page(header: locate(loc => {
  let pagenumber = loc.page()
  let header = [#smallcaps()[#title]]
  if pagenumber > 1 {
    [#h(1fr) #header]
  }
}), numbering: "1 / 1",)
#show "CLEVR-Change": name => smallcaps[#name]


// document begins

#heading(numbering: none, outlined: false)[
  #text(size: 20pt)[
    #title
  ]
]
\

This package contains the code for *image change captioning*. The image change captioning task require a model to compare two images and describe what is changed. The code is adapted from @tu2023self.

In this assignment, you'll need to 
1. Implement several code snippets, including 
  - InfoNCE Loss Function
  - Position Encoding in Transformer
  - Attention in Transformer
2. Run the training and evaluation scripts on your own, on a small subset of CLEVR-Change dataset.


= Environment Configuration <installation>
+ Requirement: Linux + NVIDIA GPU. 
  - Not tested on Windows/MacOS. 
  - Not tested on Google Colab. But you can try to upload the files and run through the notebook.
+ Make virtual environment with Python 3.8
+ Install PyTorch 1.8. Refer to #link("https://pytorch.org/get-started/previous-versions/#v182-with-lts-support")[Installing previous versions of PyTorch].
+ Install requirements (```sh pip install -r requirements.txt```)
+ Download `en_core_web_sm` english text model for spaCy, by ```sh python3 -m spacy download en_core_web_sm```
+ Setup COCO caption eval tools (#link("https://github.com/mtanti/coco-caption")[github]). Or ```sh pip install cocoevalcaps```.

= Data Preparation <data>
1. Download CLEVR-Change data from here: #link(
    "https://drive.google.com/file/d/1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe/view?usp=sharing",
  )[google drive link], and unzip.
  #rect[
  ```shell-unix-generic
  tar -xzvf clevr_change.tar.gz
  ```]

  Extracting this file will create `data` directory and fill it up with CLEVR-Change dataset.

2. Preprocess data 

  - We are providing the preprocessed data here: #link(
  "https://drive.google.com/file/d/1FA9mYGIoQ_DvprP6rtdEve921UXewSGF/view?usp=sharing",)[google drive link]. You can skip the procedures explained below and just download them using the following command:
    #rect[
    ```shell-unix-generic
    cd data
    tar -xzvf clevr_change_features.tar.gz
    ```
    ]
  - Extract visual features using ImageNet pretrained ResNet-101:
    #rect[
    ```shell-unix-generic
    # processing default images
    python scripts/extract_features.py --input_image_dir ./data/images --output_dir ./data/features --batch_size 128
    
    # processing semantically changes images
    python scripts/extract_features.py --input_image_dir ./data/sc_images --output_dir ./data/sc_features --batch_size 128
    
    # processing distractor images
    python scripts/extract_features.py --input_image_dir ./data/nsc_images --output_dir ./data/nsc_features --batch_size 128
    ```]
  
  - Build vocab and label files using caption annotations:
    #rect[
    ```shell-unix-generic
    python scripts/preprocess_captions_transformer.py --input_captions_json ./data/change_captions.json --input_neg_captions_json ./data/no_change_captions.json --input_image_dir ./data/images --split_json ./data/splits.json --output_vocab_json ./data/transformer_vocab.json --output_h5 ./data/transformer_labels.h5
    ```]

= Code Implementation
In this assignment, you will need to implement some code snippets in the model architecture to finally train the model:
1. InfoNCE Loss. Position: \
  - `models/CBR.py, Line 13`
  - `utils/utils.py, Line 292`
  - Note: the two snippets should be identical. You only need to calculate the InfoNCE loss with given unnormalized similarity matrix.
2. Position Encoding. Position: \
  - `models/transformer_decoder.py, Line 31`
  - Note: you're going to implement the sinusoidal position encoding as proposed in original Transformer@NIPS2017_3f5ee243 paper. Also refer to #link("https://kazemnejad.com/blog/transformer_architecture_positional_encoding/")[this blog] for a quick explanation.
3. Attention Mechanisms. Position: \
  - Self-attention: `models/transformer_decoder.py, Line 131`
  - Cross-attention: `models/transformer_decoder.py, Line 173`
  - Attention: `models/SCORER.py, Line 48`
  - Note: these snippets are mostly similar. No residual links or LayerNorms to add, as we have already put them in the place if needed.
  

The places you need to modify in source files start with a comment including ```== To Implement ==```. If you find it difficult to implement them, read the paper and remaining code to better comprehend how each component work together. You can also refer to public Transformer and InfoNCE codes for reference.

= Training <training>
To train the proposed method, run the following commands:
#rect[```sh
# create a directory or a symlink to save the experiments logs/snapshots etc.
mkdir experiments
# OR
ln -s $PATH_TO_DIR$ experiments

# this will start the visdom server for logging
# start the server on a tmux session since the server needs to be up during training
python -m visdom.server

# start training
python train.py --cfg configs/dynamic/transformer_quick.yaml
```]

Note that we use a fractional of the whole data (\~10%) for training in this assignment. If you want to try full training (takes \~2 4090 hours), replace the config file to `configs/dynamic/transformer.yaml` (no additional score!).

= Testing/Inference <testinginference>
To test/run inference on the test dataset, run the following command

#rect[```shell-unix-generic
python test.py --cfg configs/dynamic/transformer.yaml  --snapshot 10000 --gpu 1
```]

The command above will take the model snapshot at 10000th iteration and run inference using GPU ID 1.

= Evaluation <evaluation>
- Caption evaluation

Run the following command to run evaluation:

#rect[```shell-unix-generic
# This will run evaluation on the results generated from the validation set and print the best results
python evaluate.py --results_dir ./experiments/SCORER+CBR/eval_sents --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```]

Once the best model is found on the validation set, you can run inference on test set for that specific model using the command exlpained in the `Testing/Inference` section and then finally evaluate on test set:

#rect[```shell-unix-generic
python evaluate.py --results_dir ./experiments/SCORER+CBR/test_output/captions --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```]

The results are saved in `./experiments/SCORER+CBR/test_output/captions/eval_results.txt`

= Hand-In Requirements
You are going to hand in following meterials for scoring:
1. #smallcaps[*Python source files*]. \
  Only submit the four modified files: `models/CBR.py, models/transformer_decoder.py, utils/utils.py, models/SCORER.py`
2. #smallcaps[*Brief report*] of your own implementation, including the evaluation results on both validation and test set.

= Reference
#bibliography("bib.bib", title: none)

