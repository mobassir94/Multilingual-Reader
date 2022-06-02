# Multilingual-Reader
in this repository we will share our works related to multilingual document reading (english,bangla and arabic).this is a work in progress,we will gradually update the repo inshaa allah



# Environment Setup

**DEV LOCAL ENVIRONMENT**  

```python
OS          : Ubuntu 20.04.3 LTS       
Memory      : 23.4 GiB 
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.36.8
```

**python requirements**
* dev - cpu - test -install 
> stable test environment 

* Manual Setup
```python
conda create -n mlreader python=3.8  -y
conda activate mlreader
conda install -n mlreader ipykernel --update-deps --force-reinstall -y
./install.sh
```
# Stack
* Line based detector model: ```paddleOCR en-dbnet```
* Word based detector model: ```paddleOCR ml-dbnet```
* English recognizer: ```paddleocr - en -SVTR_LCnet```
* Arabic recognizer: ```paddleocr - ar``` 
* Word classifier : Custom 
    * [Kaggle Dataset Link](https://www.kaggle.com/datasets/mobassir/multilingual-document-images)
    * [onnx conversion kernel](https://www.kaggle.com/code/nazmuddhohaansary/batchonnx/notebook)
    * [training kernel](/mlt_words/WordImage_LANG_Classifier_pytorch.ipynb)


# Change-log (Dev branch)
### 02-06-22
- [x] merging solved 
- [x] lang model auto download
- [x] classifier addition

# Docs
* ```docs/dev.md```: dev branch doc
* ```weights/weights.md```: custom weights integration doc
