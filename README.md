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

* **pip requirements**: ```pip install -r requirements.txt``` 

> Its better to use a virtual environment 
> OR use conda-

* **conda**: use environment.yml: ```conda env create -f environment.yml```

* dev - cpu - test -install 

> stable test environment 

```python
pip install termcolor
pip install paddlepaddle
pip install paddleocr
pip uninstall protobuf
pip install --no-binary protobuf protobuf
```



# TODO (Dev branch)

**entry from**: ```test.ipynb```

- [ ] requirement versioning:
    - [ ] remove local cached versions
    - [ ] fix conda repo

- [x] pipe english recognizer paddle
- [x] pipe arabic recognizer paddle
- [x] line and word based document sorting

![ ](/tests/issue_check.png)

- [ ] integrate classifier model
- [ ] bangla recognizer: 
    - [ ] base version(temporary): easyocr integration with freelist crops (integrate easyocr and paddleocr)
    - [ ] cluster version (temporary): word modification and cleaning pipeline
- [ ] ```text_dict``` as universal call variable
    - [x] line based sort visualization 
    - [x] filter false word dets with line=-1 iden
    - [ ] chnage final text_dict as a line - word query: ```text_dict[line_no][word_no]={"crop":image,"box":free-format-location,"lang":language,"text":text}```

**current state**:

![](/tests/cs.png)



