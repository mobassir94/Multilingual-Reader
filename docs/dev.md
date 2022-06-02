# TODO (Dev branch)

**entry from**: ```test.ipynb```

- [x] pipe english recognizer paddle
- [x] pipe arabic recognizer paddle
- [x] line and word based document sorting
- [x] requirement versioning
- [x] integrate classifier model
- [ ] bangla recognizer: 
    - [ ] base version(temporary): easyocr integration with freelist crops (integrate easyocr and paddleocr)
    - [ ] cluster version (temporary): word modification and cleaning pipeline
- [ ] ```text_dict``` as universal call variable
    - [x] line based sort visualization 
    - [x] filter false word dets with line=-1 iden
    - [ ] chnage final text_dict as a line - word query: ```text_dict[line_no][word_no]={"crop":image,"box":free-format-location,"lang":language,"text":text}```





**current state**:

* line-word image with filtered noise

![ ](issue_check.png)

* current text_dict 

![](cs.png)