# Datasets and weights used for this task can be found here : https://www.kaggle.com/datasets/mobassir/multilingual-document-images

1.doctr_word_cropper.ipynb -> uses doctr to demonstrate how multilingual word image cropping works

2.paddle_word_cropper.ipynb -> uses paddleocr to demonstrate how multilingual word image cropping works

3. pdf_to_imgs.py -> provides the python script that can be used to convert pdf documents into images

4.WordImage_LANG_Classifier_pytorch.ipynb -> contains code(train+inference) for multilingual word image classification(model training log is attached in this notebook in markdown cell)

Note : in our experiment we've observed that paddle word cropper is far better than doctr word cropped, so for mlt word image language classification task we used paddle word cropper.



