# DBSR
Super-Resolution on Degraded Low-Resolution Images Using Convolutional Neural Networks (DBSR)

# DBSR Network

1. **DBSRCNN Network:**
Image Deblurring And Super-Resolution Using Deep Convolutional Neural Networks,
F. Albluwi, V. Krylov and R. Dahyot, IEEE International Workshop on Machine Learning for Signal Processing (MLSP 2018 <http://mlsp2018.conwiz.dk/home.htm> ), September 2018, Aalborg, Danemark.
![dbsrcnn arct](https://user-images.githubusercontent.com/16929158/45629859-4bd2dc80-ba8f-11e8-82f4-409c28a32777.png)

2. **DBSR Network:**
DBSR Network is an extension of DBSRCNN Network with extra 3 layers to enhance the extracted features inside the network.
![DBSR_network](https://user-images.githubusercontent.com/16929158/60619236-173ad200-9dd0-11e9-9ff9-2c3c3cefcda7.png)


# DBSRCNN-Keras
This code is to process the blurred low-resolution images to get deblurred high-residual images.

If this code is helpful for you, please cite this paper: Super-Resolution on Degraded Low-ResolutionImages Using Convolutional Neural Networks, F. Albluwi, V. Krylov and R. Dahyot, 27th European Signal Processing Conference (Eusipco 2019 ), September 2019.

## Dependencies
1. Python 3.6.5, and above.
2. TensorFlow 1.1.0, and above.
3. Keras 2.2.2, and above.
4. Matlab.
5. Matconvnet. 

## Generating data
1. blur images by Gaussian filter (imgaussfilt) at different levels (sigma = 1, 2, and 3).
2. resize images with 'bicubic' function using upscaling factor = 3, published papers recently generally use Matlab to produce low-resolution image.
3. For a fair comparison with SRCNN network; training set 291 images (Yang91 + 200 BSD) are used.

## Training
1. Generate training patches using Matlab: run generate_train.m and generate_test.m.
2. Use Keras with TensorFlow (tf) as a backend to train DBSRCNN model; Adam is used to optimizing the network for fast convergence: run DBSRCNN_train.py to produce DBSRCNN_blur model.
3. Convert Keras model to.Mat for testing using Matconvnet: run load_save.py first, then run save_model.m to produce Matconvnet model.
4. Run NB_SRCNN_Concat_blur_test.m in “test” folder to test the model; Set5 and Set14 are used as testing data.

## Some Qualitative Results

SISR with different models on images after Gaussian blur with different sigma = 2,3. The results show the non-blind and
blind scenarios. Each result is accompanied by zoom and PSNR(dB). In blind scenarios sigma = [0.5, 3].
![im1](https://user-images.githubusercontent.com/16929158/60619661-148cac80-9dd1-11e9-852b-f8ab44700a5e.png)
SISR performance of different models on Butterfly image after Gaussian blur at sigma = 2. In blind scenarios sigma = [0.5, 3].
![im2](https://user-images.githubusercontent.com/16929158/60619873-911f8b00-9dd1-11e9-8144-d8e8ae9ec90a.png)
