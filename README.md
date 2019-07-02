# DBSR
Super-Resolution on Degraded Low-ResolutionImages Using Convolutional Neural Networks (DBSR)

# DBSR Network


# DBSRCNN-Keras
This code is to process the blurred low-resolution images to get deblurred high-residual images.

If this code is helpful for you, please cite this paper: Image Deblurring And Super-Resolution Using Deep Convolutional Neural Networks, F. Albluwi, V. Krylov and R. Dahyot, 27th European Signal Processing Conference (Eusipco 2019 ), September 2019.

## Dependencies
1. Python 3.6.5
2. TensorFlow 1.1.0.
3. Keras 2.2.2.
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

