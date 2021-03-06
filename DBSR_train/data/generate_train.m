clear;close all;
%% settings
folder = 'Train_291_aug';
savepath = 'train_291_aug_scale3_blur1.h5';
size_input = 31;
size_label = 31;
stride = 21;

%% scale factors
scale = 3;  % s = 2, 3, or 4

%% blurring level
blur = 1; % 1, 2, 3, or 4  % blur images

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];

for i = 1 : length(filepaths)
  
    image = imread(fullfile(folder,filepaths(i).name));
    if size(image,3)==3            
        image = rgb2ycbcr(image);
        image = im2double(image(:, :, 1));
        im_label = modcrop(image, scale);
        [hei,wid] = size(im_label);
        im_input=imgaussfilt(im_label, blur);
        im_input = imresize(imresize(im_input,1/scale,'bicubic'),[hei,wid],'bicubic');  % bicubic LR
		
        for x = 1 : stride : hei-size_input+1
            for y = 1 :stride : wid-size_input+1
			
                subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
                subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
                count=count+1;
                
                data(:, :, 1, count) = subim_input;
                label(:, :, 1, count) = subim_label;
				
            end
        end
    end  
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    %batchno
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
