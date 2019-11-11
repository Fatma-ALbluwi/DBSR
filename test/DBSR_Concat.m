% the Shallow and Deep DBSR for n layers + one concatente layer:
% DBSR is for Non-blind deblurring super-resolution CNN

function im_h_y = DBSR_Concat(im_l_y, model, scale)

weight = model.weight;
bias = model.bias;
im_y = single(imresize(im_l_y,scale,'bicubic'));

%% the first layer
convfea1 = vl_nnconv(im_y,weight{1},bias{1}, 'Pad',4);
convfea1 = vl_nnrelu(convfea1);

%% the second layer
convfea2 = vl_nnconv(convfea1,weight{2},bias{2}, 'Pad', 2);
convfea2 = vl_nnrelu(convfea2);

%% concatenated layer
convfea12 = vl_nnconcat({convfea1, convfea2});

%% mapping layer
convfea3 = vl_nnconv(convfea12,weight{3},bias{3}, 'Pad', 2);
convfea3 = vl_nnrelu(convfea3);

%% for 8 layers + one concatente layer: 
convfea4 = vl_nnconv(convfea3,weight{4},bias{4}, 'Pad', 2);
convfea4 = vl_nnrelu(convfea4);
convfea5 = vl_nnconv(convfea4,weight{5},bias{5}, 'Pad', 2);
convfea5 = vl_nnrelu(convfea5);
convfea6 = vl_nnconv(convfea5,weight{6},bias{6}, 'Pad', 2);
convfea6 = vl_nnrelu(convfea6);
convfea7 = vl_nnconv(convfea6,weight{7},bias{7}, 'Pad', 2);
convfea7 = vl_nnrelu(convfea7);
convfea8 = vl_nnconv(convfea7,weight{8},bias{8}, 'Pad', 2);  
 im_h_y = convfea8;
