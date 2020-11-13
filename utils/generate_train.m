clear;
close all;
folder = 'E:\Project\VideoEnhancement\EDSR\datasets\train\DIV2K_train_HR\';

% savepath = 'E:\Project\VideoEnhancement\pytorch-edsr-master\data\edsr_x4.h5';

%% scale factors
scale = 4;

size_label = 192;
size_input = size_label/scale;
stride = 96;

%% downsizing
downsizes = [1,0.7,0.5];

% data = zeros(size_input, size_input, 3, 1);
% label = zeros(size_label, size_label, 3, 1);
% 
% count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];    % 注意文件格式

length(filepaths)

for i = 1 : length(filepaths)
    for flip = 1: 3
        for degree = 1 : 4
            for downsize = 1 : length(downsizes)
                image_name = filepaths(i).name; % 图像名
                image = imread(fullfile(folder,image_name));
                if flip == 1
                    image = flipdim(image ,1);
                end
                if flip == 2
                    image = flipdim(image ,2);
                end
                
                image = imrotate(image, 90 * (degree - 1));
                image = imresize(image,downsizes(downsize),'bicubic');

                if size(image,3)==3
                    %image = rgb2ycbcr(image);
                    image = im2double(image);
                    im_label = modcrop(image, scale);
                    [hei,wid, c] = size(im_label);
                else
                    image = label2rgb(image);
                    image = im2double(image);
                    im_label = modcrop(image, scale);
                    [hei,wid, c] = size(im_label);
                    
                end
                    filepaths(i).name
                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            subim_label = im_label(x : x+size_label-1, y : y+size_label-1, :);
                            subim_input = imresize(subim_label,1/scale,'bicubic');
                            save_LR_image_name = strcat(folder,'LR\',int2str(downsize),'_',int2str(degree),'_',int2str(flip),'_',image_name);
                            save_HR_image_name = strcat(folder,'HR\',int2str(downsize),'_',int2str(degree),'_',int2str(flip),'_',image_name);
                            fprintf('%s\n',save_HR_image_name)
                            imwrite(subim_input, save_LR_image_name);
                            imwrite(subim_label, save_HR_image_name);
%                             figure;
%                             imshow(subim_input);
%                             figure;
%                             imshow(subim_label);
%                             count=count+1;
%                             data(:, :, :, count) = subim_input;
%                             label(:, :, :, count) = subim_label;
                        end
                    end
               
            end
        end
    end
end

% order = randperm(count);
% data = data(:, :, :, order);
% label = label(:, :, :, order);

% %% writing to HDF5
% chunksz = 64;
% created_flag = false;
% totalct = 0;
% 
% for batchno = 1:floor(count/chunksz)
%     batchno
%     last_read=(batchno-1)*chunksz;
%     batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
%     batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
%     startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
%     curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
%     created_flag = true;
%     totalct = curr_dat_sz(end);
% end
% 
% % h5disp(savepath);