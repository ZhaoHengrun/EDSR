close all;      %生成一组原图和小图
clear;
up_scale = 4;   % 缩放比例
file_path = 'E:\Project\VideoEnhancement\EDSR\datasets\test\DIV2K\';% 图像文件夹路径
img_path_list = dir(strcat(file_path,'*.png'));%获取该文件夹中所有.jpg格式的图像
img_num = length(img_path_list);%获取图像总数
psnr_avr = 0;   %psnr平均值
if img_num > 0 %有满足条件的图像
    for pn = 1:img_num %逐一读取图像
        image_name = img_path_list(pn).name;% 图像名
        i1 =  imread(strcat(file_path,image_name));%读取图像
        fprintf('%d %s\n',pn,strcat(file_path,image_name));% 显示正在处理的图像名
        i1 = modcrop(i1, up_scale);
%         if size(i1,3)>1
%             i1 = rgb2ycbcr(i1);
%             i1 = i1(:, :, 1);
%         end
        im_gnd = i1;
        im_gnd = single(im_gnd)/255;
        i2 = imresize(im_gnd, 1/up_scale, 'bicubic');
%         i2 = imresize(i2, up_scale, 'bicubic');       % bicubic

%         im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
%         i2 = shave(uint8(i2 * 255), [up_scale, up_scale]);
        
%         [psnr_val, snr] = psnr(im_gnd, i2);
%         psnr_avr = psnr_avr + psnr_val;
%         fprintf('psnr: %0.4f\n', psnr_val);
        imwrite(i2, strcat(file_path,'/LR/',image_name));
        imwrite(im_gnd, strcat(file_path,'/HR/',image_name));
    end
    psnr_avr = psnr_avr / img_num;
    fprintf('\npsnr_avr:%0.4f\n', psnr_avr);
else
    fprintf('no image');
end

% psnr_bic = compute_psnr(im_gnd,im_b)