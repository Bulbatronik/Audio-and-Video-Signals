clc;clear all;close all;
%a
img = im2double(imread('parking_lot.png'));
figure
imshow(img)

%b
img_gray = rgb2gray(img);
H = fspecial('log', 3, 0.5);% laplacian of gaussian!
img_log = imfilter(img_gray,H,'symmetric','conv');
figure
imagesc(img_log)

%c
img_log = abs(img_log);
max_img_log = max(img_log(:));
mask = im2bw(img_log,0.05*max_img_log);
figure
imagesc(mask)

%d
mask = medfilt2(mask, [7,7]);
figure
imagesc(mask)

%e
free_count = 0;
free_vis = zeros(size(img));

%f
p_h = 50;
p_w = 25;
thr = p_h*p_w*0.3; %30% of the area
for i = 1:p_w:size(img,2)
    for j = 1:p_h:size(img,1)
        img_crop = mask(j:(j+p_h-1),i:(i+p_w-1));

        count = sum(img_crop(:));
        if (count<thr)
            free_count = free_count + 1;
            free_vis(j:(j+p_h-1),i:(i+p_w-1),1) = 0;%r 
            free_vis(j:(j+p_h-1),i:(i+p_w-1),2) = 1;%g
            free_vis(j:(j+p_h-1),i:(i+p_w-1),3) = 0;%b 
        else
            free_vis(j:(j+p_h-1),i:(i+p_w-1),1) = 1;%r 
            free_vis(j:(j+p_h-1),i:(i+p_w-1),2) = 0;%g
            free_vis(j:(j+p_h-1),i:(i+p_w-1),3) = 0;%b 
        end
    end
end

%g
figure
imshow(0.3*free_vis+0.7*img)
