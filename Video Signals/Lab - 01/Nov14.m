clc; clear all; close all
%% import image
I = imread('parrot.jpg');%store an image

figure
imshow(I)%show an image

imwrite(I,'my_image.png')% store an image
%% matrix is 3D. Each element - unsigned int (8 bit)
clc; clear all, close all;
img_gray = imread('Lena_grayscale.bmp');% grey scale
figure
imshow(img_gray)
% origin - top left corner (right - x; down - y)
% [y,x] - column, row
colorbar% intensity (0 - black, 1 - white) [0, 255]
%%
img_gray_16 = im2uint16(img_gray);% increase color depth to 16 bits
figure
imshow(img_gray_16)
colorbar
% no noticible changes, just scaled values (differ represent and different
% correspondance to intencities)
%% pixel depth - how the data is stored and treated in matlab, NOT SPATIAL RESOLUTION AND COLOR
img_gray_double = im2double(img_gray);
figure
imshow(img_gray_double)
colorbar% not limited to use a predefined set of values
%%
img_rgb = imread('parrot.jpg');%RGB image

figure
imshow(img_rgb)%show an image
% 3 different values for a given location
% treated as 8bit unsigned int
% EX. red = [255, 0, 0]
% X, Y, 3 - tensor dimention
%%
img_rgb_red_plane = img_rgb(:,:,1);%red part
img_rgb_green_plane = img_rgb(:,:,2);%green part
img_rgb_blue_plane = img_rgb(:,:,3);%blue part
% same spatial resolution
figure
subplot(2,2,1)
imshow(img_rgb)
title('truecolor image')

subplot(2,2,2)
imshow(img_rgb_red_plane)
title('truecolor image - red plane')

subplot(2,2,3)
imshow(img_rgb_green_plane)
title('truecolor image - green plane')

subplot(2,2,4)
imshow(img_rgb_blue_plane)
title('truecolor image - blue plane')
%% to colors
null_image_3 = uint8(zeros(size(img_rgb,1),size(img_rgb,2),size(img_rgb,3)));

img_rgb_red = null_image_3;
img_rgb_red(:,:,1) = img_rgb(:,:,1);

img_rgb_green = null_image_3;
img_rgb_green(:,:,2) = img_rgb(:,:,2);

img_rgb_blue = null_image_3;
img_rgb_blue(:,:,3) = img_rgb(:,:,3);

% VISUALIZE
figure
subplot(2,2,1)
imshow(img_rgb)
title('truecolor image')

subplot(2,2,2)
imshow(img_rgb_red)
title('truecolor image - red plane')

subplot(2,2,3)
imshow(img_rgb_green)
title('truecolor image - green plane')

subplot(2,2,4)
imshow(img_rgb_blue)
title('truecolor image - blue plane')
% high values of red are represented as red (same goes for the other channels
% img_rgb(1,1,:)
%% conversion between different colors representation
clc; clear all, close all

img_rgb = imread('parrot.jpg');
figure
imshow(img_rgb)
title('truecolor image')

% gray scale
img_gray = rgb2gray(img_rgb);% lin combination with specific weights
figure
imshow(img_gray)
title('grayscale image')
%% back and white
%img_bw_o5 = im2bw(img_gray, 0.5);% 0.5 - percentage
img_bw_o5 = im2double(img_gray) >0.5;% equivalent isong a threshold
figure% 0.1 - almost all white
imshow(img_bw_o5)
title('black and white image')
%%
img_bw_auto_global = imbinarize(img_gray);
figure
imshow(img_bw_auto_global)% chooses a threshold automatically
title ('bw image auto global ')
%%
img_bw_auto_local = imbinarize(img_gray,'adaptive');
figure
imshow(img_bw_auto_local)% chooses a threshold automatically
title ('bw image auto lcoal ')
% different threshold for each section
%%
im_bw_dither=dither(img_gray);% using just 0 and 1
figure
imshow(im_bw_dither)
title ('bw image (dither)')% more details are seen wrt loc and glob thresh.
% dithering from local info (which doesn't make sence) we recreate an original image
%% how to repres the color compressed
[img_ind, map_ind] = rgb2ind(img_rgb, 16,'nodither');% number of colors as an input parameter
% noditter vs ditter - ditter approximates to smoothen the transition 
% map - index-colors correspondance
figure
imshow(img_rgb)
title ('original image')
colorbar

figure
imshow(img_ind,map_ind)
title ('truecolor->indexed conversion(16 clolors)')
colorbar
% indexed has sharper transition between colors(instead of 256 we use 16 colors)

% go back to rgb
img_ind_rgb = ind2rgb(img_ind, map_ind);
figure
imshow(img_ind_rgb)
title ('indexed conversion(16 clolors)->truecolor')
colorbar
% only thing changed is representation (how the color is stored)
% we can't go back to the original one, since the number of info has been reduced
%% spatial representation
%% image resizing
clc; close all; clear all;
img_gray = imread("Lena_grayscale.bmp");
figure
imshow(img_gray)
title('original image')

img_gray_resized = imresize(img_gray, 1/3);
figure
imshow(img_gray_resized)
title('resized image 1/3')

% going back by enlarging(interpolation)

img_gray_enlarged_nn = imresize(img_gray_resized, 3, 'nearest');%neares neighbor
figure
imshow(img_gray_enlarged_nn)
title('enlarged image x3 (nearest neighbor)')
% 3x3 pixels will share the same value, therefore it looks pixelated
% you use values present in the orig picture

img_gray_enlarged_bl = imresize(img_gray_resized, 3, 'bilinear');%bilinear
figure
imshow(img_gray_enlarged_bl)
title('enlarged image x3 (bilinear)')% interpol linearly (plane, since 2D)

img_gray_enlarged_bc = imresize(img_gray_resized, 3, 'bicubic');%bicubic
figure% default; you can substitute 3 with [100, 200] <- image dim
imshow(img_gray_enlarged_bc)
title('enlarged image x3 (bicubic)')
% disadvantage - we go outside of the original range -> can gen new values
%% arithmetic operations on images
clc; clear all; close all;
img_gray = imread("Lena_grayscale.bmp");
img_rgb = imread("parrot.jpg");
img_gray2 = imresize(rgb2gray(img_rgb), size(img_gray)); %convert to gray with the propped dim

figure% 1
imshow(img_gray)
figure% 2
imshow(img_gray2)
figure% sum - brighter, substr - smaller
imshow(img_gray-img_gray2)% can have saturation problem
%% cropping
clc; close all; clear all;
img_rgb = imread("parrot.jpg");
% starting point, width, height
figure
imshow(imcrop(img_rgb, [1, 100, 200, 50]))% cut away some part
% no arguments specified - manually select in the picture

figure
imshow(img_rgb(100:150,1:201,:))% cut away some part
% FIRST Y DIMENTION, THEN X DIMENTION!
%% rotation
clc; close all; clear all
img_gray = imread("Lena_grayscale.bmp");
img_gray_rotated1 = imrotate(img_gray, 35);

figure% problem with the background + interpolation is used
imshow(img_gray_rotated1);% degrees

img_gray_rotated2 = imrotate(img_gray, 35, 'nearest', 'crop');
figure
imshow(img_gray_rotated2);% degrees
%% EXERCISE
% profile pictue to be cropped and resized (circular) 150x150; white background
% 1)crop; 2)resize; 3) background
clc; close all; clear all;

% what a user uploads
I = imread("Lena_grayscale.bmp");
% upper left corner - [75, 120]
tl = [75, 120];
br = [75, 120] + 220;

I_crop = I(tl(2):br(2), tl(1):br(1));

figure
imshow(I_crop)

target_size = 150;
I_crop = imresize(I_crop, [target_size,target_size]);
figure
imshow(I_crop)

% cut the circle
for i = 1:target_size%row
    for j = 1:target_size%column
        % calculate the distance between the pizel and the center
        dist_px = norm([i,j] - [target_size,target_size]./2);% middle point
        if(dist_px > (target_size/2))%radius
            I_crop(i,j) = 255;% white; 0 - black background
        end
    end
end

figure
imshow(I_crop)