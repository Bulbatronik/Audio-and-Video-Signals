%% Import/export of images
clc
clear all
close all

%Import images into matlab
I = imread('parrot.jpg');
size(I)
whos I

%image visualization
figure
imshow(I)

%save image to a file
imwrite (I, 'my_image.png');
imfinfo('my_image.png')

%% Data representation (pixel depth)

clc;
clear all;
close all;

img_gray = imread('Lena_grayscale.bmp'); 

figure 
imshow(img_gray)
colorbar 
title('uint8 grayscale image [0,255]')

% grayscale uint8 --> grayscale uint16
img_gray_uint16 = im2uint16(img_gray);
figure 
imshow(img_gray_uint16)
colorbar
title('uint8 --> uint16 [0,65535] grayscale conversion')


% grayscale uint8 --> grayscale double
img_gray_double = im2double(img_gray);
figure
imshow(img_gray_double)
colorbar
title('uint8 --> double [0,1] grayscale conversion')

% im2uint8, im2uint16, im2int16, im2single

%% Truecolor image

clc
clear all
close all

% Truecolor images (a.k.a. RGB images)
img_rgb = imread('parrot.jpg');

% what do these 3 channels represent?
img_rgb_red_plane = img_rgb(:,:,1); %in this way we obtain a 1 channel image 
img_rgb_green_plane = img_rgb(:,:,2);
img_rgb_blue_plane = img_rgb(:,:,3);

% let's visulize each color plane separatly
figure 
subplot(2,2,1)
imshow(img_rgb)
title('Truecolor image')
subplot(2,2,2)
imshow(img_rgb_red_plane) 
title('Truecolor image - RED channel');
subplot(2,2,3)
imshow(img_rgb_green_plane) 
title('Truecolor image - GREEN channel');
subplot(2,2,4)
imshow(img_rgb_blue_plane) 
title('Truecolor image - BLUE channel');

%more clear visualization
null_image_3 = uint8(zeros(size(img_rgb,1), size(img_rgb,2), 3));
img_rgb_red = null_image_3;
img_rgb_red(:,:,1) = img_rgb(:,:,1);
img_rgb_green = null_image_3;
img_rgb_green(:,:,2) = img_rgb(:,:,2);
img_rgb_blue = null_image_3;
img_rgb_blue(:,:,3) = img_rgb(:,:,3);

figure 
subplot(2,2,1)
imshow(img_rgb)
title('Truecolor image')
subplot(2,2,2)
imshow(img_rgb_red) 
title('Truecolor image - RED channel');
subplot(2,2,3)
imshow(img_rgb_green); 
title('Truecolor image - GREEN channel');
subplot(2,2,4)
imshow(img_rgb_blue); 
title('Truecolor image - BLUE channel');


%% Other image representation and conversion

clc
clear all
close all

% Truecolor images (a.k.a. RGB images)
img_rgb = imread('parrot.jpg');
figure
imshow(img_rgb)
title('Truecolor image')

% Grayscale representation and conversion
img_gray = rgb2gray(img_rgb); 
figure
imshow(img_gray)
title('Grayscale image')

% black and white representation and conversion
img_bw_05 = im2bw(img_gray, 0.5); %one channel with binary values
%img_bw_05 = im2double(img_gray)>0.5;
figure
imshow(img_bw_05)
title('Binary image (threshold 0.5)')

figure 
subplot(2,2,1)
imshow(img_gray)
title('Grayscale image')
subplot(2,2,2)
imshow(img_bw_05)
title('Binary image (threshold 0.5)')
img_bw_02 = im2bw(img_gray, 0.2); %low threshold
subplot(2,2,3)
imshow(img_bw_02)
title('Binary image (threshold 0.2)')
img_bw_08 = im2bw(img_gray, 0.8); %high threshold
subplot(2,2,4)
imshow(img_bw_08)
title('Binary image (threshold 0.8)')

%IMBINARIZE
figure 
subplot(2,2,1)
imshow(img_gray)
title('Grayscale image')
img_bw_auto_global = imbinarize(img_gray);
subplot(2,2,2)
imshow(img_bw_auto_global)
title('Binary image (Global method)')
img_bw_auto_local = imbinarize(img_gray,'adaptive');
subplot(2,2,3)
imshow(img_bw_auto_local)
title('Binary image (Local method)')
im_bw_dither = dither(img_gray);
subplot(2,2,4)
imshow(im_bw_dither)
title('Binary image (dithering)')

% indexed representation and conversion
[img_ind, map_ind] = rgb2ind(img_rgb, 16, 'nodither');
figure
imshow(img_ind, map_ind)
title('Truecolor --> Indexed conversion (16 colors)')
colorbar

% indexed --> truecolor
[img_ind_rgb] = ind2rgb(img_ind, map_ind);
figure
imshow(img_ind_rgb)
title('Indexed (16 colors) --> Truecolor conversion')

%% Image resizing 

clc;
clear all;
close all;

img_gray = imread('Lena_grayscale.bmp');
figure
imshow(img_gray)
title('Original image (Grayscale)')

img_gray_reduced = imresize(img_gray, 1/3);
figure
imshow(img_gray_reduced)
title('Reduced by 1/3')

img_gray_enlarged = imresize(img_gray_reduced, 3, 'nearest');
figure
imshow(img_gray_enlarged)
title('Reduced (1/3) and enlarged (3) using Nearest-Neighbor interp.')

img_gray_enlarged = imresize(img_gray_reduced, 3, 'bilinear');
figure
imshow(img_gray_enlarged)
title('Reduced (1/3) and enlarged (3) using Bilinear interpolation')

img_gray_enlarged = imresize(img_gray_reduced, 3, 'bicubic');
figure
imshow(img_gray_enlarged)
title('Reduced (1/3) and enlarged (3) using Bicubic interpolation')

img_gray_resized = imresize(img_gray, [200 500], 'nearest');
figure
imshow(img_gray_resized)
title('Resized to specific size using Nearest-Neighbor')


%% Image aritmetic
clc;
clear all;
close all;

img_gray_lena = imread('Lena_grayscale.bmp');
figure; imshow(img_gray_lena); title('Lena Grayscale uint8 [0,255]');

img_parrot = imread('parrot.jpg');
lena_size = size(img_gray_lena);
img_gray_parrot = imresize(rgb2gray(img_parrot), lena_size(1:2));
figure; imshow(img_gray_parrot);
title('Parrot uint8 [0,255] grayscale, resized as the previous image');

% addition
img_gray_lena_add_parrot = img_gray_lena + img_gray_parrot;
figure; imshow(img_gray_lena_add_parrot); title('Lena + Parrot');

% subtraction
img_gray_lena_sub_parrot = img_gray_lena - img_gray_parrot;
figure; imshow(img_gray_lena_sub_parrot); title('Lena - Parrot');

% multiplication
img_gray_lena_mult_parrot = img_gray_lena.*img_gray_parrot;
figure; imshow(img_gray_lena_mult_parrot); title('Lena x Parrot');

%% Crop an image
clc;
clear all;
close all;

img_rgb = imread('parrot.jpg');
figure
imshow(img_rgb)
title('Original image')

% Select the area by dragging the mouse, press the dx button and select
% 'Crop Image'
img_rgb_cropped = imcrop(img_rgb);
figure
imshow(img_rgb_cropped)
title('Cropped image')

img_rgb_cropped_spec = imcrop(img_rgb, [1 100 200 50]);
figure
imshow(img_rgb_cropped_spec)
title('Cropped image with region [1 100 200 10]')

%alternative method
img_rgb_cropped_spec_alt = img_rgb(100:150,1:201,:);
figure
imshow(img_rgb_cropped_spec_alt)
title('Cropped image with region [1 100 200 10]')

%% Image rotation
clc;
clear all;
close all;

img_gray = imread('Lena_grayscale.bmp');
figure
imshow(img_gray)
title('Original image (Grayscale)')


img_gray_rotated = imrotate(img_gray, 35, 'nearest', 'crop'); 
figure
imshow(img_gray_rotated)
title('35° rotation using Nearest-Neighbor interpolation (cropped)')

img_gray_rotated = imrotate(img_gray, 35, 'nearest');
figure
imshow(img_gray_rotated)
title('35° rotation using Nearest-Neighbor interpolation');

img_gray_rotated = imrotate(img_gray, 35, 'bilinear'); 
figure
imshow(img_gray_rotated)
title('35° rotation using Bilinear interpolation')

img_gray_rotated = imrotate(img_gray, 35, 'bicubic'); 
figure
imshow(img_gray_rotated)
title('35° rotation using Bicubic interpolation')

img_gray_transformed = imrotate(imresize(img_gray,0.5), -97, 'bicubic', 'crop'); 
figure
imshow(img_gray_transformed)
title('General transformation')

%% Ex 1 - circular profile picture
clc
close all
clear all
%150x150 round image

I = imread('Lena_grayscale.bmp');
figure
imshow(I)

tl = [75 120];
br = tl + 220;
I_crop = I(tl(2):br(2),tl(1):br(1));

target_size = 150;
I_crop = imresize(I_crop,[target_size,target_size]);

figure
imshow(I_crop)

for i=1:target_size
    for j=1:target_size
        if(((i-target_size/2)^2+(j-target_size/2)^2)>(target_size/2)^2)
            %white background
            I_crop(i,j) = 255;
        end
    end
end

figure
imshow(I_crop)

