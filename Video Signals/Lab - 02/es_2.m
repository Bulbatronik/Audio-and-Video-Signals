%% Image histogram
clc
clear all
close all

img_gray = imread('zebra.jpg');
figure
imshow(img_gray)
title('Grayscale uint8 [0,255] image')



%% histogram by hand vs matlab functions

%first method
hist_handmade = zeros(1,256);
for i = 0:255  
     for ii = 1:size(img_gray,1)
         for jj = 1:size(img_gray,2)
             if (img_gray(ii,jj) == i)
                 hist_handmade(i+1) = hist_handmade(i+1) + 1;
             end
         end
     end
end

figure
bar(0:255,hist_handmade)
xlim([0 255])
title('handmade histogram 256 bins')

%smarter way
hist_handmade = zeros(1,256);
for i = 0:255  
       hist_handmade(i+1) = sum(sum(img_gray == i));
end

figure
bar(0:255,hist_handmade)
xlim([0 255])
title('handmade histogram 256 bins')

%alternative method using hist
figure
hist(double(img_gray(:)),255);
title('histogram 256 bins using hist')

%best method using imhist
figure
imhist(img_gray);
title('Image histogram using imhist');


%different number of bins (smoother curve)
figure; imhist(img_gray,128);
title('Image histogram with 128 bin');

%% histogram of bw image
img_gray_bw = im2bw(img_gray,0.5);
figure
imshow(img_gray_bw)
title('BW Image')
figure
imhist(img_gray_bw)
title('BW Image histogram')

%% histogram of RGB image
img_rgb = imread('parrot.jpg');
figure; imshow(img_rgb);
title('Parrot image');
hist_R = imhist(img_rgb(:,:,1));
hist_G = imhist(img_rgb(:,:,2));
hist_B = imhist(img_rgb(:,:,3));
figure;
plot(0:255,hist_R,'r')
hold on
plot(0:255,hist_G,'g')
plot(0:255,hist_B,'b')
xlim([0 255])
title('rgb channels histogram');

%% Histogram and interpolation

clc;
clear all;
close all;

img_gray = imread('zebra.jpg');
figure
imshow(img_gray)
title('Original image (Grayscale)')

img_gray_reduced = imresize(img_gray, 1/3);
figure
imshow(img_gray_reduced)
title('Reduced by 1/3')

figure
imhist(img_gray_reduced);
title('Histogram of reduced image');

img_gray_enlarged = imresize(img_gray_reduced, 3, 'nearest');
figure
imshow(img_gray_enlarged)
title('Reduced (1/3) and enlarged (3) using Nearest-Neighbor interp.')

%the histogram is the same (except for a factor of 9)
figure
imhist(img_gray_enlarged);
title('Histogram of Nearest-Neighbor interp. image');

img_gray_enlarged = imresize(img_gray_reduced, 3, 'bilinear');
figure
imshow(img_gray_enlarged)
title('Reduced (1/3) and enlarged (3) using Bilinear interpolation')

%smoother but not exceed the range of values
figure
imhist(img_gray_enlarged);
title('Histogram of Bilinear interpolated image');

img_gray_enlarged = imresize(img_gray_reduced, 3, 'bicubic');
figure
imshow(img_gray_enlarged)
title('Reduced (1/3) and enlarged (3) using Bicubic interpolation')

%even smoother and exceed the range of values
figure
imhist(img_gray_enlarged);
title('Histogram of Bicubic interpolated image');

%% Histogram shifting
clc
clear all
close all

img_gray_compr = imread('zebra_compressed.jpg');
figure
imshow(img_gray_compr)
title('Grayscale uint8 [0,255] image w/ compressed histogram')
figure
imhist(img_gray_compr)
title('Compressed histogram')

% brighten the image (increase the luminance by 60)
img_gray_compr_brighter = img_gray_compr + 60;
figure
imshow(img_gray_compr_brighter)
title('Immagine with histogram translated by 60')
figure
imhist(img_gray_compr_brighter)
title('Histogram translated by 60')

% NB: doing the same with the non-compressed histogram...
img_gray = imread('zebra.jpg');
figure
imshow(img_gray)
title('Grayscale uint8 [0,255] image');
figure
imhist(img_gray)
title('Image histogram');

% translate the histogram...
img_gray_brighter = img_gray + 60;
figure
imshow(img_gray_brighter)
figure
imhist(img_gray_brighter)


%% Compression/dilation of an image histogram
clc
clear all
close all

img_gray = imread('zebra_compressed.jpg');
figure
imshow(img_gray)
title('Grayscale uint8 [0,255] image w/ compressed histogram') 
figure
imhist(img_gray)
title('Compressed histogram')

img_gray_double = double(img_gray);
min_value = min(img_gray_double(:));
max_value = max(img_gray_double(:));
img_gray_mod = uint8(255*(img_gray_double - min_value) / (max_value - min_value));

figure
imshow(img_gray_mod)
title('Image w/ histogram dilated to the whole range')
figure
imhist(img_gray_mod)
title('Histogram linearly dilated to the whole range')

h = imhist(img_gray_mod);
h_norm = h / numel(img_gray_mod); 
h_norm_cum = cumsum(h_norm);
figure
plot(h_norm_cum)
title('Cumulative distribution function')
%this is not histogram equalization!

%% Histogram equalization
clc
clear all
close all

img_gray_compress = imread('zebra_compressed.jpg');
figure; imshow(img_gray_compress);
title('Grayscale uint8 [0,255] image w/ compressed histogram');
h = imhist(img_gray_compress);

figure
imhist(img_gray_compress)
title('Compressed histogram')

% Step-by-step 256 bins equalization

h_norm = h/numel(img_gray_compress);
h_norm_cum = cumsum(h_norm);

figure
plot(h_norm_cum)
title('Cumulative distribution function')

% warning: gray level are in [0, 255] but Matlab indices in [1, 256]
img_gray_double_equal = h_norm_cum(img_gray_compress + 1);
img_gray_equal = uint8(round(255*(img_gray_double_equal))); 
h_equal = imhist(img_gray_equal);
figure
imshow(img_gray_equal)
title('Image w/ equalized histogram')

figure
imhist(img_gray_equal)
title('Equalized histogram')

h_norm_equal = h_equal / numel(img_gray_compress);
h_norm_cum_equal = cumsum(h_norm_equal);
figure
plot(h_norm_cum_equal)
title('Equilized cumulative distribution function')


% Equalization using the Matlab function histeq()
img_gray_equal_matlab_256 = histeq(img_gray_compress,256);
h_equal = imhist(img_gray_equal_matlab_256,256);
h_norm_equal = h_equal / numel(img_gray_compress);
h_norm_cum_equal = cumsum(h_norm_equal);
figure
plot(h_norm_cum_equal)
title('Equilized cumulative distribution function')
figure
imshow(img_gray_equal_matlab_256)
title('Equalized image using histeq()')
figure
imhist(img_gray_equal_matlab_256)
title('Equalized histogram using histeq()')


%% Equalization using the Matlab function histeq()
% reducing the bins from 256 to 8
% the histogram now is more uniform, even if we lose color resolution
img_gray_equal_matlab_8 = histeq(img_gray_compress, 8); 

figure
imshow(img_gray_equal_matlab_8)
title('Equalized image using histeq(), 8 bins')

figure
imhist(img_gray_equal_matlab_8)
title('Equalized histogram using histeq(), 8 bins')

h_equal = imhist(img_gray_equal_matlab_8,8);
h_norm_equal = h_equal / numel(img_gray_compress);
h_norm_cum_equal = cumsum(h_norm_equal);

figure()
plot(h_norm_cum_equal)
title('Equalized cumulative distribution using histeq(), 8 bins')

%% Histogram equalization to fix underexposed photos 
clc;
clear all;
close all;

img_gray = imread('castle.jpg');
figure
imshow(img_gray)
title('original image')
figure
imhist(img_gray)
title('original image histogram')
img_gray_equal_matlab_256 = histeq(img_gray,256);
figure
imshow(img_gray_equal_matlab_256)
title('Equalized image using histeq()');

%% Local Histogram equalization 
clc;
clear all;
close all;

img_gray = imread('castle.jpg');
figure
imshow(img_gray)
title('original image')
figure
imhist(img_gray)
title('original image histogram')

img_gray_mask = (1 == imread('castle_binary.png'));

figure
imshow(img_gray_mask)
title('mask')

h = imhist(img_gray(img_gray_mask));

figure
bar(h)
title('histogram of masked image')

h_norm = h/sum(h);
h_norm_cum = cumsum(h_norm);

figure()
plot(h_norm_cum)
title('Cumulative distribution function')

% warning: gray level are in [0, 255] but Matlab indices in [1, 256]
img_gray_double_equal = h_norm_cum(img_gray + 1);
img_gray_equal = uint8(round(255*(img_gray_double_equal))); 

figure
imshow(img_gray_equal)
title('Inermediate result')

img_gray_composed = img_gray_equal.*uint8(img_gray_mask) + img_gray.*uint8((1-img_gray_mask));
figure
imshow(img_gray_composed)
title('Final result')

%% histogram matching

clc
clear all
close all

img_gray = rgb2gray(imread('castle.jpg')); 
h = imhist(img_gray);
figure
imshow(img_gray)
title('input image')
figure
bar(0:255,h)
title('histogram of image');
h_norm = h/sum(h);
F = cumsum(h_norm);
figure
plot(F)
title('cdf of image');
G = F;
G(1:150) = linspace(0,G(150),150);
%G(1:256) = linspace(0,G(256),256); %corresponds to histogram equalization
hold on
plot(G,'--r')
legend('Original cdf', 'Desired cdf')

img_out = uint8(zeros(size(img_gray)));

%figure
for i = 0:255
    j = find(G>=F(i+1),1);
    img_out(img_gray == i) = j-1;
    %imshow(img_out)
    %pause(0.01)
end

figure
imshow(img_out) 
title('Output image')
figure
imhist(img_out) 
title('Output image hist')
h = imhist(img_out);
h_norm = h/sum(h);
F_2 = cumsum(h_norm);
figure
plot(F_2)
title('cdf of output image')


