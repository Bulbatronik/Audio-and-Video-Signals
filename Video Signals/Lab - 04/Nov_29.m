%% spatial filtering for edge extraction and edge detection
clc; clear all; close;
% flip the kernel by 180 deg for convolution

% for edges look at the gradient -> peak
% 2nd derivative - locate zero

A = im2double(imread('Lena_grayscale.bmp'));

figure
imshow(A)
title('original image')

figure
surf(A); colormap gray; shading interp
%% fiirst derivative
% Roberts
h_x = [0 0 -1; 
       0 1 0; 
       0 0 0];

h_y = [-1 0 0; 
       0 1 0; 
       0 0 0];

A_h_x = conv2(A, h_x, 'same');
%A_h_x = imfilter(A, h_x, 'conv', 'same'); % equivalently

A_h_y = conv2(A, h_y, 'same');
%A_h_y = imfilter(A, h_y, 'conv', 'same'); % equivalently

figure
imagesc(A_h_x)
title('h_x output')
colormap gray; axis equal

figure
imagesc(A_h_y)
title('h_y output')
colormap gray; axis equal
%Problems: 1)not relly related to x and y;2) asymmetric->shift;3)we just
%use 2 pixels, not the whole information
%% Prewit
close all
h_x = [1 0 -1; %vertical
       1 0 -1; 
       1 0 -1];

h_y = [1 1 1; %hoeizontal
       0 0 0; 
       -1 -1 -1];

A_h_x = conv2(A, h_x, 'same');
%A_h_x = imfilter(A, h_x, 'conv', 'same'); % equivalently

A_h_y = conv2(A, h_y, 'same');
%A_h_y = imfilter(A, h_y, 'conv', 'same'); % equivalently

figure
imagesc(A_h_x)
title('h_x output')
colormap gray; axis equal

figure
imagesc(A_h_y)
title('h_y output')
colormap gray; axis equal
% grad on x direct -> vertical edges 
% We don't use just 2 values (6). Like a LP filter by computing the average
% on the left and right. Plus, we are centered

% same
h_x = fspecial('prewit')';% transpose
h_y = fspecial('prewit'); % DEFAULT IS HORIZONTAL!
%% Sobel
close all
h_x = [1 0 -1; %vertical
       2 0 -2; 
       1 0 -1];

h_y = [1 2 1; %horizontal
       0 0 0; 
       -1 -2 -1];

A_h_x = conv2(A, h_x, 'same');
%A_h_x = imfilter(A, h_x, 'conv', 'same'); % equivalently

A_h_y = conv2(A, h_y, 'same');
%A_h_y = imfilter(A, h_y, 'conv', 'same'); % equivalently

figure
imagesc(A_h_x)
title('h_x output')
colormap gray; axis equal

figure
imagesc(A_h_y)
title('h_y output')
colormap gray; axis equal

% combine 2 gradients -> magnitude
A_h = sqrt(A_h_x.^2+A_h_y.^2);
figure
imagesc(A_h)
title('Gradient magnitude')
colormap gray; axis equal

% direction
A_h_dir = atan2(-A_h_y,A_h_x);

figure
imagesc(A_h_dir)
hmap(1:256,1) = linspace(0,1,256);
hmap(:,[2 3]) = 0.7;
huemap = hsv2rgb(hmap);
colormap(huemap)
colorbar ; axis equal
title('gradient direction')

[X, Y] = meshgrid(1:size(A,2), 1:size(A,1));
figure
imshow(A)
%quiver(X,Y,)
%% second derivative
%4 neighbour Laplacian
clc; clear, close all;

A = im2double(imread('Lena_grayscale.bmp'));

figure
imshow(A)
title('original image')

h_l = (1/4)*[0 1 0;
            1 -4 1;
            0 1 0];

A_h = conv2(A, h_l, 'same');

figure
imagesc(A_h)
title('laplacian')
colormap gray; axis equal
% noisy (image is noisy itself)
%% 8 neighbour Laplacian
clc; clear, close all;

A = im2double(imread('Lena_grayscale.bmp'));

figure
imshow(A)
title('original image')

% h_g = fspecial('gaussian', 3,1);
% A = conv2(A,h_g,'same');
% 
% figure
% imshow(A)
% title('Smothened version of the input image')

h_l = (1/8)*[1 1 1;
            1 -8 1;
            1 1 1];

A_h = conv2(A, h_l, 'same');

figure
imagesc(A_h)
title('laplacian')
colormap gray; axis equal
% before appying the laplacian LP the image to remove some noise
%%
h_log = fspecial('log',3,1);% both LP and the laplacian at once (lin oper)
A_h = conv2(A, h_log, 'same');

figure
imagesc(A_h)
title('log output')
colormap gray; axis equal

% zero crossing
mask = zeros(size(A));

for i=2:size(A_h,1)
    for j=2:size(A_h,2)
        if(A_h(i,j)*A_h(i-1,j)<=0 || A_h(i,j)*A_h(i,j-1)<=0)
            mask(i,j) = 1;%edge
        end
    end
end

figure
imshow(mask)
title('Laplacian zero crossing')


h_x = fspecial('sobel')';% transpose
h_y = fspecial('sobel');

A_h_x = conv2(A, h_x, 'same');
A_h_y = conv2(A, h_y, 'same');

A_h = sqrt(A_h_x.^2+A_h_y.^2);
figure
imagesc(A_h)
title('Gradient magnitude')
colormap gray; axis equal

%% edge thinning
mask_3 = mask & A_h_thr;

figure
imagesc(mask_3);
title('sobel + laplacian')
%% Canny edge detector (1st deriv method)
close all; clear; clc;

A = im2double(imread('Lena_grayscale.bmp'));

figure
imshow(A)
title('original image')

%low pass filtering
h_g = fspecial('gaussian', 7,3);
A = conv2(A,h_g,'same');

figure
imshow(A)
title('smothened image')

h_x = fspecial('sobel')';
h_y = fspecial('sobel');

A_h_x = conv2(A,h_x,'same');
A_h_y = conv2(A,h_y,'same');

magnitude = sqrt(A_h_x.^2+A_h_y.^2);

figure
imagesc(magnitude)
title('Gradient magnitude')
colormap gray; axis equal

angles = atan2(-A_h_y,A_h_x);

figure
imagesc(angles)
hmap(1:256,1) = linspace(0,1,256);
hmap(:,[2 3]) = 0.7;
huemap = hsv2rgb(hmap);
colormap(huemap)
colorbar ; axis equal
title('gradient direction')

% non maximum surpression
% just 4 directioms
angles(angles<0) = angles(angles<0)+180;% convert angles

angles(angles<45/2 | angles>(180+135)/2) = 0; % angle is close to 0 or 180 -> 0 (1st bin)
angles(angles>=45/2 & angles<(90+45)/2) = 45; % angle is close to 45 or 135 -> 45 (2nd bin)
angles(angles>=(90+45)/2 & angles<(90+135)/2) = 90; %3rd bin
angles(angles>=(90+135)/2 & angles<(180+135)/2) = 135; % 4th bin

figure
imagesc(angles)
hmap(1:256,1) = linspace(0,1,256);
hmap(:,[2 3]) = 0.7;
huemap = hsv2rgb(hmap);
colormap(huemap)
colorbar ; axis equal
title('4 bins gradient direction')

mask = zeros(size(A));

for i = 2:size(A,1)-1
    for j = 2:size(A,2)-1
        if(angles(i,j) == 0)
            mask(i,j) = (magnitude(i,j)==max([magnitude(i,j), magnitude(i,j-1),magnitude(i,j+1)]));
        
        elseif(angles(i,j) == 45)
            mask(i,j) = (magnitude(i,j)==max([magnitude(i,j), magnitude(i+1,j-1),magnitude(i-1,j+1)]));
        elseif(angles(i,j) == 90)
            mask(i,j) = (magnitude(i,j)==max([magnitude(i,j), magnitude(i+1,j),magnitude(i-1,j)]));
        elseif(angles(i,j) == 135)
            mask(i,j) = (magnitude(i,j)==max([magnitude(i,j), magnitude(i-1,j-1),magnitude(i+1,j+1)]));
        end
    end
end

figure
imshow(mask)
title('local maxima in gradient direction')

% hard thresholding
%t = 0.125;
t = 0.5;
mask2 = im2bw(magnitude,t);

figure
imagesc(mask2)
title('thresholded magnitude')
colormap gray; axis equal
%hysterisis thresholding
%t_high = 
%t_low = 

mask3 = mask & mask2;
figure
imagesc(mask3)
colormap gray; axis equal
title('local maxima + thr')
%% in matlab
res = edge(A, 'canny');
figure
imshow(res)
title('edge detection with canny (matlab)')