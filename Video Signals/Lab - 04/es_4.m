%% Spatial filtering for edge extraction

clc;
clear all;
close all;

A = im2double(imread('Lena_grayscale.bmp')); 
figure
imshow(A); title('Original image');

figure
surf(A); colormap gray; shading interp;

%% First derivative
clc
close all
clear all

A = im2double(imread('Lena_grayscale.bmp')); 
figure
imshow(A); title('Original image');


%Roberts
% h_x = [0 0 -1;
%         0 1 0;
%         0 0 0];
% h_y = [-1 0 0;
%         0 1 0;
%         0 0 0];

%Prewit
% h_x = [1 0 -1;
%        1 0 -1;
%        1 0 -1];
% h_y = [1 1 1;
%        0 0 0;
%        -1 -1 -1];
%h_x = fspecial('prewit')';
%h_y = fspecial('prewit');

%Sobel
% h_x = [1 0 -1;
%        2 0 -2;
%        1 0 -1];
% h_y = [1 2 1;
%        0 0 0;
%       -1 -2 -1];
h_x = fspecial('sobel')';
h_y = fspecial('sobel');


A_h_x = conv2(A, h_x, 'same');
%alternative
%A_h_x = imfilter(A, h_x, 'symmetric', 'conv', 'same');

A_h_y = conv2(A, h_y, 'same');

A_h = sqrt((A_h_x).^2 + (A_h_y).^2);
A_h_thr = im2bw(A_h, 0.5);
A_h_dir = rad2deg(atan2(-A_h_y,A_h_x));

figure
imagesc(A_h_x); title('h_x Output');
colormap gray; daspect([1 1 1])

figure
imagesc(A_h_y); title('h_y Output');
colormap gray; daspect([1 1 1])

figure
imagesc(A_h); title('Edge Magnitude');
colormap gray; daspect([1 1 1])
figure
imshow(A_h_thr); title('Edge thr');

figure()
imagesc(A_h_dir)
hmap(1:256,1) = linspace(0,1,256); 
hmap(:,[2 3]) = 0.7; 
huemap = hsv2rgb(hmap); 
colormap(huemap)
colorbar; daspect([1 1 1])
title('Gradient direction');

[X,Y] = meshgrid(1:size(A,2),1:size(A,1));
figure
imshow(A)
hold on
quiver(X,Y,A_h_x,A_h_y)
title('gradient vector field')



%% 2nd derivative (Laplacian)

clear all
close all
clc

A = im2double(imread('Lena_grayscale.bmp')); 
figure
imshow(A); title('Original image');

% gaussian smoothing
% h_g = fspecial('gaussian',3,1);
% A = conv2(A, h_g, 'same');

% Laplacian operator
%4 neighbour
% h_l =(1/4)*[0 1 0;
%        1 -4 1;
%        0 1 0];
% h_l = fspecial('laplacian',0);
%8 neighbour not separable
h_l = (1/8)*[1 1 1;
       1 -8 1;
       1 1 1];
% h_l = 3/8*fspecial('laplacian',0.5)
A_h = conv2(A, h_l, 'same'); % eg. Laplacian emphasize noise

%combine gaussian+laplacian = log
%h_log = fspecial('log',3,1);
h_log = fspecial('log',9,3); 
A_h = conv2(A, h_log, 'same');


figure
imagesc(A_h); title('Laplacian');
colormap gray; daspect([1 1 1])

%zero crossing
mask = zeros(size(A_h));
for i=2:(size(A_h,1)-1)
    for j = 2:(size(A_h,2)-1)
       if (    A_h(i,j)*A_h(i-1,j) < 0 || A_h(i,j)*A_h(i+1,j) < 0 )
                 mask(i,j) = 1;
       end
    end
end
figure
imshow(mask); title('Laplacian zero crossing');

%mixing laplacian and sobel
h_x = fspecial('sobel')';
h_y = fspecial('sobel');

A_h_x = conv2(A, h_x, 'same');
A_h_y = conv2(A, h_y, 'same');
A_h = sqrt((A_h_x).^2 + (A_h_y).^2);
A_h_thr = im2bw(A_h, 0.75);

figure
imshow(A_h_thr)
title('threshold gradient magnitude')

mask = mask & A_h_thr;

figure
imshow(mask); title('Laplacian zero crossing + Sobel thr');


%% Canny edge detector
close all
clear all
clc

%Input image
img = imread ('Lena_grayscale.bmp');
%Show input image
figure, imshow(img);
title('Original image')

img = im2double (img);

%low pass filtering
%5x5 Gaussian Filter Coefficient
B = fspecial('gaussian',5,1);
%Convolution of image by Gaussian Coefficient
A=imfilter(img, B, 'symmetric', 'conv', 'same');

figure, imshow(A);
title('Smoothened image')

%Filter for horizontal and vertical direction
h_x = fspecial('sobel')';
h_y = fspecial('sobel');

%Convolution by image by horizontal and vertical filter
A_h_x = imfilter(A, h_x, 'symmetric', 'conv', 'same');
A_h_y = imfilter(A, h_y, 'symmetric', 'conv', 'same');

%Calculate directions/orientations
angles = atan2 (-A_h_y, A_h_x);
angles = angles*180/pi;

figure, imagesc(angles); colorbar; daspect([1 1 1])
title('Edge angles')

%Keeping just direction not orientations
angles(angles<0) = angles(angles<0)+180;

%Adjusting directions to nearest 0, 45, 90, or 135 degree
angles(angles<45/2 | angles >= (180+135)/2 ) = 0;
angles(angles>=45/2 & angles<(90+45)/2) = 45;
angles(angles>=(90+45)/2 & angles<(90+135)/2) = 90;
angles(angles>=(90+135)/2 & angles<(180+135)/2) = 135;

figure, imagesc(angles); colorbar; daspect([1 1 1])
title('4 bins angles')

%Calculate magnitude
magnitude = sqrt((A_h_x.^2) + (A_h_y.^2));

figure, imshow(magnitude);
title('Edge magnitude')

mask = zeros(size(img));
%Non-Maximum Supression
for i=2:size(img,1)-1
    for j=2:size(img,2)-1
        if (angles(i,j)==0)
            mask(i,j) = (magnitude(i,j) == max([magnitude(i,j), magnitude(i,j+1), magnitude(i,j-1)]));
        elseif (angles(i,j)==45)
            mask(i,j) = (magnitude(i,j) == max([magnitude(i,j), magnitude(i+1,j-1), magnitude(i-1,j+1)]));
        elseif (angles(i,j)==90)
            mask(i,j) = (magnitude(i,j) == max([magnitude(i,j), magnitude(i+1,j), magnitude(i-1,j)]));
        elseif (angles(i,j)==135)
            mask(i,j) = (magnitude(i,j) == max([magnitude(i,j), magnitude(i+1,j+1), magnitude(i-1,j-1)]));
        end
    end
end
magnitude = mask.*magnitude;
figure, imshow(mask);
title('local maxima mask')
figure, imshow(magnitude);
title('non-maximum suppression output')

%Value for Thresholding
T_High = 0.1250;
T_Low = 0.4*T_High;
%Hysteresis Thresholding
T_Low = T_Low * max(magnitude(:));
T_High = T_High * max(magnitude(:));

res = zeros (size(size(img,1)));
for i = 1  : size(img,1)
    for j = 1 : size(img,2)
        if (magnitude(i, j) < T_Low)
            res(i, j) = 0;
        elseif (magnitude(i, j) > T_High)
            res(i, j) = 1;
        %Using 8-connected components
        elseif( magnitude(i+1,j)>T_High || magnitude(i-1,j)>T_High || ...
                magnitude(i,j+1)>T_High || magnitude(i,j-1)>T_High || ...
                magnitude(i-1, j-1)>T_High || magnitude(i-1, j+1)>T_High ||...
                magnitude(i+1, j+1)>T_High || magnitude(i+1, j-1)>T_High )
            res(i,j) = 1;
        end
    end
end

%Show final edge detection result
figure, imshow(res);
title('edge detected by Canny edge detector')

%matlab function
res = edge(img,'canny');
figure, imshow(res);
title('edge detected by Canny edge detector (matlab)')

