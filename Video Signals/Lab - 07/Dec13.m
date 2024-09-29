%% color space segmentation
clc; clear all; close all;

I = imread('flamingos.jpg');

I = im2double(imresize(I, [240 NaN]));% resize the image for computational complexity reduction
% im2double: [0. 255] -> [0, 1]

figure
imshow(I)
title('original image')
% segment by color using a mask

%% RGB REPRES
I_r = I(:,:,1);
I_g = I(:,:,2);
I_b = I(:,:,3);

figure
scatter3(I_r(:), I_g(:),I_b(:), 1, [I_r(:) I_g(:) I_b(:)]);
xlabel('r'); ylabel('y'); zlabel('b');
title('RGB colorspace')
%% HSV REPRES
I_hsv = rgb2hsv(I);

I_h = I_hsv(:,:,1);
I_s = I_hsv(:,:,2);
I_v = I_hsv(:,:,3);

figure
scatter3(I_h(:), I_s(:),I_v(:), 1, [I_r(:) I_g(:) I_b(:)]);
xlabel('h'); ylabel('s'); zlabel('v');
title('HSV colorspace')
%%
h_t = 0.01;
thr = 0.05;

min_h = h_t - thr;
max_h = h_t + thr;

%mask = I_h>min_h & I_h<max_h;% bad mask

if(min_h<0)
    mask = I_h<max_h | I_h>(1+min_h);
elseif(max_h>1)
    mask = I_h>min_h | I_h<(max_h-1);
else
    mask = I_h>min_h & I_h<max_h;
end

figure
imshow(mask)

figure
imshow(mask.*I)

%% Harris keypoint detection
clc; clear all; close all;

im = im2double(rgb2gray((imread('Harris_giraffe.png'))));

figure
imshow(im)
title('original giraffe')

% estimate the edges
dx = fspecial('prewit')';
dy = fspecial('prewit');

Ix = conv2(im, dx, 'same');
Iy = conv2(im, dy, 'same');

% figure
% imagesc(Ix)
% figure
% imagesc(Iy)

%LP filter
sigma = 2;
g = fspecial('gaussian', floor(6*sigma), sigma);

Ix2 = conv2(Ix.^2, g, 'same');
Iy2 = conv2(Iy.^2, g, 'same');
Ixy = conv2(Ix.*Iy, g ,'same');

% M = [Ix2 Ixy; Ixy, Iy2]
k = 0.04;
cim = (Ix2.*Iy2-Ixy.^2) - k*(Ix2+Iy2).^2;
%cim = (Ix2.*Iy2-Ixy.^2)./(Ix2+Iy2+eps).^2;% CHECK

figure
imagesc(cim)
colorbar

thresh = 0.0001;%0.001
mask_1 = cim>thresh;

figure
imshow(mask_1)

% non-maximum surpression
radius = 10;% morphologiacal oper
B = strel('square', radius);
mx_loc = imdilate(cim, B);

figure
imagesc(mx_loc)

mask_2 = (cim == mx_loc);
figure
imshow(mask_2)

mask = mask_1 & mask_2;
figure
imshow(mask)

[r, c] = find(mask);

figure
imshow(im)
hold on
plot(c, r, 'r+')
axis equal
%% Hough transform
clc; clear; close all;

im = zeros(600, 600);
% im(30, 30) = 100;
% im(30, 570) = 100;
% im(570, 30) = 100;
% im(570, 570) = 100;
% im(300, 300) = 100;
im(100:480, 200) = 1;
im(100:480, 400) = 1;
im(100, 200:400) = 1;
im(480, 200:400) = 1;

for i = 1:10000
    im( ceil(rand(1)*600), ceil(rand(1)*600) ) = 1;
end


figure
imshow(im)
title('original image')

% rho = x*cos(theta) +y*sin(theta)
[H, T, R] = hough(im);

figure
imagesc(T, R, log(H))
xlabel('\theta');ylabel('\rho');axis on; axis normal; 
colormap hot
title('hough transform')
% intersection represent a line
%% 
clc; clear; close all;

im = imread('SEM.jpg');

figure
imshow(im)


[H, T, R] = hough(im);

figure
imagesc(T, R, log(H))
xlabel('\theta');ylabel('\rho');axis on; axis normal; 
colormap hot
title('hough transform')
%% apply over a mask of edges
clc; clear; close all;

im = imread('SEM.jpg');

figure
imshow(im)

edge_mask = edge(im, 'sobel');
figure
imshow(edge_mask)

[H, T, R] = hough(edge_mask);

figure
imagesc(T, R, log(H))
xlabel('\theta');ylabel('\rho');axis on; axis normal; 
colormap hot
title('hough transform')

% % search for the lines
thr = ceil(0.25*max(H(:)));
radius = 30;
B = strel('square', radius);
% max_H_loc = imdilate(H,B);
% mask = (H == max_H_loc)& (H>thr);
% 
% % figure
% % imshow(mask)
% 
% [i_max, j_max] = find(mask);
% R_max = R(i_max);
% T_max = T(j_max);

% MATLAB
P = houghpeaks(H, 10, 'threshold', thr);
R_max = R(P(:,1));
T_max = T(P(:,2));

hold on
plot(T_max, R_max, 's', 'color',' white')
lines = houghlines(edge_mask, T, R, P, 'FillGap', 1000, 'MinLength',20);
figure
imshow(im)
hold on

for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1),xy(:,2), 'Linewidth', 2, 'Color', 'green')
end