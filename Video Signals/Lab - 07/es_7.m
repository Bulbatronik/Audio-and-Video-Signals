%% COLOR SPACE SEGMENTATION

clc
close all
clear all

I = imread('flamingos.jpg');

I = im2double(imresize(I,[480,NaN])); %reduce image size for computational reasons

figure
imshow(I)

I_r = I(:,:,1);
I_g = I(:,:,2);
I_b = I(:,:,3);

I_hsv = rgb2hsv(I);
I_h = I_hsv(:,:,1);
I_s = I_hsv(:,:,2);
I_v = I_hsv(:,:,3);

figure
scatter3(I_r(:),I_g(:),I_b(:),1,[I_r(:) I_g(:) I_b(:)])
xlabel('r')
ylabel('g')
zlabel('b')
title('RGB color space')

figure
scatter3(I_h(:),I_s(:),I_v(:),1,[I_r(:) I_g(:) I_b(:)])
xlabel('h')
ylabel('s')
zlabel('v')
title('HSV color space')

h_t = 0.01;
thr = 0.05;

%segmenting only in hue
min_h = (h_t - thr);
max_h = (h_t + thr);
%mask =  (I_h > min_h) & (I_h < max_h); %not correct! hue is a wheel

if(min_h<0)
 mask =  (I_h < max_h) | (I_h > 1+min_h);
elseif (max_h>1)
 mask =  (I_h > min_h) | (I_h < max_h-1);
else
 mask =  (I_h > min_h) & (I_h < max_h); 
end   
    
figure
imshow(mask)
figure
imshow(mask.*I)


%% Harris keypoint detection
clc;
clear all;
close all;

im = imread('Harris_giraffe.png');
im = im2double(rgb2gray(im));

figure()
imshow(im), title('Original image');

% Prewitt operator for gradient estimation
dx = fspecial('prewit')'; 
dy = fspecial('prewit');  
    
% derivatives
% Ix = imfilter(im,dx,'conv','same','symmetric');
% Iy = imfilter(im,dy,'conv','same','symmetric');
Ix = conv2(im,dx,'same');
Iy = conv2(im,dy,'same');

% Gaussian filter of size 6*sigma (+/- 3sigma) and minimum size 1
sigma = 2;
g = fspecial('gaussian', max(1, floor(6*sigma)), sigma);    

% Smoothed squared image derivatives
Ix2 = conv2(Ix.^2,g,'same');
Iy2 = conv2(Iy.^2,g,'same');
Ixy = conv2(Ix.*Iy,g,'same');

% Harris measure.
k = 0.04;
% M = [Ix2 Ixy;Ixy Iy2]
% cim = determinant - k * trace^2
cim = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2; 

% Noble  measure.
% cim = (Ix2.*Iy2 - Ixy.^2)./(Ix2 + Iy2 + eps);

figure
imagesc(cim), title('Corner strenght');
colorbar
daspect([1 1 1])

% show thresholded strenght (doesn't work well)
thresh = 0.0001; %Harris
%thresh = 0.001; %Noble
mask_1 = cim > thresh;
figure
imshow(mask_1);
title('Thresholded image, w/out nonmaximal suppression');

% Nonmaximal suppression and threshold
	
% Extract local maxima by performing a grey scale morphological
% dilation and then finding points in the corner strength image that
% match the dilated image and are also greater than the threshold.

radius = 10;
B = strel('square',radius);
mx = imdilate(cim, B);

mask_2 = (cim == mx);
figure
imshow(mask_2)
title('local maxima')

cim = mask_1 & mask_2;      % Find maxima.
figure
imshow(cim)
title('strong local maxima')

% Find row,col coords.
[r,c] = find(cim);

figure, imshow((im));
hold on;  plot(c, r, 'r+'), axis equal, title('Corners');


%% Hough transform - isolated points
clc;
clear all;
close all;

im = zeros(600, 600);
im(30, 30) = 1;
im(30, 570) = 1;
im(570, 30) = 1;
im(570, 570) = 1;
im(300, 300) = 1;

imshow(im), title('Original image');

% Hough transform
[H, T, R] = hough(im);

figure
imagesc(T, R, H);
title('Hough Transform of isolated points');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
colormap('hot')
colorbar


%% Hough transform - straight lines
clc;
clear all;
close all;

im = zeros(600, 600);
im(100:480, 200) = 1;
im(100:480, 400) = 1;
im(100, 200:400) = 1;
im(480, 200:400) = 1;

figure, imshow(im), title('Rectangle');

% Hough transform
[H, T, R] = hough(im);

figure
imagesc(T, R, H);
title('Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');


figure
imagesc(T, R, log(H));
title('Log Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');

%% Hough transform - noise
clc;
clear all;
close all;

im = zeros(600, 600);
im(100:480, 200) = 1;
im(100:480, 400) = 1;
im(100, 200:400) = 1;
im(480, 200:400) = 1;

for i = 1:10000
    im( ceil(rand(1)*600), ceil(rand(1)*600) ) = 1;
end
imshow(im), title('Noise image');

% Hough transform
[H, T, R] = hough(im);

figure
imagesc(T, R, H);
title('Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');

figure
imagesc(T, R, log(H));
title('Log Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');
%% Hough transform - photo
clc;
clear all;
close all;

im = imread('SEM.jpg');
figure
imshow(im), title('Original image');

edges = edge(im, 'prewitt');
figure
imshow(edges), title('Edge image');

% Hough transform
[H, T, R] = hough(edges);

figure
imagesc(T, R, H);
title('Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');


% thresh = ceil(0.25*max(H(:)));
% radius = 30;
% B=strel('square',radius);
% max_H_loc = imdilate(H,B);
% mask = (H==max_H_loc)&(H>thresh);
% [i_max,j_max]=find(mask);
% R_max = R(i_max);
% T_max = T(j_max);
%or 
P = houghpeaks(H,10,'threshold',ceil(0.25*max(H(:))));
R_max = R(P(:,1));
T_max = T(P(:,2));

hold on
plot(T_max,R_max,'s','color','white')

lines = houghlines(edges,T,R,P,'FillGap',1000,'MinLength',20);
figure, imshow(im), hold on

for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
end
title('Detected lines')


