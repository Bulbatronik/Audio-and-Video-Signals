clc;clear all; close all;
%a
I_RGB = imread('test_img.png');
figure
imshow(I_RGB)
%b
I_RGB = imresize(I_RGB,[256 256]);
figure
imshow(I_RGB)

%c
I_HSV = rgb2hsv(I_RGB);

%d
V = I_HSV(:,:,3);
min_V=min(V(:));
max_V = max(V(:));
V_new = (V-min_V)/(max_V-min_V);

%e
I_HSV_2 = I_HSV;
I_HSV_2(:,:,3) = V_new;
I_RGB_2 = hsv2rgb(I_HSV_2);
figure
imshow(I_RGB_2)

%f
R = histeq(I_RGB_2(:,:,1));
G = histeq(I_RGB_2(:,:,2));
B = histeq(I_RGB_2(:,:,3));
I_RGB_3(:,:,1) = R;
I_RGB_3(:,:,2) = G;
I_RGB_3(:,:,3) = B;
figure
imshow(I_RGB_3)

%g
h = fspecial('gaussian', 3, 0.5);
R = conv2(R, h, 'same');
G = conv2(G, h, 'same');
B = conv2(B, h, 'same');
I_RGB_4(:,:,1) = R;
I_RGB_4(:,:,2) = G;
I_RGB_4(:,:,3) = B;
figure
imshow(I_RGB_4)