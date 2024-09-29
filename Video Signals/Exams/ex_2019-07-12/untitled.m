close all
clear all
%% a
I1 = imread('1.jpg');
I1 = rgb2gray(I1); I1 = im2double(I1);
I2 = imread('2.jpg');
I2 = rgb2gray(I2); I2 = im2double(I2);
bg = imread('bg.jpg');
bg = rgb2gray(bg); bg = im2double(bg);
%% b
D1 = I1 - bg;
D2 = I2 - bg;

%% c I. and II.
B1 = abs(D1)>0.1*max(abs(D1(:))); 
B2 = abs(D2)>0.1*max(abs(D2(:))); 
%c) III.
n = round( 0.005*min(size(I1)) );
SE = strel('arbitrary',ones(n));
%c) IV.
M1 = imopen(B1,SE);
M2 = imopen(B2,SE);
%% d I. and II.
[i,j] = find(M1);
p_1(2) = sum(i)/numel(i);
p_1(1) = sum(j)/numel(j);
[i,j] = find(M2);
p_2(2) = sum(i)/numel(i);
p_2(1) = sum(j)/numel(j);
%d) III.
p2m = 0.05;
t = 2;%s
v = norm(p_2-p_1)*p2m/t %m/s
v*60*60/1000