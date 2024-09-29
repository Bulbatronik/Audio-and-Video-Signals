clc;clear all; close all;
%a
I = imread("video_frame.png");
figure
imshow(I)

%b
I_bw = I<100;% rark(fish) ->1
figure;
imshow(I_bw)

%c
L = bwlabel(I_bw);
figure
imagesc(L)

%d
min_area = 100;
count = 1;
for k = 1:max(L(:))
    mask = (L==k);
    figure
    imshow(mask)
    area = sum(mask(:));
    if area > min_area
        count = count+1;
        B = strel('arbitrary', ones(3));
        I_per = mask - imerode(mask, B);%perimiter
        figure
        imshow(I_per)

        per = sum(I_per(:));

        mass = 0.02*area-0.05*per+8;
        mass
    end
end
