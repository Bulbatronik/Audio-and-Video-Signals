clc
close all
clear all
%% a)
I=imread('video_frame.png');
figure
imshow(I)
%% b)
I_bw = I<100;
figure
imshow(I_bw)
%% c)
L = bwlabel(I_bw);
%% d)
min_area = 100;
count = 0;
for k=1:max(L(:))
    %d.1)
    mask = (L==k);
    
    %d.2)
    area = sum(mask(:));
    if(area>min_area)
    
        %d.2.i)
        count = count+1;
        %d.2.ii)
        B = strel('arbitrary', [1 1 1; 1 1 1; 1 1 1]); 
        I_per = mask - imerode(mask, B);
        %d.2.iii)
        per = sum(I_per(:));
        
        %d.2.iv)
        mass = 0.02*area - 0.05*per + 8;
    end
end