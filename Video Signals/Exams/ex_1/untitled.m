% background segmentation
clc; clear all; close all

%a
img = imread('test_img.png');
img_gray = rgb2gray(img);
figure;
imshow(img_gray)

%b
ent = zeros(size(img_gray));

%c
grid_size = 8;
for i = 1:grid_size:size(img_gray,1)
    for j =1:grid_size:size(img_gray,2)
        %c1
        I = img_gray(i:(i+grid_size-1), j:(j+grid_size-1));
        p = imhist(I);
        p = p/numel(I);%by total number of elements
        %c2
        entropy = -sum(p.*log2(p+1e-6));
        %c3
        ent(i:(i+grid_size-1), j:(j+grid_size-1)) = entropy;
    end
end

%d
ent = (ent - min(ent(:)))/(max(ent(:))-min(ent(:)));

figure
imagesc(ent)

mask = im2bw(ent, 0.5);% mask = ent>0.5;
figure
imagesc(mask)

%e
se = strel('disk', 20);
% first close(fill), then open(remove)!!!!!!!!!!!!
mask = imclose(mask,se);
figure
imagesc(mask)
mask = imopen(mask,se);
figure
imagesc(mask)

%f
res = img_gray.*uint8(mask);
figure
imshow(res)