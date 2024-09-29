clc;clear all;close all;
m=80;n=80;

a = checkerboard(10,8,8);
a = double((a>0));
figure
imshow(a);
title('a')

b = zeros(m,n);
b(1:1:m,1:1:n/2) = 1;

figure
imshow(b);
title('b')

k=6;

figure
imhist(a(:), k)
title('a hist')

figure
imhist(b(:), k)
title('b hist')

a_blured = conv2(a, 1/9*[1,1,1;1,1,1;1,1,1],'same');
figure
imshow(a_blured);
title('a blured')

figure
imhist(a_blured(:), k)
title('a blured hist')

b_blured = conv2(b, 1/9*[1,1,1;1,1,1;1,1,1],'same');
figure
imshow(b_blured);
title('b blured')

figure
imhist(b_blured(:), k)
title('b blured hist')