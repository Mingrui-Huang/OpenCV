clc;
clear;
close all;
img = imread('../images/ford.png');
img = rgb2gray(img);
[h, w] = size(img); % 得到原图像的宽高
n = 0.7;
w_new = floor(w * n); % floor向下取整
h_new = floor(h * n);
%% 最近邻法
img_near = uint8(zeros(h_new, w_new));
for i = 1: h_new
    for j = 1: w_new
        x = round(j/n); % 缩放后的图像坐标在原图像处的位置
        y = round(i/n);
        img_near(i,j) = img(y,x); %赋值
    end
end
figure;
imshow(img);
title('原始灰度');
figure;
imshow(img_near);
title('最近邻法');
%% 双线性插值法
img_bilinear = uint8(zeros(h_new, w_new));
for i = 1: h_new
    for j = 1: w_new

        x = j/n; % 缩放后的图像坐标在原图像处的位置
        y = i/n;

        % p1坐标
        p1_x = floor(x);
        p1_y = floor(y);
        % delta
        dx = x - p1_x;
        dy = y - p1_y;
        % 4邻点权重
        w1 = (1 - dx) * (1 - dy);
        w2 = dx * (1 - dy);
        w3 = (1 - dx) * dy;
        w4 = dx * dy;
        % 获取4邻点
        p1 = img(p1_y, p1_x);
        p2 = img(p1_y, p1_x + 1);
        p3 = img(p1_y + 1, p1_x);
        p4 = img(p1_y + 1, p1_x + 1);
        img_bilinear(i, j) = w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4;%公式
    end
end
figure;
imshow(img_bilinear);
title('双线性插值');

