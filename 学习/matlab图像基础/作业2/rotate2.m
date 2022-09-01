clc;
clear;
close all;
degree = 30;
img = imread('../images/ford.png');
figure;
imshow(img);
title('灰度图');
[h, w, c] = size(img);
% 新图形的高和宽
h_new = round(h*cosd(degree) + w*sind(degree)) + 1;
w_new = round(w*cosd(degree) + h*sind(degree)) + 1;
% 创建空图像
img_rotate = uint8(zeros(h_new, w_new, c));
% 创建坐标变换矩阵
trans1 = [ 1 0 0; 0 -1 0; -0.5*w 0.5*h 1];
trans2 = [cosd(degree), sind(degree), 0;
          -sind(degree), cosd(degree), 0;
          0, 0, 1];
trans3 = [1 0 0; 0 -1 0; 0.5*w_new 0.5*h_new 1];
% 创建坐标逆变换矩阵
inv_trans1 = [1 0 0; 0 -1 0; -0.5*w_new 0.5*h_new 1];
inv_trans2 = [cosd(degree), -sind(degree), 0;
              sind(degree), cosd(degree), 0;
              0, 0, 1];
inv_trans3 = [1 0 0; 0 -1 0; 0.5*w 0.5*h 1];
%计算每个像素点绕原点旋转后在新图像上的位置。
for j = 1: h
    for i = 1: w
        dst_point = [i j 1]*trans1*trans2*trans3;
        img_rotate(round(dst_point(1,2)), round(dst_point(1,1)), :) = img(j, i, :);
    end
end
figure;
imshow(img_rotate);
title('未插值');

for m = 1:h_new
    for n = 1: w_new
        src_point = [n m 1]*inv_trans1*inv_trans2*inv_trans3;
        % 判断该点是否在原图内，在的话进行双线性内插
        if (src_point(1,1)>=1 && src_point(1,1)<=w-1 && src_point(1,2)>=1 && src_point(1,2)<=h-1)
            % img_rotate(m, n) = img(round(src_point(1,2)), round(src_point(1,1)));
        % p1坐标
        p1_x = floor(src_point(1,1));
        p1_y = floor(src_point(1,2));
        % delta
        dx = src_point(1,1) - p1_x;
        dy = src_point(1,2) - p1_y;
        % 4邻点权重
        w1 = (1 - dx) * (1 - dy);
        w2 = dx * (1 - dy);
        w3 = (1 - dx) * dy;
        w4 = dx * dy;
        % 获取4邻点像素值
        p1 = img(p1_y, p1_x, :);
        p2 = img(p1_y, p1_x + 1, :);
        p3 = img(p1_y + 1, p1_x, :);
        p4 = img(p1_y + 1, p1_x + 1, :);
        img_rotate(m, n, :) = w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4;%公式
        end
    end
end
figure;
imshow(img_rotate);
title('双线性插值');



