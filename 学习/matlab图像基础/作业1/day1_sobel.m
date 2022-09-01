img = imread('house.png'); % 读取图像转为灰度
img = rgb2gray(img);

[high,width] = size(img);   % 获得图像的高度和宽度
pix = double(img);
img_Gx = uint8(zeros(high, width));
img_Gy = uint8(zeros(high, width));
img_out1 = uint8(zeros(high, width));
img_out2 = uint8(zeros(high, width));
for i = 2:high - 1   %sobel边缘检测
    for j = 2:width - 1
        Gx = (-pix(i-1, j-1) - 2*pix(i, j-1) - pix(i+1, j-1) + pix(i-1, j+1) + 2*pix(i, j+1) + pix(i+1, j));
        Gy = (-pix(i-1, j-1) - 2*pix(i-1, j) - pix(i-1, j+1) + pix(i+1, j-1) + 2*pix(i+1, j) + pix(i+1, j+1));
        G_sqrt = sqrt(Gx^2+Gy^2);
        G_abs = abs(Gx) + abs(Gy);
        img_Gx(i, j) = Gx;
        img_Gy(i, j) = Gy;
        img_out1(i,j) = G_sqrt;
        img_out2(i,j) = G_abs;
    end
end 
% 显示灰度图
figure('Name', 'img_gray');
imshow(img);
title('imgGray');
% 显示x梯度和y梯度边缘检测图
figure('name', 'gradx & grady');
subplot(121);
imshow(img_Gx);
title('gradx');
subplot(122);
imshow(img_Gy);
title('grady');
% 显示开平方和绝对值的边沿检测图
figure('name', 'sqrt & abs');
subplot(121);
imshow(img_out1);
title('gradSqrt');
subplot(122);
imshow(img_out2);
title('gradAbs');

% 对原图边界进行扩充
img_padding = padarray(img,[1,1],"symmetric","both");
[h, w] = size(img_padding);
pix2 = double(img_padding);
img_Gx2 = uint8(zeros(h, w));
img_Gy2 = uint8(zeros(h, w));
img_out12 = uint8(zeros(h, w));
img_out22 = uint8(zeros(h, w));
for i = 2:h-1   %sobel边缘检测
    for j = 2:w-1
        Gx2 = (-pix2(i-1, j-1) - 2*pix2(i, j-1) - pix2(i+1, j-1) + pix2(i-1, j+1) + 2*pix2(i, j+1) + pix2(i+1, j));
        Gy2 = (-pix2(i-1, j-1) - 2*pix2(i-1, j) - pix2(i-1, j+1) + pix2(i+1, j-1) + 2*pix2(i+1, j) + pix2(i+1, j+1));
        G_sqrt = sqrt(Gx2^2+Gy2^2);
        G_abs = abs(Gx2) + abs(Gy2);
        img_out12(i,j) = G_sqrt;
        img_out22(i,j) = G_abs;
    end
end 
% 恢复原图大小
img_out12(all(img_out12==0,2),:) = [];
img_out12(:,all(img_out12==0,1)) = [];
img_out22(all(img_out12==0,2),:) = [];
img_out22(:,all(img_out12==0,1)) = [];
% 显示padding后的边缘检测图像
figure('name', 'padding: sqrt & abs');
subplot(121);
imshow(img_out12);
title('gradSqrt');
subplot(122);
imshow(img_out22);
title('gradAbs');

