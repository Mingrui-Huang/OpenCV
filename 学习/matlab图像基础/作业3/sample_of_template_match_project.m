clear all;
close all;

I= imread('../images/go.png'); 
imshow(I);
impixelinfo;
img=rgb2gray(I);
black = img(250:283,343:376);
white = img(50:83, 78:111);
[m,n] = size(img);
img_out1=uint8(zeros(m, n));  % 结果矩阵初始化
img_out2=uint8(zeros(m, n));

img_data = double(img);   % 数据类型转换
black_data = double(black);  % 数据类型转换
white_data = double(white);

for j= 1: m-34+1
    for i= 1: n-34+1
        %找黑棋
        window_data = img_data(j:j+34-1,i:i+34-1);% 确定 img_data 在当前循环对应的窗口坐标；  
        black_abs_data = abs(window_data-black_data);   % 提取出 window_data窗口 与 black_data 相减，取绝对值，； 
        img_out1(j:j+33,i:i+33) = sum(black_abs_data(:))/1000; % 把所有元素相加求sum, 并赋值给 img_out 的相应格子。
        %找白棋
        white_abs_data = abs(window_data-white_data);
        img_out2(j:j+33,i:i+33) = sum(white_abs_data(:))/1000; 
   
    end
end
figure;
imshow(img);
impixelinfo;
title('灰度图');
figure;
imshow(img_out1);
title('找黑棋')
figure;
imshow(img_out2);
title('找白棋')
figure;
imshow(img);
hold on;
impixelinfo;
[r,c] = find(img_out1 < 40);
plot(c+15, r+15, 'g*');
[r,c] = find(img_out2 < 30);
plot(c+15, r+15, 'r+');
title('黑棋位置和白棋位置');
legend('黑棋','白棋');

        