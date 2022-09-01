clear all;
close all;

I= imread('../images/go.png'); 
imshow(I);
impixelinfo;
img=rgb2gray(I);
black = img(250:283,343:376);
white = img(50:83, 78:111);
[m,n] = size(img);
img_out1=uint8(zeros(m, n));  % ��������ʼ��
img_out2=uint8(zeros(m, n));

img_data = double(img);   % ��������ת��
black_data = double(black);  % ��������ת��
white_data = double(white);

for j= 1: m-34+1
    for i= 1: n-34+1
        %�Һ���
        window_data = img_data(j:j+34-1,i:i+34-1);% ȷ�� img_data �ڵ�ǰѭ����Ӧ�Ĵ������ꣻ  
        black_abs_data = abs(window_data-black_data);   % ��ȡ�� window_data���� �� black_data �����ȡ����ֵ���� 
        img_out1(j:j+33,i:i+33) = sum(black_abs_data(:))/1000; % ������Ԫ�������sum, ����ֵ�� img_out ����Ӧ���ӡ�
        %�Ұ���
        white_abs_data = abs(window_data-white_data);
        img_out2(j:j+33,i:i+33) = sum(white_abs_data(:))/1000; 
   
    end
end
figure;
imshow(img);
impixelinfo;
title('�Ҷ�ͼ');
figure;
imshow(img_out1);
title('�Һ���')
figure;
imshow(img_out2);
title('�Ұ���')
figure;
imshow(img);
hold on;
impixelinfo;
[r,c] = find(img_out1 < 40);
plot(c+15, r+15, 'g*');
[r,c] = find(img_out2 < 30);
plot(c+15, r+15, 'r+');
title('����λ�úͰ���λ��');
legend('����','����');

        