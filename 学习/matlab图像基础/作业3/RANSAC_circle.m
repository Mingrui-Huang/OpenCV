clear
close all
clc

%% 生成80个随机点，然后添加20个点圆
Points = 2*(rand(80,2) - 0.5);
scatter(Points(:,1),Points(:,2), 10, 'ko', 'filled');
hold on;
daspect([1 1 1]);

theta = linspace(0, 2*pi, 20);
x = 0.6*sin(theta) + 0.3;
y = 0.6*cos(theta) + 0.2;

scatter(x, y, 10, 'ko', 'filled');
savefig('./scatter_circle');

Points = [Points; cat(1, x, y)'];
X = Points(:, 1);
Y = Points(:, 2);

%% 尝试1000次
n = 1000; % try
tol = 0.02; % 容差值
for i = 1 : n
    choose = randperm(length(X));  % 所有样本点随机排序
    choose = choose(1:3);   % 随机选取3个样本点
    choose_x = X(choose);
    choose_y = Y(choose);
    
    [x0, y0, R] = circlefit(choose_x, choose_y);
    % 点到圆心的距离
    all_distance = sqrt(abs((X-x0).^2 + (Y-y0).^2));
    choose = abs(all_distance - R) < tol;
    
    choose_num(i) = sum(choose);
    choose_point{i} = choose;
    r(i) = R;
    center_x(i) = x0;
    center_y(i) = y0;
end

[m_num,index] = max(choose_num);
choose = choose_point{index};
R = r(index);
center_x = center_x(index);
center_y = center_y(index);
choose_x = X(choose);
choose_y = Y(choose);

% 绘制结果
openfig('./scatter_circle.fig');
plot(choose_x, choose_y, 'b*');
alpha=linspace(0,2*pi,100);
plot(center_x+R*cos(alpha),center_y+R*sin(alpha),'g-');
plot(center_x, center_y, 'r+');


%% 最小二乘法拟合圆
function [xc,yc,R]=circlefit(x,y)
% CIRCLEFIT fits a circle in x,y plane
% x^2+y^2+a(1)*x+a(2)*y+a(3)=0
n=length(x);
xx=x.*x;
yy=y.*y;
xy=x.*y;

A=[sum(x) sum(y) n;sum(xy) sum(yy) sum(y);sum(xx) sum(xy) sum(x)];
B=[-sum(xx+yy);-sum(xx.*y+yy.*y);-sum(xx.*x+xy.*y)];
a=A\B;
xc = -0.5*a(1);
yc = -0.5*a(2);
R = sqrt(-(a(3)-xc^2-yc^2));
end