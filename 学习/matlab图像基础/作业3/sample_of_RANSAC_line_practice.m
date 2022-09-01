clear
close all
clc

%% ����60������㣬Ȼ�����11�����ֱ�ߣ����ҵ��˳��
Points = rand(60,2);  
line = 0:0.1:1;
y = -0.5 * line + 0.8 + (rand(1,11)-0.5)/50;  % try

Points = [Points; cat(1, line, y)'];
scatter(Points(:,1), Points(:,2), 10, 'k', 'filled');
hold on
grid on
daspect([1 1 1]);

Points(:,3) = rand(size(Points,1), 1);
Points = sortrows(Points, 3);

X = Points(:, 1);
Y = Points(:, 2);

%% ����1000��
n = 1000; % try
tol = 0.02; % �ݲ�ֵ
for i = 1 : n
    choose = randperm(length(X));  % �����������������
    choose = choose(1:2);   % ���ѡȡ2��������
    choose_x = X(choose);
    choose_y = Y(choose);
    
    % 1��������2�������㣬����ֱ�߷��̣�����ɣ�������
    t = polyfit(choose_x, choose_y, 1);
    
    % 2�������ݲ�ֵtol�����ֱ�߷��������ݲ������ͳ�������ݲ���ڵĵ�ĸ���������ɣ�������
    all_distance = abs(t(1)*X-Y+t(2))/sqrt(t(1)^2+(-1)^2);
    choose = all_distance < tol;
    find_t(i,:) = t;
    choose_num(i) = sum(choose);
    choose_point{i} = choose;
   
end

%%3�������������ҳ���Ч�������������ݲ��������ʾ������Ӧ����Ч�����㣬�Լ���Ӧ��ֱ�ߣ�����ɣ�������
[m,index] = max(choose_num);
t = find_t(index, :);
choose = choose_point{index};
choose_x = X(choose);
choose_y = Y(choose);


plot(choose_x, choose_y, 'b*', choose_x, polyval(t,choose_x), 'r-');
legend('���е�', '�ڵ�', '�������ֱ��')
hold on
grid on
daspect([1 1 1]);
