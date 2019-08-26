%模拟退火算法
distance = xlsread('D.xls'); %距离矩阵
T = 100; %初始温度
a = 0.99; %温度下降速率
e = 3; %终止温度 349次循环
num = 10000; %每次退火运算次数
L = 108;
S0 = [1,1+randperm(L-2),L]; %初始解
sum = 0;
for j = 1:L-1
    sum = sum + distance(S0(j),S0(j+1));
end
% S_current = S0; %当前解
% Length_current = sum; %当前路径长度
S_best = S0; %最优路径,初值为初始路径
Length_best = sum; %最优路径长度，初值为初始路径长度
while(T>e)
    for i = 1:num
        %获得新解
        c1 = 0; c2 = 0;
        while(c1 >= c2)
            c1 = 2+floor((L-3)*rand(1));
            c2 = 2+floor((L-3)*rand(1));
        end
        dist = distance(S0(c1),S0(c2))+distance(S0(c1 +1),S0(c2 +1))-distance(S0(c1),S0(c1 +1))-distance(S0(c2),S0(c2 +1));
        for j = c1+1:c2-1
            dist = dist + distance(S0(j+1),S0(j))-distance(S0(j),S0(j+1));
        end %求两条路径距离差
        if dist<0 %无条件接受新解
           S0(c1+1:c2)=S0(c2:-1:c1+1);
           sum = sum + dist;
        elseif exp(-dist/T)>rand(1) %有条件接受新解
           S0(c1+1:c2)=S0(c2:-1:c1+1);
           sum = sum + dist;
        end
        if Length_best > sum %当前解优于最优解
            S_best = S0;
            Length_best = sum;
        end
    end
    T = a*T; %退火
end
% %获得较优初始解
% sum = inf;
% for i = 1:1000
%     temp = 0;
%     S = [1,1+randperm(L-2),L];
%     for j = 1:L-1
%         temp = temp+distance(S(j),S(j+1));
%     end
%     if temp<sum
%         sum = temp;
%         S0 = S;
%     end
% end