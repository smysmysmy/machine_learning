%ģ���˻��㷨
distance = xlsread('D.xls'); %�������
T = 100; %��ʼ�¶�
a = 0.99; %�¶��½�����
e = 3; %��ֹ�¶� 349��ѭ��
num = 10000; %ÿ���˻��������
L = 108;
S0 = [1,1+randperm(L-2),L]; %��ʼ��
sum = 0;
for j = 1:L-1
    sum = sum + distance(S0(j),S0(j+1));
end
% S_current = S0; %��ǰ��
% Length_current = sum; %��ǰ·������
S_best = S0; %����·��,��ֵΪ��ʼ·��
Length_best = sum; %����·�����ȣ���ֵΪ��ʼ·������
while(T>e)
    for i = 1:num
        %����½�
        c1 = 0; c2 = 0;
        while(c1 >= c2)
            c1 = 2+floor((L-3)*rand(1));
            c2 = 2+floor((L-3)*rand(1));
        end
        dist = distance(S0(c1),S0(c2))+distance(S0(c1 +1),S0(c2 +1))-distance(S0(c1),S0(c1 +1))-distance(S0(c2),S0(c2 +1));
        for j = c1+1:c2-1
            dist = dist + distance(S0(j+1),S0(j))-distance(S0(j),S0(j+1));
        end %������·�������
        if dist<0 %�����������½�
           S0(c1+1:c2)=S0(c2:-1:c1+1);
           sum = sum + dist;
        elseif exp(-dist/T)>rand(1) %�����������½�
           S0(c1+1:c2)=S0(c2:-1:c1+1);
           sum = sum + dist;
        end
        if Length_best > sum %��ǰ���������Ž�
            S_best = S0;
            Length_best = sum;
        end
    end
    T = a*T; %�˻�
end
% %��ý��ų�ʼ��
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