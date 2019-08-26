global popsize 
popsize = 100;
global genelength
genelength = 10;
generation_max = 12;
crossover_p = 0.9;
mutation_p = 0.09;

%产生初始种群
population = randi([0,1],popsize,genelength);
%

[value,cumsump] = fitness(population);
%best为最优解 best_solution为最优染色体
[best,index] = max(value);
best_solution = population(index,:);
%迭代
generation = 1;
while generation<generation_max
    descendant = zeros(popsize,genelength);
    for i = 1:2:popsize
        sel = select(cumsump);
        %交叉
        individual = crossover(population,crossover_p,sel);
        %变异
        descendant(i,:) = mutation(individual(1,:),mutation_p);
        descendant(i+1,:) = mutation(individual(2,:),mutation_p);
    end
    population = descendant;
    [value,cumsump] = fitness(population);
    [new,new_solution] = max(value);
    if new>best
        best = new;
        best_solution = population(new_solution,:);
    end
    generation = generation + 1;
end
%可视化



%解码(二进制到十进制)
function x = decode(population)
global popsize
global genelength
x = zeros(popsize,1);
for i = 1:popsize
    for j = 0:(genelength-1)
        x(i) = x(i) + power(2,j) * population(i,genelength-j);
    end
end
end
%选择优良个体进行交叉互换(轮盘选择法)
function sel = select(cumsump)
sel = zeros(2,1);
for i = 1:2
    a = rand;
    [sel(i),~] = find(cumsump>a,1);
end
end
%交叉(一个位点)
function x = crossover(population,crossover_p,sel)
global genelength
p = Cro_Mut(crossover_p);
if p == 1
    position = round(genelength * rand);
    x(1,:) = [population(sel(1),1:position),population(sel(2),position+1:end)];
    x(2,:) = [population(sel(2),1:position),population(sel(1),position+1:end)];
else
    x(1,:) = population(sel(1),:);
    x(2,:) = population(sel(2),:);
end
end
%变异(一个位点)
function x = mutation(individual,mutation_p)
global genelength
x = individual;
p = Cro_Mut(mutation_p);
if p == 1
    position = round((genelength - 1) * rand) + 1;
    x(position) = abs(individual(position)-1);
end
end
%判断是否发生交叉互换或变异
function x = Cro_Mut(p)
if p > rand
    x = 1;
else
    x = 0;
end
end
%适应度函数
function [value, cumsump] = fitness(population)
x = decode(population);
value = f(x);
p = value./sum(value);
cumsump = cumsum(p);
end
%目标函数
function y = f(x)
y = 200 * exp(-0.05 .* x) .* sin(x);
end