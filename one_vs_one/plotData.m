function plotData(X, y)
% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);
posi = find(y==2);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 1, 'MarkerSize', 7)
hold on;
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
hold on
plot(X(posi, 1), X(posi, 2),'r.');
hold off;
end
