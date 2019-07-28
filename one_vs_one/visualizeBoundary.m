function visualizeBoundary(X, y, model)
% Plot the training data on top of the boundary
plotData(X, y)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals1 = zeros(size(X1));
vals2 = zeros(size(X1));
vals3 = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals1(:, i) = predict(model.BinaryLearners{1}, this_X);
   vals2(:, i) = predict(model.BinaryLearners{2}, this_X);
   vals3(:, i) = predict(model.BinaryLearners{3}, this_X);
end

% Plot the SVM boundary
hold on
contour(X1, X2, vals1, [0.5 0.5], 'b');
hold on
contour(X1, X2, vals2, [0.5 0.5], 'r');
hold on
contour(X1, X2, vals3, [0.5 0.5], 'g');
hold off;

end
