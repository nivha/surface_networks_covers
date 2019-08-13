function [points] = points_on_polygon(number_of_points)
points = zeros(number_of_points, 2);
for index = 1:number_of_points
    points(index,:) = [cos(2*pi*index/number_of_points) sin(2*pi*index/number_of_points)];
end
end

