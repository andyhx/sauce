function [ GradientX, GradientY ] = hog_gradient( Image )

GradX = [1; 0; -1];
GradY = GradX';

GradientX = conv2(double(Image), GradX, 'same');
GradientY = conv2(double(Image), GradY, 'same');

end

