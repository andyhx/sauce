clear all;
close all;

PositiveSetTrain = 'F:/inz/priv/sanity_test/train/pos';
NegativeSetTrain = 'F:/inz/priv/sanity_test/train/neg';

PositivesT = dir(PositiveSetTrain);

for i=3:length(PositivesT)
    ImageFile = [PositiveSetTrain, '/', PositivesT(i, 1).name];
    Image = imread(ImageFile);
    ImageGrayscale = rgb2gray(Image);
    
    Covariances = cov_features(ImageGrayscale, 16);
    
    Food(i-2, :) = Covariances(:);
    Class(i-2, 1) = 1;

    fprintf('%s\n', ImageFile);
end

NegativesT = dir(NegativeSetTrain);
N = length(PositivesT)-2;

for i=3:length(NegativesT)
    ImageFile = [NegativeSetTrain, '/', NegativesT(i, 1).name];
    Image = imread(ImageFile);
    ImageGrayscale = rgb2gray(Image);
    
    Covariances = cov_features(ImageGrayscale, 16);
    
    Food(N+i-2, :) = Covariances(:);
    Class(N+i-2, 1) = 0;
    fprintf('%s\n', ImageFile);
end

options = optimset('maxiter', 10000);
SVM=svmtrain(Food,Class,'quadprog_opts', options);
